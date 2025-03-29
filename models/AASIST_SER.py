import torch
import torch.nn as nn
import torchaudio
from .AASIST import Model
from .ACRNN import acrnn

class AASISTWithEmotion(nn.Module):
    def __init__(self, aasist_config, ser_config, n_mels=40, sample_rate=16000):
        super().__init__()
        
        # 1. Загрузка и заморозка AASIST
        self.aasist = Model(aasist_config)
        self.aasist.load_state_dict(torch.load(aasist_config["aasist_path"]))
        self.aasist.eval()
        for p in self.aasist.parameters():
            p.requires_grad = False

        # 2. Загрузка и заморозка SER
        self.ser = acrnn()
        self.ser.load_state_dict(torch.load(ser_config["ser_path"]))
        self.ser.eval()
        for p in self.ser.parameters():
            p.requires_grad = False

        # 3. Инициализация Mel-спектрограмм
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            win_length=int(0.025*sample_rate),
            hop_length=int(0.01*sample_rate),
            n_fft=int(0.025*sample_rate)
        )

        # 4. Размерности
        self.aasist_feat_dim = 5 * aasist_config["gat_dims"][1]
        self.ser_feat_dim = 256

        # 5. Классификатор с нормализацией
        self.feature_norm = nn.LayerNorm(self.aasist_feat_dim + self.ser_feat_dim)
        self.classifier = nn.Linear(self.aasist_feat_dim + self.ser_feat_dim, 2)

    def extract_mel_features(self, x):
        """Оптимизированное извлечение Mel-фич на GPU"""
        # x: (batch, time)
        mels = self.mel_transform(x).log()  # (batch, n_mels, time)
        delta1 = torchaudio.functional.compute_deltas(mels)
        delta2 = torchaudio.functional.compute_deltas(delta1)
        features = torch.stack([mels, delta1, delta2], dim=1)  # (batch, 3, n_mels, time)
        return features[:, :, :, :300]  # Обрезка

    def forward(self, x, Freq_aug=False):
        # 1. Получаем фичи AASIST
        with torch.no_grad():
            aasist_feat, _ = self.aasist(x, Freq_aug=Freq_aug)
            
            # 2. Получаем фичи SER (полностью на GPU)
            mel_features = self.extract_mel_features(x)
            ser_feat = self.ser(mel_features)

        # 3. Конкатенация и классификация
        combined = torch.cat([aasist_feat, ser_feat], dim=1)
        combined = self.feature_norm(combined)
        output = self.classifier(combined)
        
        return combined, output

    @property
    def device(self):
        return next(self.parameters()).device