import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import python_speech_features as ps
import numpy as np
from .AASIST import Model
from .ACRNN import acrnn

class AASISTWithEmotion(nn.Module):
    def __init__(self, aasist_config, ser_config, 
                 freeze_ser=True, fusion_dim=256,
                 n_mels=40, sample_rate=16000):
        super().__init__()
        
        # 1. Инициализация AASIST
        self.aasist = Model(aasist_config)
        state_dict = torch.load(aasist_config["aasist_path"], map_location='cpu')
        self.aasist.load_state_dict(state_dict)

        # 2. Инициализация SER модели (ACRNN) - оставляем оригинальную 3-канальную версию
        self.ser = acrnn()
        ser_state_dict = torch.load(ser_config["ser_path"], map_location='cpu')
        self.ser.load_state_dict(ser_state_dict)
        
        if freeze_ser:
            for param in self.ser.parameters():
                param.requires_grad = False

        # 3. Параметры для Mel-спектрограмм
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.frame_length = int(0.025 * sample_rate)  # 25ms
        self.frame_step = int(0.01 * sample_rate)     # 10ms

        # 4. Слои для обработки выхода SER модели
        self.ser_bn = nn.BatchNorm1d(num_features=50)
        self.ser_selu = nn.SELU(inplace=True)

        # 5. Определение размерностей
        self.aasist_feat_dim = 5 * aasist_config["gat_dims"][1]
        self.ser_feat_dim = 256  # Размерность после max_pool2d((3,4))

        # 6. Слои для объединения признаков
        self.feature_norm = nn.LayerNorm(self.aasist_feat_dim + self.ser_feat_dim)
        self.fc = nn.Linear(self.aasist_feat_dim + self.ser_feat_dim, fusion_dim)
        self.fusion = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, 2)
        )

    def extract_mel_features(self, x):
        """Аналог оригинального метода extract_mel с python_speech_features"""
        # x: [batch, time] numpy array
        batch_features = []
        for waveform in x:
            # Конвертируем в numpy если нужно
            if torch.is_tensor(waveform):
                waveform = waveform.cpu().numpy()
            
            # Вычисляем 3 канала
            mel_spec = ps.logfbank(waveform, samplerate=self.sample_rate, nfilt=self.n_mels,
                                 winlen=0.025, winstep=0.01)
            delta1 = ps.delta(mel_spec, 2)
            delta2 = ps.delta(delta1, 2)
            
            # Обрезаем до 300 кадров как в оригинале
            mel_spec = mel_spec[:300]
            delta1 = delta1[:300]
            delta2 = delta2[:300]
            
            # Собираем 3 канала
            features = np.stack([mel_spec, delta1, delta2], axis=0)  # [3, 300, n_mels]
            batch_features.append(features)
        
        return torch.from_numpy(np.array(batch_features)).float()

    def forward(self, x, Freq_aug=False):
        # x: [batch, time] аудио waveform
        if x.dim() == 3:
            x = x.squeeze(1)

        # 1. Извлечение признаков из AASIST
        aasist_last_hidden, aasist_output = self.aasist(x, Freq_aug=Freq_aug)

        # 2. Подготовка 3-канальных Mel-фич для SER
        with torch.no_grad():
            # Конвертируем в numpy для python_speech_features
            x_np = x.cpu().numpy() if x.is_cuda else x.numpy()
            
            # Получаем 3-канальные фичи [batch, 3, 300, n_mels]
            mel_features = self.extract_mel_features(x_np).to(x.device)
            
            # Извлекаем SER-эмбеддинги
            ser_features = self.ser(mel_features)

        # 3. Слияние признаков
        print("AASIST hidden shape:", aasist_last_hidden.shape)
        print("SER features shape:", ser_features.shape)

        combined = torch.cat([aasist_last_hidden, ser_features], dim=1)
        combined = self.feature_norm(combined)
        
        # 4. Классификация
        hidden = self.fc(combined)
        output = self.fusion(hidden)

        return hidden, output

    @property
    def device(self):
        return next(self.parameters()).device