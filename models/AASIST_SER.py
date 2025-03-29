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
                 freeze_ser=True, n_mels=40, sample_rate=16000):
        super().__init__()
        
        # 1. Загрузка AASIST (полная заморозка)
        self.aasist = Model(aasist_config)
        self.aasist.load_state_dict(torch.load(aasist_config["aasist_path"]))
        self.aasist.eval()
        for p in self.aasist.parameters():
            p.requires_grad = False

        # 2. Загрузка SER (полная заморозка)
        self.ser = acrnn()
        self.ser.load_state_dict(torch.load(ser_config["ser_path"]))
        self.ser.eval()
        for p in self.ser.parameters():
            p.requires_grad = False

        # 3. Параметры для Mel-спектрограмм
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.frame_length = int(0.025 * sample_rate)  # 25ms
        self.frame_step = int(0.01 * sample_rate)     # 10ms

        # 4. Определение размерностей
        self.aasist_feat_dim = 5 * aasist_config["gat_dims"][1]
        self.ser_feat_dim = 256  # Размерность выхода SER

        # 5. Простой линейный классификатор
        self.classifier = nn.Linear(self.aasist_feat_dim + self.ser_feat_dim, 2)

    def extract_mel_features(self, x):
        """Извлечение Mel-фич с python_speech_features"""
        batch_features = []
        for waveform in x:
            if torch.is_tensor(waveform):
                waveform = waveform.cpu().numpy()
            
            mel_spec = ps.logfbank(waveform, 
                                  samplerate=self.sample_rate, 
                                  nfilt=self.n_mels,
                                  winlen=0.025, 
                                  winstep=0.01)
            delta1 = ps.delta(mel_spec, 2)
            delta2 = ps.delta(delta1, 2)
            
            # Обрезка до 300 кадров
            features = np.stack([mel_spec[:300], 
                                delta1[:300], 
                                delta2[:300]], axis=0)
            batch_features.append(features)
        
        return torch.from_numpy(np.array(batch_features)).float()

    def forward(self, x, Freq_aug=False):
        # 1. Фичи AASIST
        aasist_feat, _ = self.aasist(x, Freq_aug=Freq_aug)
        
        # 2. Фичи SER (в режиме inference)
        with torch.no_grad():
            x_np = x.cpu().numpy() if x.is_cuda else x.numpy()
            mel_features = self.extract_mel_features(x_np).to(x.device)
            ser_feat = self.ser(mel_features)

        # 3. Конкатенация + классификация
        combined = torch.cat([aasist_feat, ser_feat], dim=1)
        output = self.classifier(combined)
        
        return combined, output  # Возвращаем фичи и логиты

    @property
    def device(self):
        return next(self.parameters()).device