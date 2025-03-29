import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from .AASIST import Model
from .ACRNN import acrnn

class AASISTWithEmotion(nn.Module):
    def __init__(self, aasist_config, ser_config, 
                 freeze_ser=True, fusion_dim=256,
                 n_mels=64, sample_rate=16000):
        super().__init__()
        
        # 1. Инициализация AASIST
        self.aasist = Model(aasist_config)
        state_dict = torch.load(aasist_config["aasist_path"], map_location='cpu')
        self.aasist.load_state_dict(state_dict)

        # 2. Инициализация SER модели (ACRNN)
        self.ser = acrnn()
        ser_state_dict = torch.load(ser_config["ser_path"], map_location='cpu')
        self.ser.load_state_dict(ser_state_dict)
        
        if freeze_ser:
            for param in self.ser.parameters():
                param.requires_grad = False

        # 3. Преобразование в Mel-спектрограмму
        self.to_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=n_mels,
            power=2  # Используем power=2 для совместимости с AmplitudeToDB
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()

        # 4. Слои для обработки выхода SER модели
        self.ser_bn = nn.BatchNorm1d(num_features=50)
        self.ser_selu = nn.SELU(inplace=True)

        # 5. Определение размерностей
        self.aasist_feat_dim = 5 * aasist_config["gat_dims"][1]
        self.ser_feat_dim = 64  # Размерность после max_pool2d((3,4))

        # 6. Слои для объединения признаков
        self.feature_norm = nn.LayerNorm(self.aasist_feat_dim + self.ser_feat_dim)
        self.fc = nn.Linear(self.aasist_feat_dim + self.ser_feat_dim, fusion_dim)
        self.fusion = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, 2)
        )

    def forward(self, x, Freq_aug=False):
        # x: [batch, time] или [batch, channels, time]
        if x.dim() == 3:
            x = x.squeeze(1)  # Удаляем ось каналов, если есть

        # 1. Извлечение признаков из AASIST (работает с waveform)
        aasist_features = self.aasist(x, Freq_aug=Freq_aug)

        # 2. Преобразуем waveform в Mel-спектрограмму для SER
        with torch.no_grad():
            # Вычисляем спектрограмму и переводим в dB
            mel_spec = self.to_spectrogram(x)  # [batch, n_mels, time]
            log_mel_spec = self.to_db(mel_spec)
            
            # Добавляем ось каналов для conv2d: [batch, 1, n_mels, time]
            log_mel_spec = log_mel_spec.unsqueeze(1)
            
            # Извлекаем SER-эмбеддинги
            ser_features = self.ser(log_mel_spec)
            ser_features = F.max_pool2d(ser_features, (3, 4))
            ser_features = self.ser_bn(ser_features)
            ser_features = self.ser_selu(ser_features)
            ser_features = ser_features.mean(dim=(2, 3))  # [batch, ser_feat_dim]

        # 3. Слияние признаков
        combined = torch.cat([aasist_features, ser_features], dim=1)
        combined = self.feature_norm(combined)
        
        # 4. Классификация
        hidden = self.fc(combined)
        output = self.fusion(hidden)

        return hidden, output

    @property
    def device(self):
        return next(self.parameters()).device