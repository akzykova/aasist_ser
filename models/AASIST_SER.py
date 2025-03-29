import torch
import torch.nn as nn
from .AASIST import Model
from .ACRNN import acrnn  # Предполагается, что ACRNN модель определена где-то

class AASISTWithEmotion(nn.Module):
    def __init__(self, aasist_config, ser_config, 
                 freeze_ser=True, fusion_dim=256):
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

        # Добавляем слои для обработки выхода SER модели (как в оригинальном коде)
        self.ser_bn = nn.BatchNorm1d(num_features=50)
        self.ser_selu = nn.SELU(inplace=True)

        # 3. Определение размерностей
        self.aasist_feat_dim = 5 * aasist_config["gat_dims"][1]
        self.ser_feat_dim = 64  # Размерность после max_pool2d((3,4)) в оригинальном коде

        # 4. Слои для объединения признаков
        self.feature_norm = nn.LayerNorm(self.aasist_feat_dim + self.ser_feat_dim)

        self.fc = nn.Linear(self.aasist_feat_dim + self.ser_feat_dim, fusion_dim)
        self.fusion = nn.Sequential(
            self.fc,
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, 2)
        )

    def forward(self, x, Freq_aug=False):
        # 1. Извлечение признаков из AASIST
        aasist_features = self.aasist(x, Freq_aug=Freq_aug)

        # 2. Извлечение признаков из SER модели
        with torch.no_grad():
            ser_features = self.ser(x)
            ser_features = F.max_pool2d(ser_features, (3, 4))
            ser_features = self.ser_bn(ser_features)
            ser_features = self.ser_selu(ser_features)
            
            # Преобразуем к нужной размерности [batch, features]
            ser_features = ser_features.mean(dim=(2, 3))  # Адаптируйте под вашу архитектуру

        # 3. Слияние признаков
        combined = torch.cat([
            aasist_features,
            ser_features.to(aasist_features.device)
        ], dim=1)

        self.last_hidden_state = self.fc(combined)
        output = self.fusion(self.feature_norm(combined))

        return self.last_hidden_state, output

    @property
    def device(self):
        return next(self.parameters()).device