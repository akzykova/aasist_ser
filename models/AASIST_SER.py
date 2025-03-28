import torch
import torch.nn as nn
from .AASIST import Model
from transformers import HubertModel, AutoConfig

class AASISTWithEmotion(nn.Module):
    def __init__(self, aasist_config, ser_model_name="facebook/hubert-base-ls960", freeze_ser=True, fusion_dim=256):
        super().__init__()
        # Инициализация AASIST
        self.aasist = Model(aasist_config)  # Оригинальный AASIST
        if "aasist_path" in aasist_config:
            self.aasist.load_state_dict(torch.load(aasist_config["aasist_path"]))
        
        # Инициализация HuBERT
        self.ser_model = HubertModel.from_pretrained(ser_model_name)
        if freeze_ser:
            for param in self.ser_model.parameters():
                param.requires_grad = False
        
        # Определение размерностей
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, aasist_config["nb_samp"])
            aasist_feat_dim = self.aasist(dummy_input).shape[-1]
            ser_feat_dim = self.ser_model.config.hidden_size
        
        # Слой слияния признаков
        self.feature_norm = nn.LayerNorm(aasist_feat_dim + ser_feat_dim)
        self.fusion = nn.Sequential(
            nn.Linear(aasist_feat_dim + ser_feat_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, 2)
        )
        
        # Ресемплинг для HuBERT
        self.resample = torchaudio.transforms.Resample(
            orig_freq=16000,  # Будет установлено в forward
            new_freq=16000
        )

    def forward(self, x, Freq_aug=False, sample_rate=16000):
        # 1. Обработка в AASIST (с поддержкой Freq_aug)
        aasist_features = self.aasist(x, Freq_aug=Freq_aug)
        
        # 2. Подготовка аудио для HuBERT
        if sample_rate != 16000:
            self.resample.orig_freq = sample_rate
            x = self.resample(x)
        
        # 3. Извлечение эмоциональных признаков
        with torch.set_grad_enabled(not all(not p.requires_grad for p in self.ser_model.parameters())):
            ser_outputs = self.ser_model(x.squeeze(1))  # [batch, seq_len, features]
            ser_features = ser_outputs.last_hidden_state.mean(dim=1)  # [batch, features]
        
        # 4. Слияние признаков
        combined = torch.cat([aasist_features, ser_features], dim=1)
        normalized = self.feature_norm(combined)
        output = self.fusion(normalized)
        
        # Возвращаем в том же формате, что и оригинальный AASIST
        return None, output  # (features, logits)