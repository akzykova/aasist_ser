import torch
import torch.nn as nn
import torchaudio
from .AASIST import Model
from transformers import HubertModel, AutoConfig

class AASISTWithEmotion(nn.Module):
    def __init__(self, aasist_config, ser_model_name="facebook/hubert-base-ls960", freeze_ser=True, fusion_dim=256):
        super().__init__()
        
        # 1. Инициализация AASIST
        self.aasist = Model(aasist_config)
        if "aasist_path" in aasist_config:
            state_dict = torch.load(aasist_config["aasist_path"], map_location='cpu')
            self.aasist.load_state_dict(state_dict)
        
        # 2. Инициализация HuBERT
        self.ser_model = HubertModel.from_pretrained(ser_model_name)
        self.ser_model.eval()
        if freeze_ser:
            for param in self.ser_model.parameters():
                param.requires_grad = False
        
        # 3. Динамическое определение размерностей
        self.aasist_feat_dim = self._get_aasist_output_dim(aasist_config)
        ser_config = AutoConfig.from_pretrained(ser_model_name)
        self.ser_feat_dim = ser_config.hidden_size
        
        # 4. Слои для объединения признаков
        self.feature_norm = nn.LayerNorm(self.aasist_feat_dim + self.ser_feat_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.aasist_feat_dim + self.ser_feat_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, 2)
        )
        
        # 5. Ресемплинг
        self.resample = torchaudio.transforms.Resample(
            orig_freq=16000,
            new_freq=16000
        )

    def _get_aasist_output_dim(self, config):
        """Динамически определяет размерность выхода AASIST"""
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, config["nb_samp"])
            output = self.aasist(dummy_input)
            return output.shape[-1]

    def forward(self, x, Freq_aug=False):
        # 1. Проверка и корректировка входных данных
        if x.dim() == 4:
            x = x.squeeze(1)  # [batch, 1, 1, samples] -> [batch, 1, samples]
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, samples] -> [batch, 1, samples]
        
        # 2. Получение признаков AASIST
        aasist_features = self.aasist(x, Freq_aug=Freq_aug)
        
        # 3. Подготовка и обработка в HuBERT
        x_hubert = x.squeeze(1)  # [batch, samples]
        with torch.set_grad_enabled(not all(not p.requires_grad for p in self.ser_model.parameters())):
            ser_features = self.ser_model(x_hubert).last_hidden_state.mean(dim=1)
        
        # 4. Слияние признаков
        combined = torch.cat([
            aasist_features,
            ser_features.to(aasist_features.device)
        ], dim=1)
        
        return None, self.fusion(self.feature_norm(combined))

    @property
    def device(self):
        return next(self.parameters()).device