import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Model, AutoConfig
from .AASIST import Model

class AASISTWithEmotion(nn.Module):
    def __init__(self, aasist_config, wav2vec_model_name="facebook/wav2vec2-base-960h", freeze_wav2vec=True, fusion_dim=256):
        super().__init__()

        # 1. Инициализация AASIST
        self.aasist = Model(aasist_config)
        if "aasist_path" in aasist_config:
            state_dict = torch.load(aasist_config["aasist_path"], map_location='cpu')
            self.aasist.load_state_dict(state_dict)

        # 2. Инициализация Wav2Vec2
        self.wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_model_name)
        if freeze_wav2vec:
            for param in self.wav2vec.parameters():
                param.requires_grad = False

        # 3. Определение размерностей
        self.aasist_feat_dim = 5 * aasist_config["gat_dims"][1]
        wav2vec_config = AutoConfig.from_pretrained(wav2vec_model_name)
        self.wav2vec_feat_dim = wav2vec_config.hidden_size

        # 4. Слои для объединения признаков
        self.feature_norm = nn.LayerNorm(self.aasist_feat_dim + self.wav2vec_feat_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.aasist_feat_dim + self.wav2vec_feat_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, 2)
        )

    def forward(self, x, Freq_aug=False):
        # 1. Извлечение признаков из AASIST
        aasist_features = self.aasist(x, Freq_aug=Freq_aug)

        # 2. Подготовка аудио для Wav2Vec2 (убираем канал)
        x_wav2vec = x.squeeze(1)  # [batch, samples]

        # 3. Извлечение признаков из Wav2Vec2
        with torch.no_grad():
            wav2vec_features = self.wav2vec(x_wav2vec).last_hidden_state.mean(dim=1)

        # 4. Слияние признаков
        combined = torch.cat([
            aasist_features,
            wav2vec_features.to(aasist_features.device)
        ], dim=1)

        return None, self.fusion(self.feature_norm(combined))

    @property
    def device(self):
        return next(self.parameters()).device
