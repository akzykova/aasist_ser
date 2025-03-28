import torch
import torch.nn as nn
from .AASIST import Model
from transformers import HubertModel, AutoConfig

class AASISTWithEmotion(nn.Module):
    def __init__(self, aasist_config, ser_model_name="facebook/hubert-large-lls60", freeze_ser=True, fusion_dim=256):
        super().__init__()
        # AASIST model
        print('Trying to upload AASIST')
        self.aasist = Model(aasist_config)
        if "aasist_path" in aasist_config:
            self.aasist.load_state_dict(torch.load(aasist_config["aasist_path"]))

        print('AASIST done')
        # SER model (HuBERT)
        self.ser_model = HubertModel.from_pretrained(ser_model_name)
        if freeze_ser:
            for param in self.ser_model.parameters():
                param.requires_grad = False
        
        # Feature fusion
        ser_config = AutoConfig.from_pretrained(ser_model_name)
        aasist_feat_dim = aasist_config.get('embd_dim', 256)  # default AASIST feature dim
        ser_feat_dim = ser_config.hidden_size  # HuBERT feature dim
        
        self.feature_norm = nn.LayerNorm(aasist_feat_dim + ser_feat_dim)
        self.fusion = nn.Sequential(
            nn.Linear(aasist_feat_dim + ser_feat_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, 2)
        )

    def forward(self, x):
        # AASIST features
        aasist_features = self.aasist(x)  # [batch, aasist_feat_dim]
        
        # SER features (HuBERT)
        with torch.set_grad_enabled(not self.ser_model.training):
            ser_outputs = self.ser_model(x)
            ser_features = ser_outputs.last_hidden_state.mean(dim=1)  # [batch, ser_feat_dim]
        
        # Feature fusion
        combined = torch.cat([aasist_features, ser_features], dim=1)  # [batch, aasist_feat_dim + ser_feat_dim]
        normalized = self.feature_norm(combined)
        output = self.fusion(normalized)
        
        return output