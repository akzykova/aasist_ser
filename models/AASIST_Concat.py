import torch
import torch.nn as nn
import torch.nn.functional as F
from .AASIST import Model
from .ACRNN import acrnn

class Model(nn.Module):
    def __init__(self, aasist_config, ser_config):
        super().__init__()
        
        self.aasist = Model(aasist_config)
        self.aasist.load_state_dict(torch.load(aasist_config["aasist_path"]))
        for p in self.aasist.parameters():
            p.requires_grad = False

        self.ser = acrnn()
        self.ser.load_state_dict(torch.load(ser_config["ser_path"]))
        for p in self.ser.parameters():
            p.requires_grad = False

        self.aasist_feat_dim = 5 * aasist_config["gat_dims"][1]
        self.ser_feat_dim = 256
        self.test_linear = nn.Linear(self.aasist_feat_dim + self.ser_feat_dim, 2)

    def forward(self, x, x_emo, Freq_aug=False):
        with torch.no_grad():
            aasist_feat, _ = self.aasist(x, Freq_aug=Freq_aug)
            ser_feat = self.ser(x_emo)
        
        aasist_feat = F.normalize(aasist_feat, p=2, dim=1)
        ser_feat = F.normalize(ser_feat, p=2, dim=1)
        
        combined_feat = torch.cat([aasist_feat, ser_feat], dim=1)
        
        combined_feat = F.normalize(combined_feat, p=2, dim=1)
        
        out = self.test_linear(combined_feat)
        
        return combined_feat, out

    @property
    def device(self):
        return next(self.parameters()).device