import torch
import torch.nn as nn
import torch.nn.functional as F
from .AASIST import Model as AASISTModel
from .ACRNN import acrnn

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        aasist_config = config['aasist_config']
        ser_config = config['ser_config']

        self.aasist = AASISTModel(aasist_config)
        self.ser = acrnn()

        self.aasist_feat_dim = 5 * aasist_config["gat_dims"][1]
        self.ser_feat_dim = 256
        self.test_linear = nn.Linear(self.aasist_feat_dim + self.ser_feat_dim, 2)

    def forward(self, x, x_emo):
        x_emo = x_emo.permute(0, 2, 1, 3)
        with torch.no_grad():
            aasist_feat, _ = self.aasist(x)
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