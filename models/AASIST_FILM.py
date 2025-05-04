import torch
import torch.nn as nn
from .AASIST import Model
from .ACRNN import acrnn

class FiLMBlock(nn.Module):
    def __init__(self, sv_dim, cm_dim, hidden_dim=128):
        super().__init__()
        self.cm_ln = nn.LayerNorm(cm_dim)
        self.condition_to_gamma_beta = nn.Sequential(
            nn.Linear(cm_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )

        self.gamma_linear = nn.Linear(hidden_dim, sv_dim)
        self.beta_linear = nn.Linear(hidden_dim, sv_dim)
        
        self.sv_ln = nn.LayerNorm(sv_dim)

    def forward(self, e_sv, e_cm):
        e_cm_norm = self.cm_ln(e_cm)
        gamma_beta = self.condition_to_gamma_beta(e_cm_norm)
        gamma = self.gamma_linear(gamma_beta)
        beta = self.beta_linear(gamma_beta)
        
        e_sv_norm = self.sv_ln(e_sv)
        e_mod1 = gamma * e_sv_norm + beta

        return e_mod1



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

        self.film = FiLMBlock(self.aasist_feat_dim, self.ser_feat_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.aasist_feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x, x_emo, Freq_aug=False):
        with torch.no_grad():
            aasist_feat, _ = self.aasist(x, Freq_aug=Freq_aug)
            ser_feat = self.ser(x_emo)

        modulated_features = self.film(aasist_feat, ser_feat)
        
        output = self.classifier(modulated_features)
        
        return modulated_features, output

    @property
    def device(self):
        return next(self.parameters()).device