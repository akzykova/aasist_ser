import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from .AASIST import Model
from .ACRNN import acrnn

# class FiLMBlock(nn.Module):
#     def __init__(self, sv_dim, cm_dim, hidden_dim=128):
#         super().__init__()
#         self.cm_ln = nn.LayerNorm(cm_dim)
#         self.condition_to_gamma_beta = nn.Sequential(
#             nn.Linear(cm_dim, hidden_dim),
#             nn.ReLU(),
#             nn.BatchNorm1d(hidden_dim),
#             nn.Linear(hidden_dim, 2 * sv_dim)
#         )
        
#         self.sv_ln = nn.LayerNorm(sv_dim)

#     def forward(self, e_sv, e_cm):
#         e_cm_norm = self.cm_ln(e_cm)
#         gamma_beta = self.condition_to_gamma_beta(e_cm_norm)  # (batch_size, 2 * sv_dim)
#         gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)      # (batch_size, sv_dim), (batch_size, sv_dim)
        
#         e_sv_norm = self.sv_ln(e_sv)
#         e1 = gamma * e_sv_norm + beta

#         return e1



class AASISTWithEmotion(nn.Module):
    def __init__(self, aasist_config, ser_config, n_mels=40, sample_rate=16000):
        super().__init__()
        
        self.aasist = Model(aasist_config)
        self.aasist.load_state_dict(torch.load(aasist_config["aasist_path"]))
        self.aasist.eval()
        for p in self.aasist.parameters():
            p.requires_grad = False

        

        self.ser = acrnn()
        self.ser.load_state_dict(torch.load(ser_config["ser_path"]))
        self.ser.eval()
        for p in self.ser.parameters():
            p.requires_grad = False

        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.frame_length = int(0.025 * sample_rate)
        self.frame_step = int(0.01 * sample_rate)

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.frame_length,
            win_length=self.frame_length,
            hop_length=self.frame_step,
            n_mels=n_mels,
            power=2
        )

        self.aasist_feat_dim = 5 * aasist_config["gat_dims"][1]
        self.ser_feat_dim = 256
        self.test_linear = nn.Linear(self.aasist_feat_dim + self.ser_feat_dim, 2)


    def extract_mel_features(self, x):
        mel_spec = self.mel_transform(x)
        log_mel = torch.log(mel_spec + 1e-6)
        delta1 = torchaudio.functional.compute_deltas(log_mel)
        delta2 = torchaudio.functional.compute_deltas(delta1)
        return torch.stack([log_mel[..., :300], delta1[..., :300], delta2[..., :300]], dim=1)

    def forward(self, x, Freq_aug=False):
        with torch.no_grad():
            aasist_feat, _ = self.aasist(x, Freq_aug=Freq_aug)
            ser_feat = self.ser(self.extract_mel_features(x))
        
        aasist_feat = F.normalize(aasist_feat, p=2, dim=1)
        ser_feat = F.normalize(ser_feat, p=2, dim=1)
        
        combined_feat = torch.cat([aasist_feat, ser_feat], dim=1)
        
        combined_feat = F.normalize(combined_feat, p=2, dim=1)
        
        out = self.test_linear(combined_feat)
        
        return combined_feat, out

    @property
    def device(self):
        return next(self.parameters()).device