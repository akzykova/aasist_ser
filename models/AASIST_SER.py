import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from .AASIST import Model
from .ACRNN import acrnn

class FiLMBlock(nn.Module):
    def __init__(self, sv_dim, cm_dim):
        super().__init__()
        self.shared_proj = nn.Sequential(
            nn.Linear(cm_dim, cm_dim),
            nn.ReLU(),
            nn.LayerNorm(cm_dim)
        )

        self.gamma_head = nn.Linear(cm_dim, sv_dim)
        self.beta_head = nn.Linear(cm_dim, sv_dim)

    def forward(self, e_sv, e_cm):
        shared = self.shared_proj(e_cm)
        gamma = self.gamma_head(shared)  # [B, sv_dim]
        beta = self.beta_head(shared)    # [B, sv_dim]
        
        return gamma * e_sv + beta


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

        self.film_block = FiLMBlock(
            sv_dim=self.aasist_feat_dim,
            cm_dim=self.ser_feat_dim
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.aasist_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

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

        e1 = self.film_block(aasist_feat, ser_feat)
        
        output = self.classifier(e1)
        
        return e1, output

    @property
    def device(self):
        return next(self.parameters()).device