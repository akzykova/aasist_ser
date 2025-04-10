import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from .AASIST import Model
from .ACRNN import acrnn

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
        self.frame_length = int(0.025 * sample_rate)  # 25ms
        self.frame_step = int(0.01 * sample_rate) 

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.frame_length,
            win_length=self.frame_length,
            hop_length=self.frame_step,
            n_mels=n_mels,
            power=2  # Для совместимости с logfbank
        )

        self.aasist_feat_dim = 5 * aasist_config["gat_dims"][1]
        self.ser_feat_dim = 256

        # self.aasist_norm = nn.LayerNorm(self.aasist_feat_dim)
        # self.ser_norm = nn.LayerNorm(self.ser_feat_dim) 
        self.layer_norm = nn.LayerNorm(self.aasist_feat_dim + self.ser_feat_dim)       
        self.classifier = nn.Linear(self.aasist_feat_dim + self.ser_feat_dim, 2)

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.classifier.bias, 0.0)
            # nn.init.ones_(self.aasist_norm.weight)
            # nn.init.zeros_(self.aasist_norm.bias)
            # nn.init.ones_(self.ser_norm.weight)
            # nn.init.zeros_(self.ser_norm.bias)

    def extract_mel_features(self, x):
        mel_spec = self.mel_transform(x)  # (batch, n_mels, time)
        
        log_mel = torch.log(mel_spec + 1e-6)
        
        delta1 = torchaudio.functional.compute_deltas(log_mel)
        delta2 = torchaudio.functional.compute_deltas(delta1)
        
        features = torch.stack([
            log_mel[..., :300],
            delta1[..., :300],
            delta2[..., :300]
        ], dim=1)  # (batch, 3, n_mels, 300)
        
        return features

    def forward(self, x, Freq_aug=False):
        with torch.no_grad():
            aasist_feat, _ = self.aasist(x, Freq_aug=Freq_aug)
            
            mel_features = self.extract_mel_features(x)
            ser_feat = self.ser(mel_features)

        # aasist_feat = self.aasist_norm(aasist_feat)
        # ser_feat = self.ser_norm(ser_feat)
        
        combined = self.layer_norm(torch.cat([aasist_feat, ser_feat], dim=1))
        output = self.classifier(combined)
        
        return combined, output

    @property
    def device(self):
        return next(self.parameters()).device