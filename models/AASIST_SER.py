import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from .AASIST import Model
from .ACRNN import acrnn

class FiLMLayer(nn.Module):
    """FiLM (Feature-wise Linear Modulation) layer"""
    def __init__(self, feature_dim, condition_dim):
        super().__init__()
        self.gamma = nn.Linear(condition_dim, feature_dim)
        self.beta = nn.Linear(condition_dim, feature_dim)
        
    def forward(self, x, condition):
        # x: (batch, feature_dim)
        # condition: (batch, condition_dim)
        gamma = self.gamma(condition)  # (batch, feature_dim)
        beta = self.beta(condition)    # (batch, feature_dim)
        return x * gamma + beta


class AASISTWithEmotion(nn.Module):
    def __init__(self, aasist_config, ser_config, n_mels=40, sample_rate=16000):
        super().__init__()
        
        # Инициализация AASIST (размороженный)
        self.aasist = Model(aasist_config)
        self.aasist.load_state_dict(torch.load(aasist_config["aasist_path"]))
        # self.aasist.eval()
        # for p in self.aasist.parameters():
        #     p.requires_grad = False

        # Инициализация SER (замороженный)
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

        self.film = FiLMLayer(self.aasist_feat_dim, self.ser_feat_dim)
        
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
        aasist_feat, _ = self.aasist(x, Freq_aug=Freq_aug)

        with torch.no_grad():
            ser_feat = self.ser(self.extract_mel_features(x))

        # Применяем FiLM к AASIST фичам с условием от SER
        modulated_features = self.film(aasist_feat, ser_feat)
        
        # Классификация
        output = self.classifier(modulated_features)
        
        return modulated_features, output

    @property
    def device(self):
        return next(self.parameters()).device