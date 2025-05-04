import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
import librosa
import os
import python_speech_features as ps
import torchaudio
import torchaudio.transforms as T


def genSpoof_list(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            #key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x

class DatasetCustom(Dataset):
    def __init__(self, audio_dir, label=0, file_extensions='.flac', cut_len=64600, n_mels = 40, sample_rate=16000):
        self.audio_dir = audio_dir
        self.label = label
        self.cut_len = cut_len
        self.file_extensions = file_extensions
        self.file_list = []

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

        for root, _, files in os.walk(audio_dir):
            for file in files:
                if file.lower().endswith(self.file_extensions):
                    self.file_list.append(os.path.join(root, file))
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        filename = os.path.basename(filepath)

        wave = self.load_audio(filepath)
        
        wave = self.pad(wave)
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave = wave / max_val  

        feature_tensor = torch.tensor(wave, dtype=torch.float32)
        spec_tensor = self.extract_mel_features(wave)
        spec_tensor = torch.tensor(spec_tensor, dtype=torch.float32)
        
        return feature_tensor, spec_tensor, torch.tensor(self.label, dtype=torch.float32), filename
    
    def load_audio(self, filepath):
        wave, fs = librosa.load(filepath, sr=16000)
        return np.array(wave, dtype=float)
    
    def extract_mel_features(self, x):
        mel_spec = self.mel_transform(x)
        log_mel = torch.log(mel_spec + 1e-6)
        delta1 = torchaudio.functional.compute_deltas(log_mel)
        delta2 = torchaudio.functional.compute_deltas(delta1)
        return torch.stack([log_mel[..., :300], delta1[..., :300], delta2[..., :300]], dim=1)
    
    def pad(self, x, max_len=64600):
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        num_repeats = int(max_len / x_len) + 1
        padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
        return np.array(padded_x, dtype=float)


# def get_dataloader(audio_dir, label=0, batch_size=24, num_workers=None, shuffle=True, pin_memory=True):
#     if num_workers is None:
#         num_workers = min(4, os.cpu_count() // 2)

#     dataset = DatasetCustom(audio_dir, label=label)
#     generator = torch.Generator()
#     generator.manual_seed(seed)
    
#     loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         shuffle=shuffle,
#         pin_memory=pin_memory,
#         persistent_workers=num_workers > 0,
#         drop_last=False
#     )
    
#     return loader

class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir, n_mels=40, sample_rate=16000):
        """self.list_IDs : list of strings (each string: utt key),
           self.labels   : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

        # Для спектрограмм
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

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad_random(X, self.cut)

        mel_spec = self.mel_transform(torch.tensor(X_pad, dtype=torch.float32))
        log_mel = torch.log(mel_spec + 1e-6)
        
        delta1 = torchaudio.functional.compute_deltas(log_mel)
        delta2 = torchaudio.functional.compute_deltas(delta1)
        spec_tensor = torch.stack([log_mel[..., :300], delta1[..., :300], delta2[..., :300]], dim=1)

        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, spec_tensor, y


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir, n_mels=40, sample_rate=16000):
        """self.list_IDs : list of strings (each string: utt key),"""
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

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

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad(X, self.cut)

        mel_spec = self.mel_transform(torch.tensor(X_pad, dtype=torch.float32))
        log_mel = torch.log(mel_spec + 1e-6)
        
        delta1 = torchaudio.functional.compute_deltas(log_mel)
        delta2 = torchaudio.functional.compute_deltas(delta1)
        spec_tensor = torch.stack([log_mel[..., :300], delta1[..., :300], delta2[..., :300]], dim=1)

        x_inp = Tensor(X_pad)
        return x_inp, spec_tensor, key

