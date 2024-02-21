from typing import Any
import torch
from torchaudio.transforms import TimeMasking, FrequencyMasking
import torchvision.transforms.functional as F_vision
import random

from typing import Union

class NormalizeAugment:
    def __init__(self) -> None:
        pass

    def __call__(self, x: torch.Tensor) -> Any:
        return (x - x.mean()) / torch.sqrt(x.var() + 1e-7)

class CropAugment:
    def __init__(self, n_mel_channels: int, ratio: float = 0.25) -> None:
        self.ratio = ratio
        self.n_mel_channels = n_mel_channels

    def __call__(self, x: torch.Tensor, length: int) -> Any:
        length = int(length)
        time = x.size(-1)

        start_time_cut = random.randint(0, int(length - length * self.ratio))
        end_time_cut = random.randint(int(start_time_cut + length * self.ratio), length)

        x = x[:, :, start_time_cut : end_time_cut]

        x = F_vision.resize(x.unsqueeze(1), (self.n_mel_channels, time)).squeeze(1)

        return x

class SpecAugment:
    def __init__(self, freq_augment: int = 27, time_augment: int = 10, time_mask_ratio: float = 0.05) -> None:
        self.time_masker = TimeMasking(time_mask_param=time_augment, p=time_mask_ratio)
        self.freq_masker = FrequencyMasking(freq_mask_param=freq_augment)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self.freq_masker(x)
        x = self.time_masker(x)

        return x
