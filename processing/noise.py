from typing import Any
import torch
from torchaudio.transforms import TimeMasking, FrequencyMasking


class SpecAugment:
    def __init__(self, freq_augment: int = 27, time_augment: int = 10, time_mask_ratio: float = 0.05) -> None:
        self.time_masker = TimeMasking(time_mask_param=time_augment, p=time_mask_ratio)
        self.freq_masker = FrequencyMasking(freq_mask_param=freq_augment)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self.freq_masker(x)
        x = self.time_masker(x)

        return x