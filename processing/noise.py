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
    
class OnlineAugment:
    def __init__(self, freq_augment: int = 35, time_augment: int = 10, time_mask_ratio: float = 0.065) -> None:
        self.spec_augment = SpecAugment(freq_augment=freq_augment, time_augment=time_augment, time_mask_ratio=time_mask_ratio)

    def __call__(self, mels: torch.Tensor) -> Any:
        return self.spec_augment(mels)
    
class TargetAugment:
    def __init__(self) -> None:
        pass

    def __call__(self, signals: torch.Tensor) -> Any:
        signals = (signals - signals.mean(dim=0)) / torch.sqrt(signals.var(dim=0) + 1e-7)

        return signals