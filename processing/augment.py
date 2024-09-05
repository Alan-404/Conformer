import torch
from torchaudio.transforms import SpecAugment

from typing import Union

class ConformerAugment:
    def __init__(self, n_time_masks: int = 10, time_mask_param: int = 35, n_freq_masks: int = 10, freq_mask_param: int = 35, ratio: float = 0.05, zero_masking: bool = True, device: Union[str, int] = 'cpu') -> None:
        self.spec_augment = SpecAugment(
            n_time_masks=n_time_masks,
            time_mask_param=time_mask_param,
            n_freq_masks=n_freq_masks,
            freq_mask_param=freq_mask_param,
            p=ratio,
            zero_masking=zero_masking
        ).to(device)

    def __call__(self, mels: torch.Tensor) -> torch.Tensor:
        return self.spec_augment(mels)