import torch
import torch.nn as nn
from torchaudio.transforms import SpecAugment
import random
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
    
# class TimeWrapping(nn.Module):
#     def __init__(self, W: int = 80) -> None:
#         super().__init__()
#         self.W = W
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.ndim == 2:
#             freq_length, time_length = x.size()
#         else:
#             _, freq_length, time_length = x.size()

#         y = freq_length // 2