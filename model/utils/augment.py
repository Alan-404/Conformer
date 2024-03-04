import torch
import torch.nn as nn

import random
import copy
import math

from typing import Optional

class SpecAugment(nn.Module):
    def __init__(self, time_mask_param: int, freq_mask_param, ratio: float = 1, n_time_masks: int = 1, n_freq_masks: int = 1, zero_masking: bool = True) -> None:
        super().__init__()
        self.time_masking = Masking(n_masks=n_time_masks, mask_param=time_mask_param, p=ratio, zero_masking=zero_masking)
        self.freq_masking = Masking(n_masks=n_freq_masks, mask_param=freq_mask_param, p=1, zero_masking=zero_masking)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        assert x.ndim >= 2
        x = self.time_masking(x, lengths)
        x = x.transpose(-1, -2)
        x = self.freq_masking(x)
        x = x.transpose(-1, -2)
        
        return x

class Masking(nn.Module):
    def __init__(self, n_masks: int = 1, mask_param: int = 10, p: float = 1, zero_masking: bool = True) -> None:
        super().__init__()
        self.n_masks = n_masks

        self.mask_param = mask_param

        self.p = p

        self.zero_masking = zero_masking

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None, return_indices: bool = False):
        assert x.ndim >= 1
        mask_value = 0.0
        
        if not self.zero_masking:
            mask_value = x.mean()
        
        max_length = length = x.size(-1)
        if lengths is not None:
            length = torch.min(lengths)

        if return_indices:
            indices = torch.ones((x.size(0), max_length))

        for _ in range(self.n_masks):
            p = random.uniform(0, self.p)
            
            num_span = math.ceil(self.mask_param * p)
            start = random.randint(0, length - num_span)
            end = random.randint(start, start + num_span)

            x[..., :, start: end] = mask_value 

            if return_indices:
                indices[:, start: end] = 0

        if return_indices:
            return x, indices

        return x