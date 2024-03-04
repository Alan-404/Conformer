import torch
import torch.nn as nn

import random
import copy
import math

from typing import Optional

class SpecAugment(nn.Module):
    def __init__(self, mask_param: int, n_times: int = 1, ratio: float = 1.0, zero_masking: bool = True) -> None:
        super().__init__()
        self.time_masking = Masking(n_masks=n_times, mask_param=mask_param, p=ratio, zero_masking=zero_masking)
        self.freq_masking = Masking(n_masks=n_times, mask_param=mask_param, p=ratio, zero_masking=zero_masking)

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
        mel = copy.deepcopy(x)
        mask_value = 0.0
        
        if not self.zero_masking:
            mask_value = x.mean()
        
        max_length = length = x.size(-1)
        if lengths is not None:
            length = torch.min(lengths)

        if return_indices:
            indices = torch.ones((x.size(0), max_length))

        for _ in range(self.n_masks):
            p = random.uniform(1 - self.p, 1)
            
            start = random.randint(0, length - self.mask_param)
            end = random.randint(start, start + int(self.mask_param * p))

            mel[..., :, start: end] = mask_value 

            if return_indices:
                indices[:, start: end] = 0

        if return_indices:
            return mel, indices

        return mel