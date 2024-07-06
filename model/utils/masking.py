import torch
from typing import Optional

def generate_padding_mask(lengths: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    '''
        lengths: Tensor, shape = (batch_size)
        Return Padding Mask with shape = (batch_size, max_length)
    '''
    if max_length is None:
        max_length = lengths.max()
    
    x = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device).unsqueeze(0) # shape = [1, max_length]

    return lengths.unsqueeze(dim=-1) > x