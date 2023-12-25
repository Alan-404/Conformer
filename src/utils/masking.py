import torch
import numpy as np

def generate_mask(lengths: torch.Tensor):
    max_len = torch.max(lengths)
    mask = []
    for length in lengths:
        mask.append(torch.tensor(np.array([1] * length + [0] * (max_len - length))))

    return torch.stack(mask)