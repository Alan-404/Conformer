import torch

def generate_mask(lengths: torch.Tensor):
    max_len = torch.max(lengths)
    mask = []
    for length in lengths:
        mask.append(torch.tensor([1] * int(length.item()) + [0] * int(max_len.item() - length.item())))
    return torch.stack(mask)