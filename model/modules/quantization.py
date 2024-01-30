import torch
import torch.nn as nn
import torch.nn.functional as F

class Quantization(nn.Module):
    def __init__(self, n_mel_channels: int, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=d_model * (((n_mel_channels - 1) // 2 - 1) // 2), out_features=d_model)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        return x
