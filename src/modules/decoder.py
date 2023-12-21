import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return x