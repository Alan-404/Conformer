import torch
import torch.nn as nn

class Transducer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Transformer(batch_first=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transformer(x)
        return x