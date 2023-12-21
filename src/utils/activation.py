import torch
import torch.nn as nn

class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, gate = x.chunk(chunks=2, dim=self.dim)
        return outputs * gate.sigmoid()