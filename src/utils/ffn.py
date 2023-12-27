import torch
import torch.nn as nn
from src.utils.activation import Swish

class FeedForwardModule(nn.Module):
    def __init__(self, dim: int, eps: float, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=dim, eps=eps)
        self.hidden_linear = nn.Linear(in_features=dim, out_features=4 * dim)
        self.swish = Swish()
        self.dropout_1 = nn.Dropout(p=dropout_rate)
        self.out_linear = nn.Linear(in_features=4*dim, out_features=dim)
        self.dropout_2 = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layer_norm(x)
        y = self.hidden_linear(y)
        y = self.swish(y)
        y = self.dropout_1(y)
        y = self.out_linear(y)
        y = self.dropout_2(y)

        return (1/2) * y + x