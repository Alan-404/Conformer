import torch
import torch.nn as nn
from model.utils.activation import Swish

class FeedForwardModule(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=dim)
        self.hidden_linear = nn.Linear(in_features=dim, out_features=4 * dim)
        self.swish = Swish()
        self.dropout_1 = nn.Dropout(p=dropout_rate)
        self.out_linear = nn.Linear(in_features=4*dim, out_features=dim)
        self.dropout_2 = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.hidden_linear(x)
        x = self.swish(x)
        x = self.dropout_1(x)
        x = self.out_linear(x)
        x = self.dropout_2(x)

        return x