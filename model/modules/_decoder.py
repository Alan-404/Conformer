import torch
import torch.nn as nn
from typing import Optional
from model.utils.activation import Swish

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.activation = Swish()
        self.norm = nn.BatchNorm1d(num_features=d_model)
        self.linear = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        x = self.activation(x)
        x = x.transpose(1,2)
        x = self.norm(x)
        x = x.transpose(1,2)
        x = self.linear(x)
        
        return x