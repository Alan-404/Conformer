import torch
import torch.nn as nn
from typing import Optional
from src.utils.activation import Swish

# class Decoder(nn.Module):
#     def __init__(self, vocab_size: int, d_model: int) -> None:
#         super().__init__()
#         self.linear = nn.Linear(in_features=d_model, out_features=vocab_size)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.linear(x)
#         return x
    

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n: int = 1, hidden_dim: int = 640) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=hidden_dim, num_layers=n, batch_first=True)
        self.activation = Swish()
        self.norm = nn.BatchNorm1d(num_features=hidden_dim)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=vocab_size)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths=lengths.cpu().numpy(), batch_first=True, enforce_sorted=False)
        
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        if lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        x = self.activation(x)
        x = self.norm(x)
        x = self.linear(x)
        return x