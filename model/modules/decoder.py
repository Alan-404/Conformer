import torch
import torch.nn as nn
from model.utils.activation import Swish

from typing import Optional

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.activation = Swish()
        self.norm = nn.BatchNorm1d(num_features=hidden_dim)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=vocab_size)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths=lengths.cpu(), batch_first=True, enforce_sorted=True)
        self.lstm.flatten_parameters()
        x, _= self.lstm(x)
        if lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.activation(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1,2)
        x = self.linear(x)
        
        return x