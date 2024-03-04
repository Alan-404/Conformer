import torch
import torch.nn as nn
from typing import Optional
from model.utils.activation import Swish

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=1, groups=d_model)
        self.pointwise_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.activation = Swish()
        self.norm = nn.BatchNorm1d(num_features=d_model)
        self.linear = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: torch.Tensor):
        x = x.transpose(-1, -2)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.activation(x)
        x = self.norm(x)
        x = x.transpose(-1, -2)
        x = self.linear(x)
        return x

class RNNDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n: int = 1, hidden_dim: int = 640) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=hidden_dim, num_layers=n, batch_first=True)
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
        x = x.transpose(1,2)
        x = self.norm(x)
        x = x.transpose(1,2)
        x = self.linear(x)
        
        return x