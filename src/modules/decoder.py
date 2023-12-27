import torch
import torch.nn as nn
from typing import Optional

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=hidden_dim)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=vocab_size)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lengths is not None:
            lengths = lengths.cpu().numpy()
            x = nn.utils.rnn.pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
        
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        if lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = self.linear(x)
        return x