import torch
import torch.nn as nn
from src.modules.encoder import Encoder
from src.modules.decoder import Decoder
from typing import Optional

class Conformer(nn.Module):
    def __init__(self, vocab_size: int, n_mel_channels: int, n: int, d_model: int, heads: int, kernel_size: int, hidden_dim: int, eps: float=1e-5, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.encoder = Encoder(n_mel_channels=n_mel_channels, n=n, d_model=d_model, heads=heads, kernel_size=kernel_size, eps=eps, dropout_rate=dropout_rate)
        self.decoder = Decoder(vocab_size=vocab_size, d_model=d_model, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        x, lengths = self.encoder(x, lengths)
        x = self.decoder(x, lengths)
        if lengths is not None:
            return x, lengths
        return x