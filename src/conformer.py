import torch
import torch.nn as nn
from src.modules.encoder import Encoder
from src.modules.decoder import Decoder
from typing import Optional

class Conformer(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 n_mel_channels: int, 
                 encoder_n_layers: int, 
                 encoder_dim: int, 
                 heads: int, 
                 kernel_size: int,
                 decoder_n_layers: int = 1,
                 decoder_dim: int = 640, 
                 eps: float=1e-5,
                 dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.encoder = Encoder(n_mel_channels=n_mel_channels, n=encoder_n_layers, d_model=encoder_dim, heads=heads, kernel_size=kernel_size, eps=eps, dropout_rate=dropout_rate)
        self.decoder = Decoder(vocab_size=vocab_size, d_model=encoder_dim, n=decoder_n_layers, hidden_dim=decoder_dim, eps=eps)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        x, lengths = self.encoder(x, lengths)
        x = self.decoder(x, lengths)
        if lengths is not None:
            return x, lengths
        return x