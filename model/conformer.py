import torch
import torch.nn as nn
from model.modules.encoder import Encoder
from model.modules.decoder import Decoder
from typing import Optional, Tuple
    
class Conformer(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 n_mel_channels: int, 
                 n_blocks: int, 
                 d_model: int, 
                 heads: int, 
                 kernel_size: int,
                 dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.encoder = Encoder(n_mel_channels=n_mel_channels, n=n_blocks, d_model=d_model, heads=heads, kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.decoder = Decoder(vocab_size=vocab_size, d_model=d_model)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x, lengths = self.encoder(x, lengths)
        x = self.decoder(x)
        
        return x, lengths