import torch
import torch.nn as nn

from model.modules.encoder import Encoder
from model.modules.decoder import Decoder

from typing import Optional, Tuple
    
class Conformer(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 n_mel_channels: int = 80, 
                 n_conformer_blocks: int = 16, 
                 d_model: int = 256, 
                 n_heads: int = 4, 
                 kernel_size: int = 31,
                 lstm_hidden_dim: int = 640,
                 n_lstm_layers: int = 1,
                 dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.encoder = Encoder(n_mel_channels=n_mel_channels, n_blocks=n_conformer_blocks, d_model=d_model, n_heads=n_heads, kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.decoder = Decoder(vocab_size=vocab_size, d_model=d_model, hidden_dim=lstm_hidden_dim, n_layers=n_lstm_layers)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:        
        x, lengths = self.encoder(x, lengths)
        x = self.decoder(x, lengths)
        return x, lengths