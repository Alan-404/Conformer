import torch
import torch.nn as nn
<<<<<<< HEAD

from src.utils.position import PositionalEncoding
from src.utils.convolution import Extractor
=======
from src.utils.convolution import Extractor
from src.utils.position import PositionalEncoding
>>>>>>> 38e5bbddb652930aa03db79baf9ed0021784d1a3
from src.utils.block import ConformerBlock

from typing import Optional

class Encoder(nn.Module):
    def __init__(self, n_mel_channels: int, n: int, d_model: int, heads: int, kernel_size: int, eps: float, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.extractor = Extractor(in_channels=n_mel_channels, out_channels=d_model)
        self.linear = nn.Linear(in_features=d_model, out_features=d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.positional_embedding = PositionalEncoding(d_model=d_model)
        self.layers = nn.ModuleList([ConformerBlock(d_model=d_model, heads=heads, kernel_size=kernel_size, eps=eps, dropout_rate=dropout_rate) for _ in range(n)])
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, _, n_ctx = x.size()
        x = self.extractor(x)
        x = x.transpose(-1, -2)
        x = self.linear(x)
        x = self.dropout(x)

        pos_embedding = self.positional_embedding(n_ctx).repeat([batch_size, 1,1])
    
        for layer in self.layers:
            x = layer(x, pos_embedding, mask)

        return x