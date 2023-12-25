import torch
import torch.nn as nn
from src.utils.position import PositionalEncoding
from src.utils.convolution import Extractor
from src.utils.block import ConformerBlock
from src.utils.masking import generate_mask
from typing import Optional

class Encoder(nn.Module):
    def __init__(self, n_mel_channels: int, n: int, d_model: int, heads: int, kernel_size: int, eps: float, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.extractor = Extractor(in_channels=n_mel_channels, out_channels=d_model)
        self.linear = nn.Linear(in_features=d_model, out_features=d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.positional_embedding = PositionalEncoding(d_model=d_model)
        self.layers = nn.ModuleList([ConformerBlock(d_model=d_model, heads=heads, kernel_size=kernel_size, eps=eps, dropout_rate=dropout_rate) for _ in range(n)])
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        print(x.size())
        x = self.extractor(x)
        x = x.transpose(-1, -2)
        x = self.linear(x)
        x = self.dropout(x)
        print(x.size())

        pos_embedding = self.positional_embedding(x.size(1))

        pos_embedding = pos_embedding.repeat([x.size(0), 1,1])

        mask = None
        if lengths is not None:
            lengths = torch.ceil(lengths / 2)
            mask = generate_mask(lengths).to(x.device) == 0
            mask = mask.unsqueeze(1).unsqueeze(1)
    
        for layer in self.layers:
            x = layer(x, pos_embedding, mask)

        if lengths is not None:
            return x, lengths
        return x