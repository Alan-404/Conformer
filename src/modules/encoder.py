import torch
import torch.nn as nn
from src.utils.convolution import ConvolutionSubsampling
from src.utils.block import ConformerBlock
from src.utils.masking import generate_mask
from typing import Optional

class Encoder(nn.Module):
    def __init__(self, n_mel_channels: int, n: int, d_model: int, heads: int, kernel_size: int, eps: float, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.subsampling = ConvolutionSubsampling(channels=d_model)
        self.linear = nn.Linear(in_features=d_model * ((n_mel_channels - 1) // 2), out_features=d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layers = nn.ModuleList([ConformerBlock(d_model=d_model, heads=heads, kernel_size=kernel_size, eps=eps, dropout_rate=dropout_rate) for _ in range(n)])
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Subsampling Mel - Spectrogram
        x, lengths = self.subsampling(x, lengths)
        # Pre - Project
        x = self.linear(x)
        x = self.dropout(x)
        # Mask Generation
        mask = None
        if lengths is not None:
            mask = generate_mask(lengths).to(x.device)
            mask = (mask == 0).unsqueeze(1).unsqueeze(1)
        # Conformer Handling
        for layer in self.layers:
            x = layer(x, mask)

        return x, lengths