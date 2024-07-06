import torch
import torch.nn as nn
from model.utils.convolution import ConvolutionSubsampling
from model.utils.block import ConformerBlock
from model.utils.masking import generate_padding_mask
from model.utils.position import RelativePositionalEncoding
from typing import Optional

class Encoder(nn.Module):
    def __init__(self, n_mel_channels: int, n: int, d_model: int, heads: int, kernel_size: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.downsampling_conv = ConvolutionSubsampling(channels=d_model)
        self.linear = nn.Linear(in_features=d_model * (((n_mel_channels - 1) // 2 - 1) // 2), out_features=d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.rel_pe = RelativePositionalEncoding(d_model=d_model)
        self.layers = nn.ModuleList([ConformerBlock(d_model=d_model, heads=heads, kernel_size=kernel_size, dropout_rate=dropout_rate) for _ in range(n)])
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Subsampling Mel - Spectrogram
        x, lengths = self.downsampling_conv(x, lengths)

        print(x.shape)

        # Pre - Project
        x = self.linear(x)
        
        x = self.dropout(x)
        
        # Mask Generation
        mask = None
        if lengths is not None:
            mask = (~generate_padding_mask(lengths))[:, None, None, :]
        
        # Conformer Handling
        rel_pos = self.rel_pe(x)
        for layer in self.layers:
            x = layer(x, rel_pos, mask)

        return x, lengths