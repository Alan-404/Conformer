import torch
import torch.nn as nn
from src.utils.ffn import FeedForwardModule
from src.utils.attention import MultiHeadSelfAttentionModule
from src.utils.convolution import ConvolutionModule
from typing import Optional

class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, kernel_size: int, eps: float, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.ffn_1 = FeedForwardModule(dim=d_model, eps=eps, dropout_rate=dropout_rate)
        self.attention = MultiHeadSelfAttentionModule(heads=heads, d_model=d_model, eps=eps, dropout_rate=dropout_rate)
        self.conv = ConvolutionModule(channels=d_model, kernel_size=kernel_size, eps=eps, dropout_rate=dropout_rate)
        self.ffn_2 = FeedForwardModule(dim=d_model, eps=eps, dropout_rate=dropout_rate)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model, eps=eps)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # sub - layer 1
        ffn_1_out = (1/2) * self.ffn_1(x) + x
        # sub - layer 2
        attention_out = self.attention(ffn_1_out, mask) + ffn_1_out
        # sub - layer 3
        conv_output = self.conv(attention_out) + attention_out
        # sub - layer 4
        ffn_2_out = (1/2) * self.ffn_2(conv_output) + conv_output
        # sub - layer 5
        output = self.layer_norm(ffn_2_out)
        
        return output