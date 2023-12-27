import torch
import torch.nn as nn
from src.utils.activation import GLU, Swish
from typing import Optional

class ConvolutionModule(nn.Module):
    def __init__(self, channels: int, kernel_size: int, eps: float = 1e-5, dropout_rate: float = 0.0) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.layer_norm = nn.LayerNorm(normalized_shape=channels, eps=eps)
        self.pointwise_conv_1 = nn.Conv1d(in_channels=channels, out_channels=channels * 2, kernel_size=1, stride=1, padding=0)
        self.glu = GLU(dim=1)
        self.deepwise_conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding, groups=channels)
        self.batch_norm = nn.BatchNorm1d(num_features=channels, eps=eps)
        self.swish = Swish()
        self.pointwise_conv_2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = x.transpose(-1, -2)
        x = self.pointwise_conv_1(x)
        x = self.glu(x)
        x = self.deepwise_conv(x)
        x = self.swish(x)
        x = self.pointwise_conv_2(x)
        x = self.dropout(x)
        x = x.transpose(-1, -2)
        return x
    
class ConvolutionSubsampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.act_1 = nn.GELU()
        self.max_pool_1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv_2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.act_2 = nn.GELU()
        self.max_pool_2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.act_1(x)
        x = self.max_pool_1(x)
        x = self.conv_2(x)
        x = self.act_2(x)
        x = self.max_pool_2(x)

        if lengths is not None:
            lengths = torch.ceil(torch.ceil(lengths/2)/4)
            
        return x, lengths