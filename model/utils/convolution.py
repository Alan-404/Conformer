import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.activation import GLU, Swish
from typing import Optional

class ConvolutionModule(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.layer_norm = nn.LayerNorm(normalized_shape=channels)
        self.pointwise_conv_1 = nn.Conv1d(in_channels=channels, out_channels=channels * 2, kernel_size=1, stride=1, padding=0)
        self.glu = GLU(dim=1)
        self.deepwise_conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding, groups=channels)
        self.batch_norm = nn.BatchNorm1d(num_features=channels)
        self.swish = Swish()
        self.pointwise_conv_2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = x.transpose(-1, -2)
        x = self.pointwise_conv_1(x)
        x = self.glu(x)
        x = self.deepwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv_2(x)
        x = self.dropout(x)
        x = x.transpose(-1, -2)
        return x

class ConvolutionSubsampling(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=3, stride=2)
        self.act_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2)
        self.act_2 = nn.ReLU()

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        x = x.unsqueeze(1)
        x = self.conv_1(x)
        x = self.act_1(x)
        x = self.conv_2(x)
        x = self.act_2(x)

        batch_size, dim, subsampling_channels, subsampling_length = x.size()

        x = x.permute([0, 3, 1, 2])
        x = x.contiguous().view(batch_size, subsampling_length, dim * subsampling_channels)

        if lengths is not None:
            lengths = ((lengths - 1) // 2 - 1) // 2
            
        return x, lengths
    
class DepthWiseSeperableConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depth_wise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, groups=in_channels)
        self.point_wise_conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = self.depth_wise_conv(x)
        x = self.point_wise_conv(x)

        return x
    
class DownsamplingConvolution(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv_1 = DepthWiseSeperableConvolution(in_channels=1, out_channels=channels)
        self.conv_2 = DepthWiseSeperableConvolution(in_channels=channels, out_channels=channels)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor]):
        x = x.unsqueeze(dim=1)

        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = F.relu(x)

        batch_size, dim, subsampling_channels, subsampling_length = x.size()

        x = x.permute([0, 3, 1, 2])
        x = x.contiguous().view(batch_size, subsampling_length, dim * subsampling_channels)

        if lengths is not None:
            lengths = ((lengths - 1) // 2 - 1) // 2

        return x, lengths