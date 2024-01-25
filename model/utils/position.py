import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.register_buffer("angles", self.__encode_embedding(d_model))
        
    def __encode_ctx(self, n_ctx: int) -> torch.Tensor:
        pos = torch.arange(n_ctx)
        pos = pos.unsqueeze(-1)
        return pos.type(torch.float32)
    
    def __encode_embedding(self, embedding_dim: int) -> torch.Tensor:
        angles = torch.arange(embedding_dim)
        angles[1::2] = angles[0::2]
        angles = 1/(torch.pow(10000, angles/embedding_dim))
        angles = angles.unsqueeze(0)
        return angles
    
    def forward(self, n_ctx: int) -> torch.Tensor:
        pos = self.__encode_ctx(n_ctx).to(self.angles.device)
        
        pos_angles = torch.matmul(pos, self.angles)
        pos_angles[:, 0::2] = torch.sin(pos_angles[:, 0::2])
        pos_angles[:, 1::2] = torch.cos(pos_angles[:, 1::2])

        pos_angles = pos_angles.unsqueeze(0)
        return pos_angles

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        angles = torch.matmul(position, self.div_term)

        pe_positive[:, 0::2] = torch.sin(angles)
        pe_positive[:, 1::2] = torch.cos(angles)
        pe_negative[:, 0::2] = torch.sin(-1 * angles)
        pe_negative[:, 1::2] = torch.cos(-1 * angles)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)

        pe = pe.repeat([x.size(0), 1, 1]) # Repeat by Batch Size
        
        return pe.to(x.device)