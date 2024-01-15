import torch
import torch.nn as nn
from src.conformer import Encoder
import copy

class BYOL(nn.Module):
    def __init__(self, n_mel_channels: int, n: int, d_model: int, heads: int, kernel_size: int, eps: float, dropout_rate: float, embedding_size: int, moving_average_decay: float = 0.99) -> None:
        super().__init__()
        self.student = Network(
            n_mel_channels=n_mel_channels,
            n=n,
            d_model=d_model,
            heads=heads,
            kernel_size=kernel_size,
            eps=eps,
            dropout_rate=dropout_rate,
            embedding_size=embedding_size
        )

        self.teacher = self.__get_teacher()
        self.updater = EMA(alpha=moving_average_decay)

        self.predict = nn.Linear(in_features=embedding_size, out_features=embedding_size)

    @torch.no_grad()
    def update_moving_average(self):
        for student_params, teacher_params in zip(self.student.parameters(), self.teacher.parameters()):
            old_weight, up_weight = teacher_params.data, student_params.data
            teacher_params.data = self.updater.update_average(old_weight, up_weight)

    @torch.no_grad()
    def __get_teacher(self):
        return copy.deepcopy(self.student)
    
    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        if return_embedding:
            return self.student(x, return_embedding=True)
        
        proj = self.student(x)
        student_pred = self.predict(proj)

        with torch.no_grad():
            teacher_proj = self.teacher(x)

        return student_pred, teacher_proj

class EMA:
   def __init__(self, alpha):
       super().__init__()
       self.alpha = alpha

   def update_average(self, old, new):
       if old is None:
           return new
       return old * self.alpha + (1 - self.alpha) * new

class Network(nn.Module):
    def __init__(self, n_mel_channels: int, n: int, d_model: int, heads: int, kernel_size: int, eps: float, dropout_rate: float, embedding_size: int) -> None:
        super().__init__()
        self.projection = Encoder(
            n_mel_channels=n_mel_channels,
            n=n,
            d_model=d_model,
            heads=heads,
            kernel_size=kernel_size,
            eps=eps,
            dropout_rate=dropout_rate
        )

        self.mlp = nn.Linear(in_features=d_model, out_features=embedding_size)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        x = self.projection(x)

        if return_embedding:
            return x
        
        x = self.mlp(x)
        return x

