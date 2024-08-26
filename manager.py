import os

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from common import change_format_single_gpu

from typing import Tuple

class CheckpointManager:
    def __init__(self, saved_folder: str, n_savings: int = 3) -> None:
        self.saved_folder = saved_folder
        self.n_savings = n_savings
        
        self.saved_samples = []

        if os.path.exists(saved_folder) == False:
            os.makedirs(saved_folder)

    def load_checkpoint(self, checkpoint: str, model: Module, optimizer: Optimizer, scheduler: LRScheduler, world_size: int = 1) -> Tuple[int, int]:
        checkpoint_data = torch.load(checkpoint, map_location='cpu')

        model_state_dict = checkpoint_data['model']
        if world_size == 1:
            model_state_dict = change_format_single_gpu(model_state_dict)
        model.load_state_dict(model_state_dict)

        optimizer.load_state_dict(checkpoint_data['optimizer'])
        scheduler.load_state_dict(checkpoint_data['scheduler'])
        n_steps = checkpoint_data['n_steps']
        n_epochs = checkpoint_data['n_epochs']

        return n_steps, n_epochs
    
    def save_checkpoint(self, model: Module, optimizer: Optimizer, scheduler: LRScheduler, n_steps: int, n_epochs: int) -> None:
        checkpoint_data = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'n_steps': n_steps,
            'n_epochs': n_epochs
        }

        saved_path = f"{self.saved_folder}/{n_steps}.pt"
        torch.save(checkpoint_data, saved_path)

        if len(self.saved_samples) == self.n_savings:
            os.remove(f"{self.saved_folder}/{self.saved_samples[0]}.pt")
            self.saved_samples.pop(0)
        
        self.saved_samples.append(n_steps)
