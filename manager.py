import os

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from checkpoint import load_model

from typing import Tuple, Literal

class CheckpointManager:
    def __init__(self, saved_folder: str, n_savings: int = 3) -> None:
        self.saved_folder = saved_folder
        self.n_savings = n_savings
        
        self.saved_samples = []

        if os.path.exists(saved_folder) == False:
            os.makedirs(saved_folder)

    def load_checkpoint(self, checkpoint: str, model: Module, optimizer: Optimizer, scheduler: LRScheduler, world_size: int = 1) -> Tuple[int, int]:
        checkpoint_data = torch.load(checkpoint, map_location='cpu')

        load_model(checkpoint_data['model'], model, world_size=world_size)
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

class EarlyStopping:
    def __init__(self, n_patiences: int = 3, condition_metric: Literal['up', 'down'] = 'up') -> None:
        self.early_stop = False

        self.max_patiences = n_patiences
        self.current_patiences = 0

        self.up = (condition_metric == 'up')

        self.current_score = None
        self.best_score = None

    def __call__(self, score: float) -> None:
        if self.current_score is not None:
            if (self.current_score > score) == self.up:
                self.current_patiences += 1
            elif self.current_score == score:
                self.current_score += 0.5
            else:
                self.current_patiences = 0
            
            self.current_score = score

            if self.current_patiences >= self.max_patiences:
                self.early_stop = True
        else:
            self.current_score = score
            self.best_score = score