import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

class CheckpointManager:
    def __init__(self, saved_folder: str, n_saved: int = 3) -> None:
        self.saved_folder = saved_folder
        self.n_saved = n_saved
        
        self.saved_samples = []
        print("Init Checkpoint Manager")
        if os.path.exists(saved_folder):
            os.makedirs(saved_folder)

    def load_checkpoint(self, checkpoint: str, model: nn.Module, optimizer: optim.Optimizer, scheduler: lr_scheduler.LRScheduler):
        checkpoint_data = torch.load(checkpoint, map_location='cpu')

        model = model.load_state_dict(checkpoint_data['model'])
        optimizer = optimizer.load_state_dict(checkpoint_data['optimizer'])
        scheduler = scheduler.load_state_dict(checkpoint_data['scheduler'])
        n_steps = checkpoint_data['n_steps']
        n_epochs = checkpoint_data['n_epochs']

        return model, optimizer, scheduler, n_steps, n_epochs
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, scheduler: lr_scheduler.LRScheduler, n_steps: int, n_epochs: int):
        checkpoint_data = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'n_steps': n_steps,
            'n_epochs': n_epochs
        }

        saved_path = f"{self.saved_folder}/{n_steps}.pt"
        torch.save(checkpoint_data, saved_path)

        if len(self.saved_samples) == self.n_saved:
            os.remove(f"{self.saved_folder}/{self.saved_samples[0]}.pt")
        self.saved_samples.append(n_steps)
