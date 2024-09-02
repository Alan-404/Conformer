import torch
from torchaudio.transforms import SpecAugment

class ConformerAugment:
    def __init__(self) -> None:
        self.spec_augment = SpecAugment(
            n_time_masks=10,
            time_mask_param=35,
            n_freq_masks=10,
            freq_mask_param=35,
            p=0.05,
            zero_masking=True
        ).to(self.device)

    def __call__(self, mels: torch.Tensor) -> torch.Tensor:
        return self.spec_augment(mels)