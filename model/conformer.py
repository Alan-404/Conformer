import torch
import torch.nn as nn
from model.modules.encoder import Encoder
from model.modules.decoder import Decoder
from model.modules.audio import MelSpectrogram
from model.modules.augment import SpecAugment
from typing import Optional, Tuple
    
class Conformer(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 sample_rate: int = 16000,
                 n_fft: int = 400,
                 win_length: Optional[int] = None,
                 hop_length: int = 160,
                 fmin: float = 0.0,
                 fmax: Optional[float] = 8000,
                 n_mel_channels: int = 80, 
                 n_conformer_blocks: int = 16, 
                 d_model: int = 256, 
                 n_heads: int = 4, 
                 kernel_size: int = 31,
                 lstm_hidden_dim: int = 640,
                 n_lstm_layers: int = 1,
                 dropout_rate: float = 0.0,
                 n_masks: int = 10,
                 mask_param: int = 27,
                 mask_ratio: float = 0.05) -> None:
        super().__init__()
        win_length = win_length if win_length is not None else n_fft
        fmax = fmax if fmax is not None else sample_rate // 2
        self.hop_length = hop_length

        self.mel_spectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_max=fmax,
            f_min=fmin,
            norm='slaney',
            n_mels=n_mel_channels
        )

        self.spec_augment = SpecAugment(
            n_time_masks=n_masks,
            n_freq_masks=n_masks,
            time_mask_param=mask_param,
            freq_mask_param=mask_param,
            p=mask_ratio
        )

        self.encoder = Encoder(n_mel_channels=n_mel_channels, n_blocks=n_conformer_blocks, d_model=d_model, n_heads=n_heads, kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.decoder = Decoder(vocab_size=vocab_size, d_model=d_model, hidden_dim=lstm_hidden_dim, n_layers=n_lstm_layers)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.mel_spectrogram(x)
        if lengths is not None:
            lengths = (lengths // self.hop_length) + 1
        if self.training:
            x = self.spec_augment(x)
        x, lengths = self.encoder(x, lengths)
        x = self.decoder(x, lengths)
        return x, lengths
    
    @torch.inference_mode()
    def infer(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.forward(x, lengths)