import os
import torch
import fire

import torchsummary

from processing.processor import ConformerProcessor
from model.conformer import Conformer

from common import map_weights

def build_model(
        saved_path: str,
        checkpoint: str,
        vocab_path: str = None, 
        lm_path: str = None,
        pad_token: str = "<pad>", 
        unk_token: str = "<unk>", 
        word_delim_token: str = "|", 
        num_mels: int = 80, 
        sampling_rate: int = 16000, 
        fft_size: int = 400, 
        hop_length: int = 160, 
        win_length: int = 400, 
        fmin: float = 0.0, 
        fmax: float = 8000.0,
        # Model Hyper - Params
        n_blocks: int = 17,
        d_model: int = 512,
        heads: int = 8,
        kernel_size: int = 31,
        n_layers: int = 1,
        hidden_dim: int = 640,
        dropout_rate: float = 0.1,
    ):
    assert os.path.exists(checkpoint)

    processor = ConformerProcessor(
        vocab_path=vocab_path,
        unk_token=unk_token,
        pad_token=pad_token,
        word_delim_token=word_delim_token,
        sampling_rate=sampling_rate,
        num_mels=num_mels,
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
        lm_path=lm_path
    )

    model = Conformer(
        vocab_size=len(processor.dictionary.get_itos()),
        n_mel_channels=processor.num_mels,
        n_blocks=n_blocks,
        d_model=d_model,
        heads=heads,
        kernel_size=kernel_size,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate
    )

    hyper_params = {
        'vocab_size': len(processor.dictionary.get_itos()),
        'n_mel_channels': processor.num_mels,
        'n_blocks': n_blocks,
        'd_model': d_model,
        'heads': heads,
        'kernel_size': kernel_size,
        'n_layers': n_layers,
        'hidden_dim': hidden_dim,
        'dropout_rate': dropout_rate
    }

    torchsummary.summary(model)

    model.load_state_dict(map_weights(torch.load(checkpoint, map_location='cpu')['state_dict']))
    model.eval()

    torch.save({
        'hyper_params': hyper_params,
        'processor_params': processor.params,
        'state_dict': model.state_dict() 
    }, saved_path)


if __name__ == '__main__':
    fire.Fire(build_model)