import os
import torch

import fire

import torchsummary

from processing.processor import ConformerProcessor
from model.conformer import Conformer

import pickle


def build_model(
        saved_folder: str,
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
        encoder_n_layers: int = 17,
        encoder_dim: int = 512,
        heads: int = 8,
        kernel_size: int = 31,
        decoder_n_layers: int = 1,
        decoder_dim: int = 640,
        dropout_rate: float = 0.1,
    ):
    assert os.path.exists(checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

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
        encoder_n_layers=encoder_n_layers,
        encoder_dim=encoder_dim,
        heads=heads,
        kernel_size=kernel_size,
        decoder_n_layers=decoder_n_layers,
        decoder_dim=decoder_dim,
        dropout_rate=dropout_rate
    ).to(device)

    torchsummary.summary(model)

    model.load_state_dict(torch.load(checkpoint, map_location=device)['model'])
    model.eval()

    if os.path.exists(saved_folder) == False:
        os.mkdir(saved_folder)

    with open(f"{saved_folder}/model.bin", 'wb') as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

    

