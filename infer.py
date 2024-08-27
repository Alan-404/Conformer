import torch

from model.conformer import Conformer
from processing.processor import ConformerProcessor
from processing.lm import KenLanguageModel

import librosa
import numpy as np

from typing import Optional, Union

def predict(
        model: Conformer,
        lm: KenLanguageModel,
        processor: ConformerProcessor,
        audio: Union[torch.Tensor, np.ndarray, str],
        sr: int = 16000,
        device: Union[str, int] = 'cuda'
    ):
    if isinstance(audio, str):
        audio, _ = librosa.load(audio, sr=sr)
        audio = torch.tensor(audio, device=device)
    elif isinstance(audio, np.ndarray):
        audio = torch.tensor(audio, device=device)
    elif audio.device == 'cpu' and device != 'cpu':
        audio = audio.to(device)

    

def infer(path: str):
    pass