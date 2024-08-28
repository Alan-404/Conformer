import torch

from model.conformer import Conformer
from processing.processor import ConformerProcessor
from processing.lm import KenLanguageModel

import librosa
import numpy as np

from typing import Optional, Union


def infer(path: str):
    pass