import torch

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import pandas as pd

def create_pseudo_labels(checkpoint: str = "facebook/wav2vec2-large-960h"):
    model = Wav2Vec2ForCTC.from_pretrained(checkpoint).to("cuda")
    processor = Wav2Vec2Processor.from_pretrained(checkpoint)