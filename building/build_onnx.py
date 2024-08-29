import torch

from model.conformer import Conformer
from processing.processor import ConformerProcessor

from checkpoint import load_model

from typing import Optional

def build_onnx(
        checkpoint: str,
        num_samples: Optional[int] = None,
        batch_size: int = 1,
        # Audio Config 
        n_mels: int = 80,
        # Text Config
        tokenizer_path: str = "./tokenizer/vi.json",
        pad_token: str = "<PAD>",
        delim_token: str = "|",
        unk_token: str = "<UNK>",
        # Model Config
        n_conformer_blocks: int = 17, 
        d_model: int = 512, 
        n_heads: int = 8, 
        kernel_size: int = 31,
        lstm_hidden_dim: int = 640,
        n_lstm_layers: int = 1,
        # Result Config
        device: str = 'cuda'
    ):
    processor = ConformerProcessor(
        tokenizer_path=tokenizer_path,
        pad_token=pad_token,
        delim_token=delim_token,
        unk_token=unk_token
    )

    model = Conformer(
        vocab_size=len(processor.vocab),
        n_mel_channels=n_mels,
        n_conformer_blocks=n_conformer_blocks,
        d_model=d_model,
        n_heads=n_heads,
        kernel_size=kernel_size,
        lstm_hidden_dim=lstm_hidden_dim,
        n_lstm_layers=n_lstm_layers,
        dropout_rate=0.0
    )

    load_model(checkpoint, model)
    model.to(device)
    model.eval()

    random_input = torch.rand((1, ))