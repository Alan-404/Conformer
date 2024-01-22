import os
from src.conformer import Conformer
from preprocessing.processor import ConformerProcessor
import torch

# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def build_onnx(
        checkpoint: str,
        vocab_path: str, 
        audio_path: str,
        saved_path: str,
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
        dropout_rate: float = 0.1
    ):

    # Processor Setup
    processor = ConformerProcessor(
        vocab_path=vocab_path,
        num_mels=num_mels,
        sampling_rate=sampling_rate,
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
        pad_token=pad_token,
        unk_token=unk_token,
        word_delim_token=word_delim_token
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

    model.load_state_dict(torch.load(checkpoint, map_location=device)['model'])
    model.eval()

    dummy_input = processor.mel_spectrogram(processor.load_audio(audio_path)).unsqueeze(0).to(device)

    torch.onnx.export(model, 
                    dummy_input, 
                    f=saved_path, 
                    input_names=['input'],
                    output_names=['output'],
                    verbose=True,
                    dynamic_axes={
                        'input': {
                            2: 'mel_time'
                        },
                        'output': {
                            1: 'output_length'
                        }
                    }
                )
    
if __name__ == "__main__":
    import fire

    fire.Fire(build_onnx)
