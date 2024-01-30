import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import torch
from fastapi import FastAPI, UploadFile, File
from model.conformer import Conformer
from pydub import AudioSegment
from io import BytesIO
from processing.processor import ConformerProcessor
import time
import uvicorn
import fire

MAX_AUDIO_VALUE = 32768

def read_audio(data: bytes, sampling_rate: int):
    audio = AudioSegment.from_file(BytesIO(data)).set_frame_rate(sampling_rate).get_array_of_samples()
    signal = torch.Tensor(audio) / MAX_AUDIO_VALUE
    return signal

def create_app(checkpoint: str,
               # Processor Config
               vocab_path: str,
               arpa_path: str,
               pad_token: str = "<pad>",
               unk_token: str = "<unk>", 
               word_delim_token: str = "|",
               sampling_rate: int = 16000, 
               num_mels: int = 80,
               fft_size: int = 400, 
               hop_length: int = 160, 
               win_length: int = 400,
               fmin: float = 0.0,
               fmax: float = 8000.0,
               beam_alpha: float = 2.1, 
               beam_beta: float = 9.2,
               encoder_n_layers: int = 17,
               encoder_dim: int = 512,
               heads: int = 8,
               kernel_size: int = 31,
               decoder_n_layers: int = 1,
               decoder_dim: int = 640,
               dropout_rate: float = 0.0,
               device: str = "cuda") -> FastAPI:
    
    if device != "cpu":
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    
    app = FastAPI()

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
        lm_path=arpa_path,
        beam_alpha=beam_alpha,
        beam_beta=beam_beta
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
    )

    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'])
    model.to(device)
    model.eval()

    @app.post("/s2t")
    async def _(file: UploadFile = File(...)):
        try:
            start_time = time.time()

            data = await file.read()

            signal = read_audio(data, sampling_rate)
            mel = processor.mel_spectrogram(signal).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(mel)

            text = processor.decode_beam_search(logits[0].cpu().numpy())

            end_time = time.time()

            return {
                "transcription": text,
                "processing_time": end_time - start_time
            }
        except Exception as e:
            print(str(e))
            return {'error': str(e)}
        
    return app
    

def main(checkpoint: str,
        # Processor Config
        vocab_path: str,
        arpa_path: str,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>", 
        word_delim_token: str = "|",
        sampling_rate: int = 16000, 
        num_mels: int = 80,
        fft_size: int = 400, 
        hop_length: int = 160, 
        win_length: int = 400,
        fmin: float = 0.0,
        fmax: float = 8000.0,
        beam_alpha: float = 2.1, 
        beam_beta: float = 9.2,
        encoder_n_layers: int = 17,
        encoder_dim: int = 512,
        heads: int = 8,
        kernel_size: int = 31,
        decoder_n_layers: int = 1,
        decoder_dim: int = 640,
        dropout_rate: float = 0.0,
        device: str = "cuda",
        # API Config
        host: str = "0.0.0.0",
        port: int = 8000):
    
    app = create_app(
        checkpoint=checkpoint,
        vocab_path=vocab_path,
        pad_token=pad_token,
        unk_token=unk_token,
        word_delim_token=word_delim_token,
        arpa_path=arpa_path,
        beam_alpha=beam_alpha,
        beam_beta=beam_beta,
        sampling_rate=sampling_rate,
        num_mels=num_mels,
        fft_size=fft_size,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
        encoder_n_layers=encoder_n_layers,
        encoder_dim=encoder_dim,
        heads=heads,
        kernel_size=kernel_size,
        decoder_n_layers=decoder_n_layers,
        decoder_dim=decoder_dim,
        dropout_rate=dropout_rate,
        device=device
    )
    
    uvicorn.run(app, host=host, port=port)

# Start FastAPI App
if __name__ == '__main__':
    fire.Fire(main)