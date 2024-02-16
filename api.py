import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import torch
from fastapi import FastAPI, UploadFile, File
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

    model = torch.jit.load(checkpoint, map_location=device)
    model.eval()

    @app.post("/s2t")
    async def _(file: UploadFile = File(...)):
        try:
            read_audio_start = time.time()

            data = await file.read()

            signal = read_audio(data, sampling_rate)
            mel = processor.mel_spectrogram(signal).unsqueeze(0).to(device)

            read_audio_end = time.time()

            infer_start = time.time()

            with torch.inference_mode():
                logits = model(mel)
                torch.cuda.synchronize()

            infer_end = time.time()

            beam_start = time.time()

            text = processor.decode_beam_search(logits[0].cpu().numpy())

            beam_end = time.time()

            return {
                "transcription": text,
                "audio_time": len(signal) / sampling_rate,
                "read_audio": read_audio_end - read_audio_start,
                "inference": infer_end - infer_start,
                "beam_search": beam_end - beam_start
            }
        except Exception as e:
            print(str(e))
            return {'error': str(e)}
        
    return app
    

def main(model: str,
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
        # n_blocks: int = 17,
        # d_model: int = 512,
        # heads: int = 8,
        # kernel_size: int = 31,
        # n_layers: int = 1,
        # hidden_dim: int = 640,
        # dropout_rate: float = 0.0,
        device: str = "cuda",
        # API Config
        host: str = "0.0.0.0",
        port: int = 8000):
    
    app = create_app(
        checkpoint=model,
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
        device=device
    )
    
    uvicorn.run(app, host=host, port=port)

# Start FastAPI App
if __name__ == '__main__':
    fire.Fire(main)