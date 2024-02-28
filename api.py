import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import torch
from fastapi import FastAPI, UploadFile, File
from pydub import AudioSegment
from io import BytesIO
from processing._processor import ConformerProcessor
from model.conformer import Conformer
import time
import uvicorn
import fire

MAX_AUDIO_VALUE = 32768

def read_audio(data: bytes, sampling_rate: int):
    audio = AudioSegment.from_file(BytesIO(data)).set_frame_rate(sampling_rate).get_array_of_samples()
    signal = torch.tensor(audio) / MAX_AUDIO_VALUE
    return signal

def create_app(model_path: str,
               device: str = "cuda") -> FastAPI:
    
    if device != "cpu":
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path)
    
    app = FastAPI()

    processor = ConformerProcessor(**checkpoint['processor_params'])

    model = Conformer(**checkpoint['hyper_params'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    @app.post("/s2t")
    async def _(file: UploadFile = File(...)):
        try:
            read_audio_start = time.time()

            data = await file.read()

            signal = read_audio(data, processor.sampling_rate)
            mel = processor.mel_spectrogram(signal).unsqueeze(0).to(device)

            read_audio_end = time.time()

            infer_start = time.time()

            with torch.inference_mode():
                logits = model(mel)

            infer_end = time.time()

            beam_start = time.time()

            text = processor.decode_beam_search(logits[0].cpu().numpy())

            beam_end = time.time()

            del mel
            del logits
            torch.cuda.empty_cache()

            return {
                "transcription": text,
                "read_audio": read_audio_end - read_audio_start,
                "inference": infer_end - infer_start,
                "beam_search": beam_end - beam_start
            }
        except Exception as e:
            print(str(e))
            return {'error': str(e)}
        
    return app
    

def main(model_path: str,
        device: str = "cuda",
        # API Config
        host: str = "0.0.0.0",
        port: int = 8000):
    
    app = create_app(
        model_path=model_path,
        device=device
    )
    
    uvicorn.run(app, host=host, port=port)

# Start FastAPI App
if __name__ == '__main__':
    fire.Fire(main)