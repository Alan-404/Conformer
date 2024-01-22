import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import torch
from fastapi import FastAPI, UploadFile, File
from src.conformer import Conformer
from pydub import AudioSegment
from io import BytesIO
from preprocessing.processor import ConformerProcessor
import time
import uvicorn

MAX_AUDIO_VALUE = 32768

def create_app(checkpoint: str,
               vocab_path: str,
               arpa_path: str,
               sampling_rate: int):
    app = FastAPI()

    processor = ConformerProcessor(vocab_path, lm_path=arpa_path)

    def read_audio(data: bytes):
        audio = AudioSegment.from_file(BytesIO(data)).set_frame_rate(sampling_rate).get_array_of_samples()
        signal = torch.Tensor(audio) / MAX_AUDIO_VALUE
        return signal

    model = Conformer(
        vocab_size=len(processor.dictionary),
        n_mel_channels=80,
        encoder_n_layers=17,
        encoder_dim=512,
        heads=8,
        kernel_size=31
    )

    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'])
    model.to('cuda')
    model.eval()

    @app.post("/s2t")
    async def _(file: UploadFile = File(...)):
        try:
            start_time = time.time()

            data = await file.read()

            signal = read_audio(data)
            mel = processor.mel_spectrogram(signal).unsqueeze(0).to('cuda')

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
        vocab_path: str,
        arpa_path: str,
        sampling_rate: int = 16000,
        host: str = '0.0.0.0', 
        port: int = 8000):
    
    app = create_app(
        checkpoint=checkpoint,
        vocab_path=vocab_path,
        arpa_path=arpa_path,
        sampling_rate=sampling_rate
    )
    
    uvicorn.run(app, host=host, port=port)

# Start FastAPI App
if __name__ == '__main__':
    import fire

    fire.Fire(main)