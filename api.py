import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import torch
from fastapi import FastAPI, UploadFile, File
from src.conformer import Conformer
import numpy as np
from pydub import AudioSegment
from io import BytesIO
from preprocessing.processor import ConformerProcessor
import time
import uvicorn

app = FastAPI()

processor = ConformerProcessor('./vocabulary/dictionary.json', lm_path='./lm/lm.arpa')

def read_audio(data: bytes):
    audio = AudioSegment.from_file(BytesIO(data)).set_frame_rate(16000).get_array_of_samples()
    signal = np.array(audio)/ 32768
    return torch.Tensor(signal)

model = Conformer(
    vocab_size=len(processor.dictionary),
    n_mel_channels=80,
    encoder_n_layers=17,
    encoder_dim=512,
    heads=8,
    kernel_size=31
)

model.load_state_dict(torch.load('./checkpoints/checkpoint_3.pt', map_location='cpu')['model'])
model.to('cuda')
model.eval()

@app.post("/s2t")
async def s2t(file: UploadFile = File(...)):
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
    
# Start FastAPI App
if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0', help='Host IP to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on')

    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)

