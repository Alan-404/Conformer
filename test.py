import os
import torch
from torch.utils.data import DataLoader

import torchsummary

from ignite.engine import Engine, Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from tqdm import tqdm

import fire

from processing.processor import ConformerProcessor
from model.conformer import Conformer
from dataset import ConformerInferenceDataset

from evaluation import ConformerMetric

from common import map_weights
from typing import Tuple, Optional

import time

def test(result_folder: str,
         test_path: str,
         vocab_path: str,
         checkpoint: str,
         arpa_path: Optional[str] = None,
         num_mels: int = 80,
         sampling_rate: int = 16000,
         fft_size: int = 400,
         hop_length: int = 160,
         win_length: int = 400,
         fmin: float = 0.0,
         fmax: float = 8000.0,
         pad_token: str = "<pad>",
         unk_token: str = "<unk>",
         word_delim_token: str = "|",
         n_blocks: int = 17,
         d_model: int = 512,
         heads: int = 8,
         kernel_size: int = 31,
         dropout_rate: float = 0.0,
         batch_size: int = 1,
         num_workers: int = 1,
         device: str = 'cuda',
         num_examples: int = None):
    assert os.path.exists(test_path) and os.path.exists(checkpoint)

    if os.path.exists(result_folder) == False:
        os.mkdir(result_folder)

    # Device Config
    if device == 'cpu' or torch.cuda.is_available() == False:
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cuda')

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
        word_delim_token=word_delim_token,
        lm_path=arpa_path
    )

    # Model Setup
    model = Conformer(
        vocab_size=len(processor.dictionary),
        n_mel_channels=processor.num_mels,
        n_blocks=n_blocks,
        d_model=d_model,
        heads=heads,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate
    ).to(device)
    model.eval()

    torchsummary.summary(model)

    checkpoint = torch.load(checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(map_weights(checkpoint['state_dict']))
        
    metric = ConformerMetric()

    def get_batch(signals: torch.Tensor):
        mels, mel_lengths = processor(signals)
        return mels, mel_lengths

    dataset = ConformerInferenceDataset(manifest_path=test_path, processor=processor, num_examples=num_examples)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=get_batch, num_workers=num_workers)

    predicts = []
    labels = dataset.get_labels()

    start_time = time.time()

    for data in tqdm(dataloader):
        inputs = data[0].to(device)
        lengths = data[1].to(device)

        with torch.inference_mode():
            outputs, lengths = model(inputs, lengths)

        outputs = outputs.cpu().numpy()
        lengths = lengths.cpu().numpy()

        for i in range(len(outputs)):
            item = outputs[i][:lengths[i]]

            if arpa_path is not None:
                sentence = processor.decode_beam_search(item)
            else:
                item = torch.argmax(item)
                sentence = processor.decode(item)

            predicts.append(sentence)

    end_time = time.time()

    print(f"WER Score: {metric.wer_score(predicts, labels)}")
    print(f"CER Score: {metric.cer_score(predicts, labels)}")
    
    print(f"Inference Time: {end_time - start_time}")
    print("Done Inference")
        
if __name__ == '__main__':
    fire.Fire(test)