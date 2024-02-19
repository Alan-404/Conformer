import os
import torch
from torch.utils.data import DataLoader

import torchsummary

import io

from ignite.engine import Engine, Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar

import fire

from processing.processor import ConformerProcessor
from model.conformer import Conformer
from dataset import ConformerInferenceDataset

from evaluation import ConformerMetric

from common import map_weights
from typing import Tuple


def test(result_folder: str,
         test_path: str,
         result_path: str,
         vocab_path: str,
         arpa_path: str,
         checkpoint: str,
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
         n_layers: int = 1,
         hidden_dim: int = 640,
         dropout_rate: float = 0.0,
         batch_size: int = 1,
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
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate
    ).to(device)

    checkpoint = torch.load(checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(map_weights(checkpoint['state_dict']))
        
    model.to(device)

    metric = ConformerMetric()

    def get_batch(signals: torch.Tensor):
        if batch_size != 1:
            mels, mel_lengths = processor(signals, return_length=True)
            return mels, mel_lengths
        
        mels = processor(signals, return_length=False)
        return mels

    dataset = ConformerInferenceDataset(manifest_path=test_path, processor=processor, num_examples=num_examples)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=get_batch)

    predicts = []

    def infer_step(_: Engine, batch: Tuple[torch.Tensor]) -> None:
        if batch_size == 1:
            mels = batch[0].unsqueeze(0).to(device)
            lengths = None
        else:
            mels, lengths = batch[0].to(device), batch[1].to(device)
        
        with torch.inference_mode():
            outputs = model(mels, lengths)
        
        outputs = outputs.cpu().numpy()
        if batch_size != 1:
            lengths = lengths.cpu().numpy()
            
        for index, output in enumerate(outputs):
            if lengths is not None:
                output = output[:lengths[index]]
            predicts.append(processor.decode_beam_search(output))

    engine = Engine(infer_step)
    ProgressBar().attach(engine)

    @engine.on(Events.STARTED)
    def _ (_: Engine):
        torchsummary.summary(model)

    @engine.on(Events.COMPLETED)
    def _(_: Engine):
        answers = io.open(result_path).read().strip().split("\n")
        answers = io.open(result_path).read().strip().split("\n")
        print(f"WER Score: {metric.wer_score(predicts, answers)}")
        
        df = dataset.prompts
        df['pred'] = predicts

        filename = os.path.basename(test_path)
        df.to_csv(f"{result_folder}/{filename}", index=False, sep="\t")

    engine.run(dataloader, max_epochs=1)
    print("Done Inference")
        
if __name__ == '__main__':
    fire.Fire(test)