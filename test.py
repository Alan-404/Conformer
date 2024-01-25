import os
import torch
from torch.utils.data import DataLoader

from ignite.engine import Engine, Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from dataset import ConformerTestDataset

import fire

from preprocessing.processor import ConformerProcessor
from model.conformer import Conformer

from typing import Tuple
from module import ConformerMetric
from common import map_weights

def test(result_folder: str,
         test_path: str,
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
         encoder_n_layers: int = 17,
         encoder_dim: int = 512,
         heads: int = 8,
         kernel_size: int = 31,
         decoder_n_layers: int = 1,
         decoder_dim: int = 640,
         dropout_rate: float = 0.0,
         batch_size: int  = 1,
         num_examples: int = None,
         saved_name: str = None):
    if os.path.exists(result_folder) == False:
        os.mkdir(result_folder)

    # Device Config
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

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
        encoder_n_layers=encoder_n_layers,
        encoder_dim=encoder_dim,
        heads=heads,
        kernel_size=kernel_size,
        decoder_n_layers=decoder_n_layers,
        decoder_dim=decoder_dim,
        dropout_rate=dropout_rate
    ).to(device)

    model.load_state_dict(map_weights(torch.load(checkpoint, map_location='cpu')['state_dict']))
    model.to(device)
    model.eval()

    metric = ConformerMetric()

    def get_batch(batch) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        signals = zip(*batch)
        mels, mel_lengths = processor(signals, return_length=True)

        return mels, mel_lengths

    dataset = ConformerTestDataset(test_path, processor, num_examples=num_examples)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=get_batch)

    labels = dataset.prompts['text'].to_list()
    preds = []

    def test_step(_: Engine, batch: Tuple[torch.Tensor]):
        inputs = batch[0].to(device)
        input_lengths = batch[1].to(device)
        
        with torch.no_grad():
            outputs, output_lengths = model(inputs, input_lengths)

        for logit, index in enumerate(outputs):
            preds.append(processor.decode_beam_search(logit[:output_lengths[index]]))

    
    tester = Engine(test_step)
    ProgressBar().attach(tester)

    @tester.on(Events.COMPLETED)
    def _ (_: Engine):
        print(f"WER score: {metric.wer_score(preds, labels)}")

    tester.run(dataloader, max_epochs=1)

    if saved_name is not None:
        saved_filename = saved_name
    else:
        test_name = os.path.basename(test_path)
        saved_filename = f"result_{test_name}"

    df = dataset.prompts
    df['pred'] = preds

    df.to_csv(f"{result_folder}/{saved_filename}", sep="\t", index=False)

if __name__ == '__main__':
    fire.Fire(test)