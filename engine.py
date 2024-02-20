import torch
from torch.utils.data import DataLoader

from ignite.engine import Engine, Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from model.conformer import Conformer
from processing.processor import ConformerProcessor

from dataset import ConformerInferenceDataset

from common import map_weights

import torchsummary

from typing import Tuple

class __ConformerEngine:
    def __init__(self,
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
                device: str = 'cuda',
                batch_size: int = 1,
                num_workers: int = 1) -> None:
        
        self.processor = ConformerProcessor(
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
        self.model = Conformer(
            vocab_size=len(self.processor.dictionary),
            n_mel_channels=num_mels,
            n_blocks=n_blocks,
            d_model=d_model,
            heads=heads,
            kernel_size=kernel_size,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )

        self.model.load_state_dict(map_weights(torch.load(checkpoint, map_location='cpu')['state_dict']))

        torchsummary.summary(self.model)
        self.model.eval()

        self.model.to(device)

        self.engine = Engine(self.infer_step)
        ProgressBar().attach(self.engine)

        self.engine.add_event_handler(Events.ITERATION_COMPLETED, self.get_output)

        self.predicts = []

        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

    def infer_step(self, _: Engine, batch: Tuple[torch.Tensor]) -> None:
        mels, lengths, sorted_indexes = batch[0].to(self.device), batch[1].to(self.device), batch[2]
        
        with torch.inference_mode():
            outputs, lengths = self.model(mels, lengths)
        
        origin_indexes = torch.argsort(sorted_indexes)
        outputs = outputs[origin_indexes]
        lengths = lengths[origin_indexes].cpu().numpy()
        
        outputs = outputs.cpu().numpy()
        
        for index, output in enumerate(outputs):
            output = output[:lengths[index]]
            self.predicts.append(self.processor.decode_beam_search(output))

    def get_batch(self, signals: torch.Tensor):
        mels, mel_lengths = self.processor(signals, return_length=True)

        sorted_indexes = torch.argsort(mel_lengths, descending=True)
        mel_lengths = mel_lengths[sorted_indexes]
        mels = mels[sorted_indexes]

        return mels, mel_lengths, sorted_indexes
    
    def infer_csv(self, file_path: str):
        dataset = ConformerInferenceDataset(file_path, self.processor)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.get_batch, num_workers=self.num_workers)

        self.engine.run(dataloader, max_epochs=1)

        result = self.predicts
        self.predicts.clear()

        return result
    
    def get_output(self, engine: Engine):
        print(engine.state.output)