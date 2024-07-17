from torch.utils.data import Dataset
from processing.processor import ConformerProcessor
from processing.target import TargetConformerProcessor

from typing import Optional, Tuple, List, Union
import torch

import pyarrow.parquet as pq

from torchaudio.transforms import SpecAugment

class ConformerDataset(Dataset):
    def __init__(self, manifest_path: str, num_examples: Optional[int] = None) -> None:
        super().__init__()
        self.table = pq.read_table(manifest_path)
        if num_examples is not None:
            self.table = self.table.slice(0, num_examples)

    def __len__(self) -> int:
        return self.table.num_rows
    
    def get_row_by_index(self, index: int):
        return {col: self.table[col][index].as_py() for col in self.table.column_names}
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.get_row_by_index(index)
        return row['audio'], row['tokens']

class ConformerCollate:
    def __init__(self, processor: ConformerProcessor, handler: TargetConformerProcessor, training: bool = False) -> None:
        self.processor = processor
        self.training = training

        if self.training:
            self.augment = SpecAugment(
                n_time_masks=10,
                time_mask_param=35,
                n_freq_masks=10,
                freq_mask_param=35,
                p=0.05,
                zero_masking=True
            )

            self.handler = handler

    def __call__(self, batch: Tuple[torch.Tensor, List[str]]) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        if self.training:
            signals, graphemes = zip(*batch)

            signals, signal_lengths = self.processor(signals)
            tokens, token_lengths = self.handler(graphemes)

            signal_lengths, sorted_indices = torch.sort(signal_lengths, descending=True)

            signals = signals[sorted_indices]
            tokens = tokens[sorted_indices]
            token_lengths = token_lengths[sorted_indices]

            return signals, tokens, signal_lengths, token_lengths
        else:
            signals, signal_lengths = self.processor(batch)
            return signals, signal_lengths
            