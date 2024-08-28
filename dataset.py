import os
from torch.utils.data import Dataset
from processing.processor import ConformerProcessor

from typing import Optional, Tuple, List, Union, Dict
import torch

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from torchaudio.transforms import SpecAugment

class ConformerDataset(Dataset):
    def __init__(self, manifest: Union[str, pd.DataFrame, pa.Table], processor: ConformerProcessor, training: bool = False, num_examples: Optional[int] = None) -> None:
        super().__init__()
        if isinstance(manifest, str):
            if ".parquet" in manifest:
                self.table = pq.read_table(manifest)
            else:
                self.table = pa.Table.from_pandas(pd.read_csv(manifest))
        else:
            if isinstance(manifest, pd.DataFrame):
                self.table = pa.Table.from_pandas(manifest)
            else:
                self.table = manifest
        
        if num_examples is not None:
            self.table = self.table.slice(0, num_examples)

        self.processor = processor
        self.training = training

    def __len__(self) -> int:
        return self.table.num_rows
    
    def get_row_by_index(self, index: int) -> Dict[str, str]:
        return {col: self.table[col][index].as_py() for col in self.table.column_names}

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, str], str]:
        row = self.get_row_by_index(index)
        path = row['path']
        audio = self.processor.read_audio(path)

        if self.training:
            text = row['text'].split(" ")
            return audio, text
        else:
            return audio

class ConformerCollate:
    def __init__(self, processor: ConformerProcessor, device: Union[str, int] = 'cpu', training: bool = False) -> None:
        self.processor = processor
        self.device = device
        self.training = training

        if self.training:
            self.augment = SpecAugment(
                n_time_masks=10,
                time_mask_param=35,
                n_freq_masks=10,
                freq_mask_param=35,
                p=0.05,
                zero_masking=True
            ).to(self.device)

    def __call__(self, batch: Tuple[torch.Tensor, List[str]]) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        if self.training:
            audios, graphemes = zip(*batch)

            audios, audio_lengths = self.processor(audios)
            tokens, token_lengths = self.processor.as_target(graphemes)
            
            audio_lengths, sorted_indices = torch.sort(audio_lengths, descending=True)

            audios = audios[sorted_indices]
            # audios = self.augment(audios)
            tokens = tokens[sorted_indices]
            token_lengths = token_lengths[sorted_indices]

            return audios, tokens, audio_lengths, token_lengths
        else:
            audios, audio_lengths = self.processor(batch)
            audio_lengths, sorted_indices = torch.sort(audio_lengths, descending=True)
            audios = audios[sorted_indices]
            return audios, audio_lengths, sorted_indices