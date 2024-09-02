from torch.utils.data import Dataset
from processing.processor import ConformerProcessor

from typing import Optional, Tuple, List, Union, Dict, Literal
import torch

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from torchaudio.transforms import SpecAugment
import numpy as np
import librosa

class InferenceDataset(Dataset):
    def __init__(self, prompts: pd.DataFrame, sr: int = 16000, load_type: str = 'staff', device: Union[str, int] = 'cuda') -> None:
        super().__init__()
        self.prompts = prompts
        self.sr = sr

        self.current_path = None
        self.current_audio = None

        self.type_load = 0 if load_type == 'staff' else 1
        self.device = device

    def __len__(self) -> int:
        return len(self.prompts)
    
    def load_audio(self, path: str) -> None:
        signal, _ = librosa.load(path, sr=self.sr, mono=False)
        self.current_audio = signal[self.type_load]

    def split_segment(self, audio: np.ndarray, start: float, end: float) -> np.ndarray:
        return audio[int(float(start * self.sr)) : int(float(end * self.sr))]
    
    def __getitem__(self, index: int) -> torch.Tensor:
        index_df = self.prompts.iloc[index]
        if self.current_path is None or self.current_path != index_df['path']:
            self.current_path = index_df['path']
            self.load_audio(self.current_path)
        
        audio = self.split_segment(self.current_audio, index_df['StartSegment'], index_df['EndSegment'])
        audio = torch.tensor(audio, device=self.device)

        return audio

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

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, List[str]], torch.Tensor]:
        row = self.get_row_by_index(index)
        path = row['path']
        audio = self.processor.read_audio(path)

        if self.training:
            text = row['text'].split(" ")
            return audio, text
        else:
            return audio

class ConformerCollate:
    def __init__(self, processor: ConformerProcessor, collate_type: Literal['train', 'validate', 'test'] = 'train') -> None:
        self.processor = processor
        self.collate_type = collate_type
        self.run_augment = (collate_type == 'train')

    def __call__(self, batch: Tuple[torch.Tensor, List[str]]) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        if self.collate_type != 'test':
            audios, graphemes = zip(*batch)

            audios, audio_lengths = self.processor(audios, augment=self.run_augment)
            tokens, token_lengths = self.processor.as_target(graphemes)
            
            audio_lengths, sorted_indices = torch.sort(audio_lengths, descending=True)

            audios = audios[sorted_indices]
            tokens = tokens[sorted_indices]
            token_lengths = token_lengths[sorted_indices]

            return audios, tokens, audio_lengths, token_lengths
        else:
            audios, audio_lengths = self.processor(batch)
            audio_lengths, sorted_indices = torch.sort(audio_lengths, descending=True)
            audios = audios[sorted_indices]
            return audios, audio_lengths, torch.argsort(sorted_indices, descending=False)