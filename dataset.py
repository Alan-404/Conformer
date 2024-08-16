from torch.utils.data import Dataset
from processing.processor import ConformerProcessor
from processing.assessor import ConformerAssessor

from typing import Optional, Tuple, List, Union, Dict
import torch

import pyarrow.parquet as pq

from torchaudio.transforms import SpecAugment

class ConformerDataset(Dataset):
    def __init__(self, manifest_path: str, processor: ConformerProcessor, assessor: Optional[ConformerAssessor] = None, num_examples: Optional[int] = None) -> None:
        super().__init__()
        self.table = pq.read_table(manifest_path)
        if num_examples is not None:
            self.table = self.table.slice(0, num_examples)

        self.processor = processor
        self.training = assessor is not None
        self.assessor = assessor

    def __len__(self) -> int:
        return self.table.num_rows
    
    def get_row_by_index(self, index: int) -> Dict[str, str]:
        return {col: self.table[col][index].as_py() for col in self.table.column_names}

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        row = self.get_row_by_index(index)

        path = row['path']
        audio = self.processor.load_audio(path)

        text = row['text'].split(" ")
        
        return audio, text

class ConformerCollate:
    def __init__(self, processor: ConformerProcessor, assessor: Optional[ConformerAssessor] = None) -> None:
        self.processor = processor
        self.training = assessor is not None
        self.assessor = assessor

        if self.training:
            self.augment = SpecAugment(
                n_time_masks=10,
                time_mask_param=35,
                n_freq_masks=10,
                freq_mask_param=35,
                p=0.05,
                zero_masking=True
            )

    def __call__(self, batch: Tuple[torch.Tensor, List[str]]) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        if self.training:
            audios, graphemes = zip(*batch)

            audios, audio_lengths = self.processor(audios)
            tokens, token_lengths = self.assessor(graphemes)

            audio_lengths, sorted_indices = torch.sort(audio_lengths, descending=True)

            audios = audios[sorted_indices]
            tokens = tokens[sorted_indices]
            token_lengths = token_lengths[sorted_indices]

            return audios, tokens, audio_lengths, token_lengths
        else:
            audios, audio_lengths = self.processor(batch)
            return audios, audio_lengths
            