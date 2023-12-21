from torch.utils.data import Dataset
from preprocessing.processor import ConformerProcessor
import pandas as pd
from typing import Optional, Tuple
import torch

class ConformerDataset(Dataset):
    def __init__(self, manifest_path: str, processor: ConformerProcessor, audio_path_col: str = "path", transcript_col: str = "text", num_examples: Optional[int] = None) -> None:
        super().__init__()
        self.prompts = pd.read_csv(manifest_path, sep="\t")
        self.columns = self.prompts.columns

        if num_examples is not None:
            self.prompts = self.prompts[:num_examples]

        if "start" in self.columns and "end" in self.columns:
            self.prompts['duration'] = self.prompts['end'] - self.prompts['start']
            self.prompts = self.prompts[(self.prompts['duration'] >= 0.3) & (self.prompts['duration'] <= 30)]
        
        if "type" not in self.columns:
            self.prompts['type'] = None

        assert audio_path_col in self.columns and transcript_col in self.columns

        self.audio_path_col = audio_path_col
        self.transcript_col = transcript_col

        self.processor = processor

    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        index_df = self.prompts.iloc[index]

        audio_path = index_df[self.audio_path_col]
        transcript = str(index_df[self.transcript_col])

        start = end = None
        if "start" in self.columns and "end" in self.columns:
            start = index_df["start"]
            end = index_df['end']
            
        role = None
        if index_df['type'] == "up":
            role = 0
        elif index_df['type'] == "down":
            role = 1

        return self.processor.load_audio(audio_path, start=start, end=end, role=role), transcript