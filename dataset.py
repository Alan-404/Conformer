from torch.utils.data import Dataset
from processing.processor import ConformerProcessor
import pandas as pd
from typing import Optional, Tuple
import torch
from tqdm import tqdm

class ConformerDataset(Dataset):
    def __init__(self, manifest_path: str, processor: ConformerProcessor, min_duration: float = 0.3, max_duration: float = 30.0, num_examples: Optional[int] = None, make_grapheme: bool = False) -> None:
        super().__init__()
        self.prompts = pd.read_csv(manifest_path, sep="\t")
        self.columns = self.prompts.columns

        self.prompts['text'] = self.prompts['text'].fillna('')

        if num_examples is not None:
            self.prompts = self.prompts[:num_examples]

        if "start" in self.columns and "end" in self.columns:
            self.prompts['duration'] = self.prompts['end'] - self.prompts['start']
            self.prompts = self.prompts[(self.prompts['duration'] >= min_duration) & (self.prompts['duration'] <= max_duration)].reset_index(drop=True)
        
        if "type" not in self.columns:
            self.prompts['type'] = None

        self.processor = processor

        if 'grapheme' not in self.prompts.columns or make_grapheme:
            print("Converting Text to Graphemes...")
            graphemes = []
            sentences = self.prompts['text'].to_list()
            for sentence in tqdm(sentences):
                graphemes_ = self.processor.sentence2graphemes(sentence)
                graphemes.append(" ".join(graphemes_))

            self.prompts['grapheme'] = graphemes

            cols = ['path', 'text','grapheme']
            if 'start' in self.prompts.columns and 'end' in self.prompts.columns:
                cols = ['path', 'text','start', 'end', 'type', 'grapheme']

            self.prompts[cols].to_csv(manifest_path, sep="\t", index=False)

    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        index_df = self.prompts.iloc[index]

        audio_path = index_df['path']
        transcript = index_df['grapheme']
        if type(transcript) != str:
            transcript = ['']
        else:
            transcript = transcript.split(" ")

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
    
class UnsupervisedConformerDataset(Dataset):
    def __init__(self, manifest_path: str, processor: ConformerProcessor, min_duration: float = 0.3, max_duration: float = 30.0, num_examples: Optional[int] = None, make_grapheme: bool = False) -> None:
        super().__init__()
        self.prompts = pd.read_csv(manifest_path, sep="\t")
        self.columns = self.prompts.columns

        if num_examples is not None:
            self.prompts = self.prompts[:num_examples]

        if "start" in self.columns and "end" in self.columns:
            self.prompts['duration'] = self.prompts['end'] - self.prompts['start']
            self.prompts = self.prompts[(self.prompts['duration'] >= min_duration) & (self.prompts['duration'] <= max_duration)].reset_index(drop=True)
        
        if "type" not in self.columns:
            self.prompts['type'] = None

        self.processor = processor

    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        index_df = self.prompts.iloc[index]

        audio_path = index_df['path']

        start = end = None
        if "start" in self.columns and "end" in self.columns:
            start = index_df["start"]
            end = index_df['end']
            
        role = None
        if index_df['type'] == "up":
            role = 0
        elif index_df['type'] == "down":
            role = 1

        return self.processor.load_audio(audio_path, start=start, end=end, role=role)

class ConformerInferenceDataset(Dataset):
    def __init__(self, manifest_path: str, processor: ConformerProcessor, num_examples: Optional[int] = None) -> None:
        super().__init__()
        self.prompts = pd.read_csv(manifest_path, sep="\t")
        self.columns = self.prompts.columns
    
        if num_examples is not None:
            self.prompts = self.prompts[:num_examples]

        self.processor = processor

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index: int):
        index_df = self.prompts.iloc[index]

        audio_path = index_df['path']

        start = end = None
        if "start" in self.columns and "end" in self.columns:
            start = index_df["start"]
            end = index_df['end']
            
        role = None
        if 'type' in self.columns:
            if index_df['type'] == "up":
                role = 0
            elif index_df['type'] == "down":
                role = 1

        signal = self.processor.load_audio(audio_path, start=start, end=end, role=role)

        return signal

class CharDataset(Dataset):
    def __init__(self, manifest_path: str, processor: ConformerProcessor, num_examples: Optional[int] = None) -> None:
        super().__init__()
        self.prompts = pd.read_csv(manifest_path, sep="\t")
        self.columns = self.prompts.columns

        self.prompts['text'] = self.prompts['text'].fillna('')

        if num_examples is not None:
            self.prompts = self.prompts[:num_examples]

        if "start" in self.columns and "end" in self.columns:
            self.prompts['duration'] = self.prompts['end'] - self.prompts['start']
            self.prompts = self.prompts[(self.prompts['duration'] >= 0.3) & (self.prompts['duration'] <= 30)]
        
        if "type" not in self.columns:
            self.prompts['type'] = None

        self.processor = processor

    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        index_df = self.prompts.iloc[index]

        audio_path = index_df['path']
        transcript = index_df['text']

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