from torch.utils.data import Dataset
from processing.processor import ConformerProcessor
import pandas as pd
from typing import Optional, Tuple, List, Union
import torch
from tqdm import tqdm

class ConformerDataset(Dataset):
    def __init__(self, processor: ConformerProcessor, manifest_path: Optional[str] = None, prompts: Optional[pd.DataFrame] = None, training: bool = False, min_duration: float = 0.3, max_duration: float = 30.0, num_examples: Optional[int] = None, make_grapheme: bool = False) -> None:
        super().__init__()
        if manifest_path is not None:
            self.promts = pd.read_csv(manifest_path)
        elif prompts is not None:
            self.prompts = prompts
        else:
            raise("Invalid Dataset")

        self.prompts['text'] = self.prompts['text'].fillna('')

        if num_examples is not None:
            self.prompts = self.prompts[:num_examples]

        if "start" in self.prompts.columns and "end" in self.prompts.columns:
            self.prompts['duration'] = self.prompts['end'] - self.prompts['start']
            self.prompts = self.prompts[(self.prompts['duration'] >= min_duration) & (self.prompts['duration'] <= max_duration)].reset_index(drop=True)
        elif 'duration' in self.prompts.columns:
            self.prompts = self.prompts[(self.prompts['duration'] >= min_duration) & (self.prompts['duration'] <= max_duration)].reset_index(drop=True)
            
        if "type" not in self.prompts.columns:
            self.prompts['type'] = None

        self.processor = processor

        if training:
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

                self.prompts[cols].to_csv(manifest_path, index=False)
        
        self.training = training

    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, str], torch.Tensor]:
        index_df = self.prompts.iloc[index]

        audio_path = index_df['path']
        transcript = index_df['grapheme']
        if type(transcript) != str:
            graphemes = ['']
        else:
            graphemes = transcript.split(" ")

        start = end = None
        if "start" in self.prompts.columns and "end" in self.prompts.columns:
            start = index_df["start"]
            end = index_df['end']
            
        role = None
        if "role" in self.prompts.columns:
            if index_df['type'] == "up":
                role = 0
            elif index_df['type'] == "down":
                role = 1

        audio_path = index_df['path']

        start = end = None
        if "start" in self.prompts.columns and "end" in self.prompts.columns:
            start = index_df["start"]
            end = index_df['end']
            
        role = None
        if "role" in self.prompts.columns:
            if index_df['type'] == "up":
                role = 0
            elif index_df['type'] == "down":
                role = 1

        signal = self.processor.load_audio(audio_path, start=start, end=end, role=role),
        if self.training:
            transcript = index_df['grapheme']
            if type(transcript) != str:
                graphemes = ['']
            else:
                graphemes = transcript.split(" ")
            return signal, graphemes
        else:
            return signal

class ConformerCollate:
    def __init__(self, processor: ConformerProcessor, training: bool = False) -> None:
        self.processor = processor
        self.training = training

    def __call__(self, batch: Tuple[torch.Tensor, List[str]]) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        if self.training:
            signals, graphemes = zip(*batch)

            signals, signal_lengths = self.processor(signals)
            tokens, token_lengths = self.processor.as_target(graphemes)

            signal_lengths, sorted_indices = torch.sort(signal_lengths, descending=True)

            signals = signals[sorted_indices]
            tokens = tokens[sorted_indices]
            token_lengths = token_lengths[sorted_indices]

            return signals, tokens, signal_lengths, token_lengths
        else:
            signals, signal_lengths = self.processor(batch)
            return signals, signal_lengths
            