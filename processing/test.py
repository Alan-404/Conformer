import os
import numpy as np
import json
from pydub import AudioSegment
import librosa
from typing import Union, Optional, List, Tuple, Dict
import re
import pickle
from torchaudio.transforms import MelSpectrogram
import torch
import torch.nn.functional as F
from torchtext.vocab import Vocab, vocab as create_vocab

from pyctcdecode import build_ctcdecoder

MAX_AUDIO_VALUE = 32768

class ConformerProcessor:
    def __init__(self, vocab_path: Optional[str] = None, unk_token: str = "<unk>", pad_token: str = "<pad>", word_delim_token: str = "|", sampling_rate: int = 16000, num_mels: int = 80, n_fft: int = 400, hop_length: int = 160, win_length: int = 400, fmin: float = 0.0, fmax: float = 8000.0, puncs: str = r"([:./,?!@#$%^&=`~*\(\)\[\]\"\-\\])", lm_path: Optional[str] = None, beam_alpha: float = 2.1, beam_beta: float = 9.2, device: str = 'cpu') -> None:
        self.params = {k: v for k, v in locals().items() if k != 'self'}
        
        if device != 'cpu':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Audio
        self.sampling_rate = sampling_rate
        self.num_mels = num_mels
        self.hop_length = hop_length

        self.mel_transform = MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=fmin,
            f_max=fmax,
            n_mels=num_mels
        ).to(self.device)

    def get_last_item(self, word: str, pattern: str):
        length_item = len(pattern)
        start_check = len(word) - length_item
        if word[start_check: ] == pattern:
            return word[:start_check ], pattern
        return None

    def suffix_handle(self, word, pattern):
        for key in pattern.keys():
            items = self.get_last_item(word, key)
            if items is not None:
                return items[0], pattern[key]
            
        return word, ''
    
    def stride_graphemes(self, text: str, patterns, n_grams: int = 4):
        if len(text) == 1:
            if text in patterns:
                return [text]
            return ["<unk>"]
        text = [text[i] for i in range(len(text) - 1, -1, -1)]
        text = "".join(text)
        graphemes = []
        start = 0
        if len(text) - 1 < n_grams:
            n_grams = len(text)
        num_steps = n_grams
        while start < len(text):
            found = True
            item = text[start:start + num_steps]

            item = [item[i] for i in range(len(item) - 1, -1, -1)]
            item = "".join(item)
                
            if item in patterns:
                graphemes.append(item)
            elif num_steps == 1:
                graphemes.append("<unk>")
            else:
                found = False

            if found:
                start += num_steps
                if len(text[start:]) < n_grams:
                    num_steps = len(text[start:])
                else:
                    num_steps = n_grams
            else:
                num_steps -= 1

        
        return [graphemes[i] for i in range(len(graphemes) - 1, -1, -1)]
    
    def split_handle(self, word: str, patterns: Dict[str, List[str]]):
        for item in patterns.keys():
            items = self.get_last_item(word, item)
            if items is not None:
                for pattern in patterns[item]:
                    if self.get_last_item(items[0], pattern) is not None:
                        return items[0], [*item]
                return items[0], [item]
        
        return word, []
    
    def concat_item(self, graphemes: List[str]):
        length = len(graphemes)
        if length <= 1:
            return graphemes
        
        for i in range(length - 1):
            if graphemes[i] == '':
                continue
            elif graphemes[i] == graphemes[i+1][0]:
                graphemes[i] = f"{graphemes[i]}{graphemes[i+1]}"
                graphemes[i+1] = ''

        while True:
            if '' in graphemes:
                graphemes.remove('')
            else:
                break
        
        return graphemes
    
    def split_item(self, graphemes: List[str], patterns: List[str], vowels: List[str]):
        length = len(graphemes)
        if length <= 1:
            return graphemes
        
        items = []
        for i in range(length):
            if graphemes[i] in patterns:
                if i == 0 or i == length - 1:
                    items.append(graphemes[i])
                else:
                    if graphemes[i-1] in vowels and graphemes[i+1] in vowels:
                        items += [*graphemes[i]]
                    else:
                        items.append(graphemes)
            else:
                items.append(graphemes[i])

        return items
    
    def mixed_vowel_handle(self, graphemes: List[str], special_vowels: List[str], vowels: List[str]):
        length = len(graphemes)
        if length <= 1:
            return graphemes
        
        items = []
        for i in range(length):
            if graphemes[i] in special_vowels:
                if i == length - 1:
                    items.append(graphemes[i])
                else:
                    if graphemes[i+1] in vowels:
                        items += [*graphemes[i]]
                    else:
                        items.append(graphemes)
            else:
                items.append(graphemes[i])

        return items

    def split_by_condition(self, word: str, patterns: Dict[str, List[str]]):
        if len(word) == 1:
            return word, ''
        for key in patterns.keys():
            if word[-1] != key:
                continue
            else:
                if word[-2] == key:
                    return word, ''
                else:
                    for item in patterns[key]:
                        if self.get_last_item(word[:-1], item):
                            return word, ''
                    return word[:-1], key
                
        return word, ''
    
    def get_suffixes(self, word: str, pattern: List[Dict[str, List[str]]]):
        suffixes = []

        while True:
            items = self.suffix_handle(word, pattern)
            if items[1] == '':
                break
            
            word = items[0]
            suffixes.append(items[1])

        suffixes = [suffixes[i] for i in range(len(suffixes) - 1, -1, -1)]
        return word, suffixes

    def read_pickle(self, path: str) -> np.ndarray:
        with open(path, 'rb') as file:
            signal = pickle.load(file)

        signal = librosa.resample(y=signal, orig_sr=8000, target_sr=self.sampling_rate)

        return signal
    
    def read_pcm(self, path: str) -> np.ndarray:
        audio = AudioSegment.from_file(path, frame_rate=8000, channels=1, sample_width=2).set_frame_rate(self.sampling_rate).get_array_of_samples()
        return np.array(audio).astype(np.float64) / MAX_AUDIO_VALUE
    
    def read_audio(self, path: str, role: Optional[int] = None) -> np.ndarray:
        if role is not None:
            signal, _ = librosa.load(path, sr=self.sampling_rate, mono=False)
            signal = signal[role]
        else:
            signal, _ = librosa.load(path, sr=self.sampling_rate, mono=True)

        return signal
    
    def spectral_normalize(self, x: torch.Tensor, C: int = 1, clip_val: float = 1e-5) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=clip_val) * C)

    def mel_spectrogram(self, signal: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel_transform(signal)
        log_mel = self.spectral_normalize(mel_spec)
        return log_mel
    
    def split_segment(self, signal: torch.Tensor, start: float, end: float):
        return signal[int(start * self.sampling_rate) : int(end * self.sampling_rate)]

    def load_audio(self, path: str, start: Optional[float] = None, end: Optional[float] = None, role: Optional[int] = None) -> torch.Tensor:
        if ".pickle" in path:
            signal = self.read_pickle(path)
        elif ".pcm" in path:
            signal = self.read_pcm(path)
        else:
            signal = self.read_audio(path, role)

        if start is not None and end is not None:
            signal = self.split_segment(signal, start, end)

        signal = torch.FloatTensor(signal)
        signal = signal.to(self.device)
        return signal
    
    def split_signal(self, signal: np.ndarray, threshold_length_segment_max: float = 60.0, threshold_length_segment_min: float = 0.1):
        intervals = []

        for top_db in range(30, 5, -5):
            intervals = librosa.effects.split(
            signal, top_db=top_db, frame_length=4096, hop_length=1024)
            if len(intervals) != 0 and max((intervals[:, 1] - intervals[:, 0]) / self.sampling_rate) <= threshold_length_segment_max:
                break
            
        return np.array([i for i in intervals if threshold_length_segment_min < (i[1] - i[0]) / self.sampling_rate <= threshold_length_segment_max])

    def clean_text(self, sentence: str) -> str:
        sentence = re.sub(self.puncs, "", sentence)
        sentence = re.sub(r"\s\s+", " ", sentence)
        sentence = sentence.strip().lower()

        return sentence
    
    def __call__(self, signals: List[torch.Tensor], get_signals: bool = False) -> torch.Tensor:
        lengths = torch.tensor([len(signal) for signal in signals])
        max_len = torch.max(lengths)
            
        padded_signals = []

        mel_lengths = []

        for index, signal in enumerate(signals):
            padded_signals.append(F.pad(signal, (0, max_len - lengths[index]), mode='constant', value=0.0))
            if not get_signals:
                mel_lengths.append(lengths[index] // self.hop_length + 1)

        if get_signals:
            return torch.stack(padded_signals)

        mels = self.mel_spectrogram(torch.stack(padded_signals)).type(torch.FloatTensor)
        mel_lengths = torch.stack(mel_lengths)

        return mels, mel_lengths
    
    def tokenize(self, graphemes: List[List[str]], max_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = []
        lengths = []
        for item in graphemes:
            if item != ['']:
                token = torch.tensor(np.array(self.dictionary(item)))
            else:
                token = torch.tensor(np.array([]))
            lengths.append(len(token))
            tokens.append(token)

        if max_len is None:
            max_len = np.max(lengths)

        padded_tokens = []
    
        for index, token in enumerate(tokens):
            padded_tokens.append(F.pad(token, (0, max_len - lengths[index]), mode='constant', value=self.pad_token))

        return torch.stack(padded_tokens), torch.tensor(lengths)