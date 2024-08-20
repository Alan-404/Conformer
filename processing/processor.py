import os
import numpy as np
from pydub import AudioSegment
import librosa
from typing import Union, Optional, List, Tuple, Dict
import pickle
import torch
import torch.nn.functional as F
import librosa
import json
from scipy.io import wavfile
from torchaudio.transforms import MelSpectrogram

MAX_AUDIO_VALUE = 32768.0

class ConformerProcessor:
    def __init__(self, 
                 # Audio Config
                 sample_rate: int = 16000, 
                 n_fft: int = 400, 
                 win_length: int = 400, 
                 hop_length: int = 160, 
                 n_mels: int = 80, 
                 fmin: float = 0.0, 
                 fmax: float = 8000.0,
                 norm: Optional[str] = "slaney",
                 mel_scale: str = 'slaney',
                 # Text Config
                 tokenizer_path: Optional[str] = None,
                 pad_token: str = "<PAD>",
                 delim_token: str = "|",
                 unk_token: str = "<UNK>",
                 puncs: str = r"([:./,?!@#$%^&=`~;*\(\)\[\]\"\\])",
                 # Device Config
                 device: Union[str, int] = 'cpu') -> None:
        # Audio Setup
        assert mel_scale in ['htk', 'slaney'], "Invalid Mel Scale, Only HTK or Slaney"
        if norm is not None:
            assert norm == 'slaney', "Invalid Norm, we only support Slaney Norm"
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.__mel_spectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=fmin,
            f_max=fmax,
            n_mels=n_mels,
            norm=norm,
            mel_scale=mel_scale
        ).to(device)

        # Text Setup
        with open(tokenizer_path, 'r', encoding='utf8') as file:
            patterns = json.load(file)

        self.slide_patterns = self.sort_pattern(
            patterns['single_vowel'] + patterns['composed_vowel'] + patterns['single_consonant'] + patterns['no_split']
        )
        self.vocab = [pad_token] + patterns['single_vowel'] + patterns['composed_vowel'] + patterns['single_consonant'] + patterns['no_split'] + [delim_token, unk_token]

        self.pad_token = pad_token
        self.delim_token = delim_token
        self.unk_token = unk_token

        self.pad_id = self.find_token_id(pad_token)
        self.delim_id = self.find_token_id(delim_token)
        self.unk_id = self.find_token_id(unk_token)

        self.puncs = puncs
        self.replace_dict = patterns['replace']

        self.device = device

    # Audio Functions 
    def read_audio(self, path: str) -> torch.Tensor:
        sr, signal = wavfile.read(path)
        signal = signal / MAX_AUDIO_VALUE

        if sr != self.sample_rate:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=self.sample_rate)
        
        return torch.tensor(signal, dtype=torch.float).to(self.device)

    def read_pickle(self, path: str) -> np.ndarray:
        with open(path, 'rb') as file:
            signal = pickle.load(file)

        signal = librosa.resample(y=signal, orig_sr=8000, target_sr=self.sample_rate)

        return signal
    
    def read_pcm(self, path: str) -> np.ndarray:
        audio = AudioSegment.from_file(path, frame_rate=8000, channels=1, sample_width=2).set_frame_rate(self.sample_rate).get_array_of_samples()
        return np.array(audio).astype(np.float64) / MAX_AUDIO_VALUE
    
    def read_signal(self, path: str, role: Optional[int] = None) -> np.ndarray:
        if role is not None:
            signal, _ = librosa.load(path, sr=self.sample_rate, mono=False)
            signal = signal[role]
        else:
            signal, _ = librosa.load(path, sr=self.sample_rate, mono=True)

        return signal
    
    def split_segment(self, signal: torch.Tensor, start: float, end: float):
        return signal[int(start * self.sample_rate) : int(end * self.sample_rate)]

    def load_audio(self, path: str, start: Optional[float] = None, end: Optional[float] = None, role: Optional[int] = None) -> torch.Tensor:
        if ".pickle" in path:
            signal = self.read_pickle(path)
        elif ".pcm" in path:
            signal = self.read_pcm(path)
        else:
            signal = self.read_signal(path, role)

        if start is not None and end is not None:
            signal = self.split_segment(signal, start, end)

        signal = torch.tensor(signal, dtype=torch.float)

        return signal
    
    def mel_spectrogram(self, signal: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        is_numpy = False
        if isinstance(signal, np.ndarray):
            is_numpy = True
            signal = torch.tensor(signal, dtype=torch.float)
        
        mel = self.__mel_spectrogram(signal)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        if is_numpy:
            mel = mel.numpy()
        
        return mel
    
    # Text Functions
    def sort_pattern(self, patterns: List[str]):
        patterns = sorted(patterns, key=len)
        patterns.reverse()

        return patterns
    
    def word2graphemes(self, text: str, n_grams: int = 3, reverse: bool = False) -> List[str]:
        return self.slide_graphemes(text, self.slide_patterns, reverse=reverse, n_grams=n_grams)
    
    def sentence2graphemes(self, sentence: str):
        sentence = self.clean_text(sentence.upper())
        words = sentence.split(" ")
        graphemes = []

        length = len(words)

        for index, word in enumerate(words):
            graphemes += self.word2graphemes(self.spec_replace(word))
            if index != length - 1:
                graphemes.append(self.delim_token)

        return graphemes
    
    def spec_replace(self, word: str) -> str:
        for key in self.replace_dict:
            arr = word.split(key)
            if len(arr) == 2:
                if arr[1] in self.single_vowels:
                    return word
                else:
                    return word.replace(key, self.replace_dict[key])
        return word
    
    def slide_graphemes(self, text: str, patterns: List[str], n_grams: int = 4, reverse: bool = False):
        if len(text) == 1:
            if text in patterns:
                return [text]
            return [self.unk_token]
        if reverse:
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

            if reverse:
                item = [item[i] for i in range(len(item) - 1, -1, -1)]
                item = "".join(item)
                
            if item in patterns:
                graphemes.append(item)
            elif num_steps == 1:
                graphemes.append(self.unk_token)
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

        if reverse:
            graphemes = [graphemes[i] for i in range(len(graphemes) - 1, -1, -1)]

        return graphemes
    
    def find_token_id(self, token: str) -> int:
        if token in self.vocab:
            return self.vocab.index(token)
        return self.vocab.index(self.unk_token)


    # Call Functions
    def __call__(self, audios: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        padded_audios = []
        lengths = []
        max_length = 0

        for audio in audios:
            length = len(audio)
            if length > max_length:
                max_length = length
            lengths.append(length)

        for index, audio in enumerate(audios):
            padded_audios.append(
                F.pad(audio, pad=(0, max_length - lengths[index]), value=0.0)
            )

        mels = self.mel_spectrogram(torch.stack(padded_audios))
        lengths = torch.tensor(lengths) // self.hop_length + 1

        return mels, lengths