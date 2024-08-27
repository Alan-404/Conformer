import os
import numpy as np
from pydub import AudioSegment
import librosa
from typing import Union, Optional, List, Tuple
import pickle
import torch
import torch.nn.functional as F
import librosa
import json
from scipy.io import wavfile
import re
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
        if tokenizer_path is not None:
            with open(tokenizer_path, 'r', encoding='utf8') as file:
                patterns = json.load(file)

            self.slide_patterns = self.sort_pattern(
                patterns['single_vowel'] + patterns['composed_vowel'] + patterns['single_consonant'] + patterns['no_split']
            )
            self.dictionary = patterns['dictionary']

            self.single_vowels = patterns['single_vowel']
            self.composed_vowels = patterns['composed_vowel']

            self.voiced_special = patterns['voiced_special']
            self.voiceless_special = patterns['voiceless_special']

            self.single_consonants = patterns['single_consonant']
            self.no_split = patterns['no_split']
            self.voiced = patterns['voiced']
            self.voiceless = patterns['voiceless']
                
            self.single_suffixes = patterns['single_suffix']
            self.composed_suffixes = patterns['composed_suffix']
            self.no_split_suffixes = patterns['no_split_suffix']

            self.short_items = patterns['short_item']

            self.grammar = patterns['grammar']

            self.replace_dict = patterns['replace']
            self.reverse_replace = {v: k for k, v in self.replace_dict.items()}

            self.delim_token = delim_token
            self.unk_token = unk_token
            self.pad_token = pad_token

            self.vocab = [pad_token] + patterns['single_vowel'] + patterns['composed_vowel'] + patterns['single_consonant'] + patterns['no_split'] + patterns['voiced'] + patterns['voiceless'] + patterns['voiced_special'] + patterns['voiceless_special'] + patterns['exceptions'] + patterns['short_item'] + patterns['no_split_suffix'] + [delim_token, unk_token]

            self.pad_id = self.find_token_id(pad_token)
            self.unk_id = self.find_token_id(unk_token)
            self.delim_id = self.find_token_id(delim_token)

            self.special_cases = list(self.replace_dict.keys())

            self.puncs = puncs

        self.device = device

    # Audio Functions 
    def read_audio(self, path: str) -> torch.Tensor:
        signal, sr = librosa.load(path, sr=None)

        if sr != self.sample_rate:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=self.sample_rate)
        
        return torch.tensor(signal, dtype=torch.float).to(self.device)
    
    def split_segment(self, signal: torch.Tensor, start: float, end: float):
        return signal[int(start * self.sample_rate) : int(end * self.sample_rate)]
    
    def mel_spectrogram(self, signal: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        mel = self.__mel_spectrogram(signal)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel
    
    # Text Functions
    def sort_pattern(self, patterns: List[str]) -> List[str]:
        patterns = sorted(patterns, key=len)
        patterns.reverse()

        return patterns
    
    def word2graphemes(self, text: str, n_grams: int = 3, reverse: bool = False) -> List[str]:
        return self.slide_graphemes(text, self.slide_patterns, reverse=reverse, n_grams=n_grams)
    
    def graphemes2tokens(self, graphemes: List[str]) -> torch.Tensor:
        tokens = []
        for grapheme in graphemes:
            tokens.append(self.find_token_id(grapheme))
        return torch.tensor(tokens).to(self.device)
    
    def sentence2tokens(self, sentence: str) -> torch.Tensor:
        graphemes = self.sentence2graphemes(sentence)
        tokens = self.graphemes2tokens(graphemes)
        return tokens.to(self.device)
    
    def clean_text(self, sentence: str) -> str:
        sentence = re.sub(self.puncs, " ", sentence)
        sentence = re.sub(r"\s\s+", " ", sentence)
        sentence = sentence.strip()
        return sentence
    
    def sentence2graphemes(self, sentence: str) -> List[str]:
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
    
    def spec_decode_sentence(self, sentence: str) -> str:
        words = sentence.split(" ")
        correct_words = []
        for word in words:
            correct_words.append(self.spec_decode(word))
        return correct_words
    
    def spec_decode(self, word: str) -> str:
        for index, value in enumerate(list(self.replace_dict.values())):
            if value not in word:
                return word
            arr = word.split(value)
            if len(arr) == 2:
                return word.replace(value, self.special_cases[index])
        return word
    
    def slide_graphemes(self, text: str, patterns: List[str], n_grams: int = 4, reverse: bool = False) -> List[str]:
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
    def as_target(self, list_graphemes: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        list_tokens = []
        max_length = 0
        lengths = []

        for graphemes in list_graphemes:
            length = len(graphemes)
            lengths.append(length)

            if length > max_length:
                max_length = length

            list_tokens.append(self.graphemes2tokens(graphemes))

        padded_tokens = []
        for index, tokens in enumerate(list_tokens):
            padded_tokens.append(
                F.pad(tokens, pad=(0, max_length - lengths[index]), value=self.pad_id)
            )
        
        padded_tokens = torch.stack(padded_tokens)
        lengths = torch.tensor(lengths).to(self.device)

        return padded_tokens, lengths

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
        lengths = (torch.tensor(lengths).to(self.device) // self.hop_length) + 1

        return mels, lengths