import os
import numpy as np
import json
from pydub import AudioSegment
import librosa
from typing import Union, Optional, List, Tuple, Dict
import re
import pickle
import torch
import torch.nn.functional as F
import librosa

from scipy.io import wavfile

from torchaudio.models.decoder._ctc_decoder import ctc_decoder
from torchaudio.transforms import MelSpectrogram

MAX_AUDIO_VALUE = 32768.0

class ConformerProcessor:
    def __init__(self, 
                 sample_rate: int = 16000, 
                 n_fft: int = 400, 
                 win_length: int = 400, 
                 hop_length: int = 160, 
                 n_mels: int = 80, 
                 fmin: float = 0.0, 
                 fmax: float = 8000.0,
                 norm: Optional[str] = "slaney",
                 mel_scale: str = 'htk',
                 device: str = 'cpu') -> None:
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

        self.device = device

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
        if is_numpy:
            mel = mel.numpy()
        
        return mel
    
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

class __ConformerProcessor:
    def __init__(self, 
                 sample_rate: int = 16000,
                 tokenizer_path: Optional[str] = None, pad_token: str = "<PAD>", delim_token: str = "|", unk_token: str = "<UNK>", puncs: str = r"([:./,?!@#$%^&=`~;*\(\)\[\]\"\\])",
                 lm_path: Optional[str] = None) -> None:
        
        self.sample_rate = sample_rate

        self.puncs = puncs
        
        if tokenizer_path is not None:
            patterns = json.load(open(tokenizer_path, 'r', encoding='utf8'))

            self.slide_patterns = self.sort_pattern(
                patterns['single_vowel'] + patterns['composed_vowel'] + patterns['single_consonant'] + patterns['no_split'] + patterns['voiced'] + patterns['voiceless'] + patterns['voiced_special'] + patterns['voiceless_special'] + patterns['short_item'] + patterns['single_suffix'] + patterns['composed_suffix'] + patterns['no_split_suffix'] + list(patterns['dictionary'].keys())
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
            
            self.decoder = None
            if lm_path is not None:
                self.decoder = ctc_decoder(
                    lexicon='./lm/lexicon.txt',
                    tokens=self.tokenizer.get_itos(),
                    lm=lm_path,
                    sil_token=delim_token,
                    blank_token=pad_token,
                    unk_word=unk_token
                )

    def sort_pattern(self, patterns: List[str]):
        patterns = sorted(patterns, key=len)
        patterns.reverse()

        return patterns

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
    
    def split_signal(self, signal: np.ndarray, threshold_length_segment_max: float = 60.0, threshold_length_segment_min: float = 0.1):
        intervals = []

        for top_db in range(30, 5, -5):
            intervals = librosa.effects.split(signal, top_db=top_db, frame_length=4096, hop_length=1024)
            if len(intervals) != 0 and max((intervals[:, 1] - intervals[:, 0]) / self.sample_rate) <= threshold_length_segment_max:
                break
            
        return np.array([i for i in intervals if threshold_length_segment_min < (i[1] - i[0]) / self.sample_rate <= threshold_length_segment_max])

    def clean_text(self, sentence: str) -> str:
        sentence = str(sentence)
        sentence = re.sub(self.puncs, "", sentence)
        sentence = re.sub(r"\s\s+", " ", sentence)
        sentence = sentence.strip()
        return sentence
    
    def slide_graphemes(self, text: str, patterns: List[str], n_grams: int = 4, reverse: bool = True):
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
    
    def post_handle(self, graphemes):
        items = []
        last_idx = len(graphemes) - 1
        for index, item in enumerate(graphemes):
            if item in self.single_suffixes:
                split_items = self.split_handle(item)
                # items += split_items
                if index == 0:
                    items += split_items
                else:
                    concat = f"{items[-1]}{split_items[0]}"
                    if concat in self.composed_vowels:
                        items[-1] = concat
                        items.append(split_items[1])
                    else:
                        items += split_items
                
            elif item in self.composed_suffixes:
                items += self.split_handle(item)

            elif item in self.dictionary:
                items += self.dictionary[item]

            elif item in self.no_split_suffixes:
                if index == last_idx or graphemes[index + 1] in self.short_items:
                    items.append(item)
                else:
                    items += [*item]

            elif index == 0:
                if item in self.voiced_special or item in self.voiceless_special or item in self.voiceless:
                    items.append(item[:-1])
                    items.append(item[-1])
                else:
                    items.append(item)
            elif item in self.grammar:
                if index < 2:
                    items += [*item]
                elif items[-1] == item[0]:
                    items[-1] = f"{item[0]}{item[0]}"
                    items.append(item[1])
                elif items[-1] in self.grammar[item] or (index != last_idx and graphemes[index + 1][0] in self.single_vowels):
                    items += [*item]
                else:
                    items.append(item)

            elif index == last_idx:
                if item in self.voiceless:
                    items += [*item]
                else:
                    items.append(item)

            elif item in self.voiced and (graphemes[index + 1][0] in self.single_vowels or graphemes[index + 1] in self.no_split_suffixes):
                items += [*item]
            elif item in self.voiceless and (graphemes[index + 1] not in self.single_suffixes and graphemes[index + 1] not in self.no_split_suffixes):
                items += [*item]
            elif item in self.voiceless_special:
                if graphemes[index + 1] in self.single_suffixes or graphemes[index + 1] in self.composed_suffixes or graphemes[index + 1] in self.no_split_suffixes:
                    items.append(item)
                else:
                    items.append(item[:-1])
                    items.append(item[-1])
            elif item in self.voiced_special:
                if graphemes[index + 1] in self.single_suffixes or graphemes[index + 1] in self.no_split_suffixes:
                    items.append(item)
                else:
                    items.append(item[:-1])
                    items.append(item[-1])
            else:
                items.append(item)

        graphemes = []
        for index, item in enumerate(items):
            if index == 0:
                graphemes.append(item)
            else:
                if item[0] == graphemes[-1]:
                    concat = f"{item[0]}{item[0]}"
                    if concat in self.vocab:
                        graphemes[-1] = concat
                        if len(item) > 1:
                            graphemes.append(item[1:])
                    else:
                        graphemes.append(item)
                else:
                    graphemes.append(item)

        return graphemes
    
    def split_handle(self, item: str):
        chars = [*item]
        graphemes = []
        is_vowel = False
        for index, item in enumerate(chars):
            if index == 0:
                if item in self.single_vowels:
                    is_vowel = True
                else:
                    is_vowel = False
                graphemes.append(item)
            else:
                if item in self.single_vowels:
                    if is_vowel:
                        concat = f"{graphemes[-1]}{item}"
                        if concat in self.slide_patterns:
                            graphemes[-1] = concat
                            continue
                    graphemes.append(item)
                else:
                    if not is_vowel:
                        concat = f"{graphemes[-1]}{item}"
                        if concat in self.slide_patterns:
                            graphemes[-1] = concat
                            continue
                    graphemes.append(item)

        return graphemes
    
    def grammar_handle(self, item: str, prev: str):
        if item[0] == prev:
            concat = f"{prev}{prev}"
            if concat in self.vocab:
                if len(item) > 1:
                    return [concat, item[1:]]
                else:
                    return [concat]
        for key in self.grammar:
            if key == item:
                for pattern in self.grammar[key]:
                    if prev == pattern:
                        return [prev] + [*item]
                return [prev, item]
        return None
    
    def word2graphemes(self, text: str, check: bool = True):
        extracted_graphemes = self.slide_graphemes(text, patterns=self.slide_patterns, reverse=True)
        
        return extracted_graphemes
    
    def sentence2graphemes(self, sentence: str):
        # sentence = self.spec_replace(self.clean_text(sentence.upper()))
        sentence = self.clean_text(sentence.upper())
        words = sentence.split(" ")
        graphemes = []

        length = len(words)

        for index, word in enumerate(words):
            graphemes += self.word2graphemes(self.spec_replace(word))
            if index != length - 1:
                graphemes.append(self.delim_token)

        return graphemes
    
    def find_token_id(self, token: str):
        if token in self.vocab:
            return self.vocab.index(token)
        return self.vocab.index(self.unk_token)
    
    def token2text(self, tokens: np.ndarray, get_string: bool = False) -> str:
        words = []
        for token in tokens:
            words.append(self.dictionary.vocab[token])

        if get_string:
            return "".join(words).replace(self.delim_token, " ")
        
        return words
    
    def group_tokens(self, logits: Union[torch.Tensor, np.ndarray], length: Optional[int] = None) -> Union[np.ndarray, torch.Tensor]:
        items = []
        prev_item = None

        if length is None:
            length = length = len(logits)

        for i in range(length):
            if prev_item is None:
                items.append(logits[i])
                prev_item = logits[i]
                continue
            
            if logits[i] == self.pad_token:
                prev_item = None
                continue

            if logits[i] == prev_item:
                continue

            items.append(logits[i])
            prev_item = logits[i]
        return items
    
    def decode(self, tokens: Union[torch.Tensor, np.ndarray, List[int]], group_token: bool = True):
        if group_token:
            tokens = self.group_tokens(tokens)
        return "".join(self.token2text(tokens)).replace(self.delim_token, " ")
    
    def decode_batch(self, digits: Union[torch.Tensor, np.ndarray, list], group_token: bool = True) -> List[str]:
        sentences = []
        for logit in digits:
            sentence = self.decode(logit, group_token=group_token)
            sentences.append(sentence)
        return sentences
    
    def decode_spec(self, sentence: str):
        words = sentence.split(" ")
        items = []
        for word in words:
            found = False
            for item in self.reverse_replace:
                if item in word:
                    found = True
                    if word.split(item)[1] != '':
                        items.append(word.replace(item, self.reverse_replace[item]))
                    else:
                        items.append(word) 
                    break
            if found == False:
                items.append(word)
        return " ".join(items)
    
    def spec_replace(self, word: str):
        for key in self.replace_dict:
            arr = word.split(key)
            if len(arr) == 2:
                if arr[1] in self.single_vowels:
                    return word
                else:
                    return word.replace(key, self.replace_dict[key])
        return word
    
    def graphemes2tokens(self, graphemes: List[str]):
        tokens = []
        for grapheme in graphemes:
            tokens.append(self.find_token_id(grapheme))
        return tokens

    def as_target(self, texts: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = []
        lengths = []
        max_length = 0
        for graphemes in texts:
            length = len(graphemes)
            lengths.append(length)
            if max_length < length:
                max_length = length

            tokens.append(self.graphemes2tokens(graphemes))

        padded_tokens = []
        for index, token_list in enumerate(tokens):
            padded_tokens.append(F.pad(torch.tensor(token_list), (0, max_length - lengths[index]), mode='constant', value=self.pad_id))

        return torch.stack(padded_tokens), torch.tensor(lengths)
    
    def __call__(self, signals: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = []
        max_length = 0
        
        for signal in signals:
            length = len(signal)
            lengths.append(length)
            if max_length < length:
                max_length = length

        padded_signals = []
        for index, signal in enumerate(signals):
            padded_signals.append(F.pad(signal, (0, max_length - lengths[index]), mode='constant', value=0.0))

        return torch.stack(padded_signals), torch.tensor(lengths)
    
    def decode_beam_search(self, logits: torch.Tensor, lengths: torch.Tensor):
        assert self.decoder is not None
        decoder_outputs = self.decoder(logits, lengths)
        texts = []
        for item in decoder_outputs:
            texts.append(" ".join(item[0].words))
        return texts