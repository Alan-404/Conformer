import os
import numpy as np
import json
from pydub import AudioSegment
import librosa
from typing import Union, Optional, List, Tuple
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

        # Text
        if vocab_path is not None:
            assert os.path.exists(vocab_path)

            self.pattern = json.load(open(vocab_path, 'r'))
            
            self.stride_patterns = self.pattern['vowel'] + self.pattern['consonant'] + self.pattern['composed_vowel'] + self.pattern['composed_consonant'] + self.pattern['mixed']
            
            self.first_patterns = self.pattern['vowel'] + self.pattern['consonant'] + self.pattern['composed_vowel'] + self.pattern['composed_consonant']

            self.suffix_patterns  = dict(sorted(self.pattern['suffix'].items(), key=lambda item: len(item[0]), reverse=True))
            self.prefix_patterns = sorted(self.pattern['prefix'], key=lambda item: len(item[0]), reverse=True)

            self.dictionary = Vocab(
                create_vocab(self.create_dictionary(self.first_patterns + self.pattern['except']), specials=[pad_token])
            )

            self.dictionary.insert_token(word_delim_token, index=len(self.dictionary))
            self.dictionary.insert_token(unk_token, index=len(self.dictionary))

            self.unk_token = unk_token
            self.pad_token = pad_token
            self.word_delim_token = word_delim_token
        
            self.first_patterns = sorted(self.first_patterns, key=len, reverse=True)

            self.puncs = puncs

    def create_dictionary(self, vocab: List[str]):
        items = dict()
        for index, item in enumerate(vocab):
            items[item] = index + 1
        
        return items
    
    def find_token(self, char: str) -> int:
        if char in self.dictionary:
            return self.dictionary.__getitem__(char)
        return self.unk_token
    
    def get_last_item(self, word: str, pattern: str):
        length_item = len(pattern)
        start_check = len(word) - length_item
        if word[start_check: ] == pattern:
            return word[:start_check ], pattern
        return None
    
    def get_first_item(self, word: str, pattern: str):
        length_item = len(pattern)
        if word[0: length_item] == pattern:
            return pattern, word[length_item: ]
        return None
    
    def get_by_prev(self, word: str, patterns: List[dict]):
        for key in patterns.keys():
            items = self.get_last_item(word, key)
            if items is not None:
                for pattern in patterns[key]:
                    specific_items = self.get_last_item(items[0], pattern)
                    if specific_items is not None:
                        return items[0], [*items[1]]
                return items[0], [items[1]]
        return None
    
    def lookup(self, word: str, patterns: List[dict]):
        for pattern in patterns:
            items = self.get_first_item(word, pattern)
            if items is not None:
                return patterns[pattern], items[1]
        return [], word

    def get_prefix(self, word: str, patterns: List[dict]):
        for key in patterns:
            items = self.get_first_item(word, key)
            if items is not None:
                return [*items[0]], items[1]
        return [], word
    
    def get_suffix(self, word: str, patterns: List[dict]):
        for key in patterns.keys():
            items = self.get_last_item(word, key)
            if items is not None:
                return items[0], patterns[items[1]]
        return None
    
    def get_last(self, word: str, suffix_patterns: List[dict], past_patterns: List[dict], quantity_patterns: List[dict]):
        suffixes = []
        count = 0

        last_items = None
        last_items = self.get_by_prev(word, quantity_patterns)
        if last_items is not None:
            word = last_items[0]
            suffixes.append(last_items[1])
        
        while True:
            last_items = None
            last_items = self.get_suffix(word, suffix_patterns)
            if last_items is not None:
                word = last_items[0]
                suffixes.append(last_items[1])
                count = 0
            else:
                count += 1

            last_items = self.get_by_prev(word, past_patterns)
            if last_items is not None:
                word = last_items[0]
                suffixes.append(last_items[1])
                count = 0
            else:
                count += 1

            if count > 2:
                break

        suffixes = [suffixes[i] for i in range(len(suffixes) - 1, -1, -1)]

        return word, suffixes
    
    def stride_graphemes(self, text: str, patterns: List, n_grams: int = 4):
        if len(text) == 1:
            if text in patterns:
                return [text]
            return [self.unk_token]
        graphemes = []
        start = 0
        if len(text) - 1 < n_grams:
            n_grams = len(text)
        num_steps = n_grams
        while start < len(text):
            found = True
            item = text[start:start + num_steps]
                
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

        return graphemes
    
    def split_first(self, word: str, patterns: List):
        for pattern in patterns:
            items = self.get_first_item(word, pattern)
            if items is not None:
                return items[0], items[1]
        return '', word
    
    def word2grapheme(self, word: str):
        prefix = []
        suffixes = []

        first_item = ''
        stride_items = []
        looked_item = []

        word, suffixes = self.get_last(word, self.pattern['suffix'], self.pattern['past'], self.pattern['many'])
        if word != '':
            prefix, word = self.get_prefix(word, self.pattern['prefix'])
        
            if word != '':
                looked_item, word = self.lookup(word, self.pattern['dictionary'])

                if word != '':
                    first_item, word = self.split_first(word, self.first_patterns)

                    if word != '':
                        stride_items = self.stride_graphemes(word, self.stride_patterns)

        graphemes = prefix + looked_item + [first_item] + stride_items

        for suffix in suffixes:
            graphemes += suffix
        
        results = []
        for item in graphemes:
            if item != '':
                results.append(item)

        return results
    
    def text2graphemes(self, sentence: str):
        sentence = self.clean_text(sentence)
        words = sentence.split(" ")

        length = len(words) - 1

        graphemes = []
        for index, word in enumerate(words):
            graphemes += self.word2grapheme(word)
            if index != length:
                graphemes.append(self.word_delim_token)

        return graphemes

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

    def load_vocab(self, path: str) -> List[str]:
        if os.path.exists(path):
            return json.load(open(path, encoding='utf-8'))

    def clean_text(self, sentence: str) -> str:
        sentence = re.sub(self.puncs, "", sentence)
        sentence = re.sub(r"\s\s+", " ", sentence)
        sentence = sentence.strip().lower()

        return sentence
    
    def find_specs(self, word: str):
        for index, item in enumerate(list(self.replace_dict.values())):
            if item in word:
                return (list(self.replace_dict.keys())[index], item)
        return None
    
    def post_process(self, text: str):
        words = text.split(" ")
        items = []
        for word in words:
            patterns = self.find_specs(word)
            if patterns is None or word.split(patterns[1])[1] == '':
                items.append(word)
            else:
                items.append(word.replace(patterns[1], patterns[0]))
        return " ".join(items)
    
    def decode_beam_search(self, digits: np.ndarray, beam_width: int = 170, beam_prune_logp: float = -20.0):
        text = self.ctc_lm.decode(
                    digits,
                    beam_width=beam_width,
                    beam_prune_logp=beam_prune_logp,
                    hotword_weight=self.hotwords_dict['weight'],
                    hotwords=self.hotwords_dict['items']
                )
        
        return self.post_process(text)

    def decode_batch(self, digits: Union[torch.Tensor, np.ndarray, list], group_token: bool = True) -> List[str]:
        sentences = []
        for logit in digits:
            if group_token:
                logit = self.group_tokens(logit)
            sentences.append(self.token2text(logit))
        return sentences
    
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

    def token2text(self, tokens: np.ndarray) -> str:
        text = ""
        for token in tokens:
            if token == self.word_delim_token:
                text += " "
            elif token >= 0:
                text += self.dictionary.lookup_token(token)
            else:
                break
        for item in self.special_tokens:
            text = text.replace(item, "")
        text = re.sub(r"\s\s+", " ", text)
        return text.strip()
    
    def spec_replace(self, word: str):
        for key in self.replace_dict:
            word = word.replace(key, self.replace_dict[key])
        return word
    
    def word2graphemes(self, text: str, patterns: List[str], n_grams: int = 4):
        if len(text) == 1:
            if text in patterns:
                return [text]
            return ["<unk>"]
        graphemes = []
        start = 0
        if len(text) - 1 < n_grams:
            n_grams = len(text)
        num_steps = n_grams
        while start < len(text):
            found = True
            item = text[start:start + num_steps]
            
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

        return graphemes
    
    def sentence2graphemes(self, sentence: str):
        sentence = self.clean_text(sentence)
        words = sentence.split(' ')
        graphemes = []
        for index, word in enumerate(words):
            graphemes += self.text2graphemes(word)
            if index != len(words) - 1:
                graphemes.append(self.word_delim_item)
        return graphemes
    
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