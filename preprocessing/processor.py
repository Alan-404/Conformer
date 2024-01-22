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
    def __init__(self, vocab_path: str, unk_token: str = "<unk>", pad_token: str = "<pad>", word_delim_token: str = "|", sampling_rate: int = 16000, num_mels: int = 80, n_fft: int = 400, hop_length: int = 160, win_length: int = 400, fmin: float = 0.0, fmax: float = 8000.0, puncs: str = r"([:./,?!@#$%^&=`~*\(\)\[\]\"\-\\])", lm_path: Optional[str] = None, beam_alpha: float = 2.1, beam_beta: float = 9.2) -> None:
        # Text
        self.replace_dict = dict()
        self.dictionary = Vocab()
        self.hotwords_dict = dict()

        self.create_vocab(vocab_path, pad_token=pad_token, word_delim_token=word_delim_token, unk_token=unk_token)

        self.word_delim_item = word_delim_token
        self.unk_item = unk_token

        self.unk_token = self.find_token(unk_token)
        self.pad_token = self.find_token(pad_token)
        self.word_delim_token = self.find_token(word_delim_token)

        self.special_tokens = [unk_token, pad_token]

        self.puncs = puncs

        if lm_path is not None and os.path.exists(lm_path):
            self.ctc_lm = build_ctcdecoder(
                labels=self.dictionary.get_itos(),
                kenlm_model_path=lm_path,
                alpha=beam_alpha,
                beta=beam_beta
            )
            
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
        )

    def create_vocab(self, vocab_path: str, pad_token: str, word_delim_token: str, unk_token: str) -> Vocab:
        data = json.load(open(vocab_path, encoding='utf8'))

        assert "vocab" in data.keys() and "replace" in data.keys() and "hotword" in data.keys()

        vocabs = data['vocab']
        self.replace_dict = data['replace']
        self.hotwords_dict = data['hotword']
        
        dictionary = dict()
        count = 0
        for item in vocabs:
            count += 1
            dictionary[item] = count

        self.dictionary = Vocab(
            vocab=create_vocab(
                dictionary,
                specials=[pad_token]
            ))
    
        self.dictionary.insert_token(word_delim_token, index=len(self.dictionary))
        self.dictionary.insert_token(unk_token, index=len(self.dictionary))

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

    def load_audio(self, path: str, start: Optional[float] = None, end: Optional[float] = None, role: Optional[int] = None) -> torch.Tensor:
        if ".pickle" in path:
            signal = self.read_pickle(path)
        elif ".pcm" in path:
            signal = self.read_pcm(path)
        else:
            signal = self.read_audio(path, role)

        if start is not None and end is not None:
            signal = signal[int(start * self.sampling_rate) : int(end * self.sampling_rate)]

        signal = torch.FloatTensor(signal)

        return signal

    def load_vocab(self, path: str) -> List[str]:
        if os.path.exists(path):
            return json.load(open(path, encoding='utf-8'))
    
    def find_token(self, char: str) -> int:
        if char in self.dictionary:
            return self.dictionary.__getitem__(char)
        return self.unk_token

    def clean_text(self, sentence: str) -> str:
        sentence = re.sub(self.puncs, "", sentence)
        sentence = re.sub(r"\s\s+", " ", sentence)
        sentence = sentence.strip().lower()

        return sentence
    
    def text2token(self, sentence: str) -> torch.Tensor:
        sentence = self.clean_text(sentence)
        sentence = sentence.replace(" ", self.word_delim_item)
        tokens = []
        for char in [*sentence]:
            tokens.append(self.find_token(char))
        return torch.tensor(tokens)
    
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
    
    def word2graphemes(self, text: str,  n_grams: int = 3):
        if len(text) == 1:
            if text in self.dictionary:
                return [text]
            return [self.unk_item]
        graphemes = []
        start = 0
        if len(text) - 1 < n_grams:
            n_grams = len(text)
        num_steps = n_grams
        while start < len(text):
            found = True
            item = text[start:start + num_steps]

            if num_steps == 2:
                item = self.spec_replace(item)
            
            if item in self.dictionary:
                graphemes.append(item)
            elif num_steps == 1:
                graphemes.append(self.unk_item)
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
            graphemes += self.word2graphemes(word)
            if index != len(words) -1:
                graphemes.append("|")
        return graphemes
    
    def __call__(self, signals: List[torch.Tensor], max_len: Optional[int] = None, return_length: bool = False) -> torch.Tensor:
        if max_len is None:
            max_len = np.max([len(signal) for signal in signals])

        mels = []
        mel_lengths = []

        for signal in signals:
            signal_length = len(signal)
            padded_signal = F.pad(signal, (0, max_len - signal_length), mode='constant', value=0.0)
            mels.append(self.mel_spectrogram(padded_signal))
            mel_lengths.append((signal_length // self.hop_length) + 1)

        mels = torch.stack(mels).type(torch.FloatTensor)

        if return_length:
            return mels, torch.tensor(mel_lengths)
        
        return mels
    
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