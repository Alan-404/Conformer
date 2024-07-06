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
    def __init__(self, vocab_path: Optional[str] = None, unk_token: str = "<unk>", pad_token: str = "<pad>", word_delim_token: str = "|", sampling_rate: int = 16000, num_mels: int = 80, n_fft: int = 400, hop_length: int = 160, win_length: int = 400, fmin: float = 0.0, fmax: float = 8000.0, puncs: str = r"([:./,?!@#$%^&=`~*\(\)\[\]\"\-\\])", lm_path: Optional[str] = None, beam_alpha: float = 2.0, beam_beta: float = 1.0, device: str = 'cpu') -> None:
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
            self.patterns = json.load(open(vocab_path, 'r', encoding='utf8'))

            self.vowels = self.patterns['single_vowel'] + self.patterns['composed_vowel']
            self.single_vowels = self.patterns['single_vowel']
            self.consonants = self.patterns['single_consonant'] + self.patterns['composed_consonant']

            self.puncs = puncs

            self.unk_token = unk_token
            self.delim_token = word_delim_token
            self.pad_token = pad_token

            self.dictionary = Vocab(
                vocab=create_vocab(ordered_dict=self.create_dictionary(), specials=[pad_token], special_first=True)
            )

            self.dictionary.append_token(word_delim_token)
            self.dictionary.append_token(unk_token)

            self.pad_idx = self.find_idx(pad_token)
            self.delim_idx = self.find_idx(word_delim_token)
            self.unk_token = self.find_idx(unk_token)

            if lm_path is not None and os.path.exists(lm_path):
                self.ctc_lm = build_ctcdecoder(
                    labels=self.dictionary.get_itos(),
                    kenlm_model_path=lm_path,
                    alpha=beam_alpha,
                    beta=beam_beta
                )

                self.replace_dict = self.patterns['replace']
                self.hotwords_dict = self.patterns['hotword']

    def find_specs(self, word: str):
        for index, item in enumerate(list(self.patterns['replace'].values())):
            if item in word:
                return (list(self.patterns['replace'].keys())[index], item)
        return None
    
    def spec_replace(self, word: str):
        for key in self.patterns['replace']:
            word = word.replace(key, self.patterns['replace'][key])
        return word

    def find_idx(self, token: str) -> int:
        if token in self.dictionary.get_itos():
            return self.dictionary.get_itos().index(token)
        return self.dictionary.get_itos().index(self.unk_token)

    def create_dictionary(self) -> Dict[str, int]:
        items = self.patterns['single_vowel'] + self.patterns['composed_vowel'] + self.patterns['single_consonant'] + self.patterns['composed_consonant'] + self.patterns['exception']
        count = 1

        dictionary = dict()
        for item in items:
            if item not in dictionary:
                dictionary[item] = count
                count += 1

        return dictionary
    
    def sentence2graphemes(self, sentence: str):
        sentence = self.clean_text(sentence)

        words = sentence.split(" ")
        length = len(words)

        graphemes = []
        for index, word in enumerate(words):
            graphemes += self.word2graphemes(word)
            if index != length - 1:
                graphemes.append(self.delim_token)
        
        return graphemes
    
    def handle_exception(self, graphemes: List[str]):
        items = []
        for grapheme in graphemes:
            if grapheme not in self.dictionary.get_itos():
                items = items + [grapheme[:-1]] + [grapheme[-1]]
            else:
                items.append(grapheme)

        return items
    
    def word2graphemes(self, word: str):
        word = self.spec_replace(word)

        special_items = []
        if len(self.patterns['special']) != 0:
            special_items, word = self.lookup(word, self.patterns['special'])

        suffixes = []
        graphemes = []
        specials = ''
        splitted_items = []

        if word != '' and len(self.patterns['split_condition']) != 0:
            word, specials = self.split_by_condition(word, self.patterns['split_condition'])
        if word != '' and len(self.patterns['split']) != 0:
            word, splitted_items = self.split_handle(word, self.patterns['split'])
        if word != '' and len(self.patterns['suffix']) != 0:
            word, suffixes = self.get_suffixes(word, self.patterns['suffix'])
        if word != '':
            graphemes = self.stride_graphemes(word, self.vowels + self.consonants)

        for item in suffixes:
            graphemes += item
        graphemes += splitted_items

        if specials != '':
            graphemes += [specials]

        graphemes = special_items + graphemes

        if len(self.patterns['voiced']) != 0:
            graphemes = self.split_voiced_item(graphemes, self.patterns['voiced'], self.single_vowels)

        graphemes = self.concat_item(graphemes)

        if len(self.patterns['mixed_vowel']) != 0:
            graphemes = self.mixed_vowel_handle(graphemes, self.patterns['mixed_vowel'], self.vowels)

        graphemes = self.handle_exception(graphemes)

        return graphemes
    
    def text2tokens(self, sentence: str):
        graphemes = self.sentence2graphemes(sentence)
        return self.dictionary(graphemes)
    
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
    
    def decode_beam_search(self, digits: np.ndarray, beam_width: int = 128, beam_prune_logp: float = -10.0):
        text = self.ctc_lm.decode(
                    digits,
                    beam_width=beam_width,
                    beam_prune_logp=beam_prune_logp,
                    hotword_weight=self.hotwords_dict['weight'],
                    hotwords=self.hotwords_dict['items']
                )
        
        if len(self.replace_dict) == 0:
            return text
        
        return self.post_process(text)
    
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

    def token2text(self, tokens: np.ndarray, get_string: bool = False) -> str:
        words = []
        for token in tokens:
            words.append(self.dictionary.get_itos()[token])

        if get_string:
            return "".join(words).replace(self.delim_token, " ")
        
        return words

    def get_last_item(self, word: str, pattern: str):
        length_item = len(pattern)
        start_check = len(word) - length_item
        if word[start_check: ] == pattern:
            return word[:start_check ], pattern
        return None
    
    def lookup(self, word: str, patterns: Dict[str, List[str]]):
        for pattern in patterns:
            items = self.get_first_item(word, pattern)
            if items is not None:
                return patterns[pattern], items[1]
            
        return [], word
    
    def get_first_item(self, word: str, pattern: str):
        length_item = len(pattern)
        if word[0: length_item] == pattern:
            return pattern, word[length_item: ]
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
            return [self.unk_token]
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

        
        return [graphemes[i] for i in range(len(graphemes) - 1, -1, -1)]
        # return graphemes
    
    def split_handle(self, word: str, patterns: Dict[str, List[str]]):
        for item in patterns.keys():
            items = self.get_last_item(word, item)
            if items is not None:
                for pattern in patterns[item]:
                    if self.get_last_item(items[0], pattern) is not None:
                        return items[0], [*item]
                return word, []
        
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
    
    def split_voiced_item(self, graphemes: List[str], patterns: List[str], valid_items: List[str]):
        length = len(graphemes)
        if length <= 1:
            return graphemes
        
        items = []
        for i in range(length):
            if graphemes[i] in patterns:
                if i == 0 or i == length - 1:
                    items.append(graphemes[i])
                else:
                    if i+1 == length - 1 and graphemes[i+1] == "e":
                        items.append(graphemes[i])
                    elif graphemes[i+1][0] in valid_items:
                        items += [*graphemes[i]]
                    else:
                        items.append(graphemes[i])
            else:
                items.append(graphemes[i])

        return items
    
    def split_voiceless_item(self, graphemes: List[str], patterns: List[str], vowels: List[str]):
        length = len(graphemes)
        if length == 1:
            return [*graphemes[0]]
        
        items = []
        for i in range(length):
            if graphemes[i] in patterns:
                if i == 0 or i == length - 1:
                    items += [*graphemes[i]]
                else:
                    items.append(graphemes[i])
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
                        items.append(graphemes[i])
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
    
    def grammar_handle(self, word: str, patterns: List[Dict[str, List[str]]], vowels: List[str]):
        for key in patterns:
            items = self.get_last_item(word, key)
            if items is not None:
                if len(items[0]) == 0:
                    break
                if items[0][-1] in vowels:
                    break
                for pattern in patterns[key]:
                    if self.get_last_item(items[0], pattern):
                        return items[0], [*items[1]]
                return items[0], [items[1]]

        return word, []
    
    def get_suffixes(self, word: str, pattern: List[Dict[str, List[str]]]):
        suffixes = []

        word, grammar_suff = self.grammar_handle(word, self.patterns['split'], self.patterns['single_vowel'])
        suffixes.append(grammar_suff)

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
    
    def read_signal(self, path: str, role: Optional[int] = None) -> np.ndarray:
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
            signal = self.read_signal(path, role)

        if start is not None and end is not None:
            signal = self.split_segment(signal, start, end)

        signal = torch.FloatTensor(signal)
        signal = signal.to(self.device)
        return signal
    
    def split_signal(self, signal: np.ndarray, threshold_length_segment_max: float = 60.0, threshold_length_segment_min: float = 0.1):
        intervals = []

        for top_db in range(30, 5, -5):
            intervals = librosa.effects.split(signal, top_db=top_db, frame_length=4096, hop_length=1024)
            if len(intervals) != 0 and max((intervals[:, 1] - intervals[:, 0]) / self.sampling_rate) <= threshold_length_segment_max:
                break
            
        return np.array([i for i in intervals if threshold_length_segment_min < (i[1] - i[0]) / self.sampling_rate <= threshold_length_segment_max])

    def clean_text(self, sentence: str) -> str:
        sentence = str(sentence)
        sentence = re.sub(self.puncs, "", sentence)
        sentence = re.sub(r"\s\s+", " ", sentence)
        sentence = sentence.strip().lower()

        return sentence
    
    def split_signal(self, signal: np.ndarray, threshold_length_segment_max: float = 60.0, threshold_length_segment_min: float = 0.1):
        intervals = []

        for top_db in range(30, 5, -5):
            intervals = librosa.effects.split(
            signal, top_db=top_db, frame_length=2048, hop_length=512)
            if len(intervals) != 0 and max((intervals[:, 1] - intervals[:, 0]) / self.sampling_rate) <= threshold_length_segment_max:
                break
            
        return np.array([i for i in intervals if threshold_length_segment_min < (i[1] - i[0]) / self.sampling_rate <= threshold_length_segment_max])
    
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
    
    def tokenize(self, graphemes: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = []

        lengths = []

        max_length = 0

        for item in graphemes:
            length = len(item)
            lengths.append(length)

            if max_length < length:
                max_length = length

            tokens.append(torch.tensor(self.dictionary(item)))

        padded_tokens = []
        for index, item in enumerate(tokens):
            padded_tokens.append(F.pad(item, (0, max_length - lengths[index]), mode='constant', value=self.pad_idx))

        return torch.stack(padded_tokens), torch.tensor(lengths)