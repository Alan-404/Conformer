import torch

from torchaudio.models.decoder._ctc_decoder import ctc_decoder
from typing import List

class KenLanguageModel:
    def __init__(self, lm_path: str, lexicon_path: str, tokens: List[str], pad_token: str = "<PAD>", delim_token: str = "|", unk_token: str = "<UNK>") -> None:
        self.decoder = ctc_decoder(
            lexicon=lexicon_path,
            tokens=tokens,
            lm=lm_path,
            sil_token=delim_token,
            blank_token=pad_token,
            unk_word=unk_token
        )

    def batch_decode_beam_search(self, logits: torch.Tensor, lengths: torch.Tensor):
        decoder_outputs = self.decoder(logits, lengths)
        texts = []
        for item in decoder_outputs:
            texts.append(" ".join(item[0].words))
        return texts