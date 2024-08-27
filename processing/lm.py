import torch
from pyctcdecode import build_ctcdecoder
from typing import List, Union, Iterable, Optional
import numpy as np

# from torchaudio.models.decoder._ctc_decoder import ctc_decoder
# from typing import List

# class KenLanguageModel:
#     def __init__(self, lm_path: str, lexicon_path: str, tokens: List[str], pad_token: str = "<PAD>", delim_token: str = "|", unk_token: str = "<UNK>") -> None:
#         self.decoder = ctc_decoder(
#             lexicon=lexicon_path,
#             tokens=tokens,
#             lm=lm_path,
#             sil_token=delim_token,
#             blank_token=pad_token,
#             unk_word=unk_token
#         )

#     def batch_decode_beam_search(self, logits: torch.Tensor, lengths: torch.Tensor):
#         decoder_outputs = self.decoder(logits, lengths)
#         texts = []
#         for item in decoder_outputs:
#             texts.append(" ".join(item[0].words))
#         return texts


class KenLanguageModel:
    def __init__(self, 
                 lm_path: str, 
                 vocab: List[str], 
                 alpha: float = 2.1, 
                 beta: float = 9.2,
                 beam_width: int = 190,
                 beam_prune_logp: float = -20,
                 hotwords: Union[List[str], Iterable[str]] =  ['fpt', 'wifi', 'modem', 'fpt telecom', 'internet',   'free',  'lag', 'tổng đài viên', 'reset', 'check', 'test', 'code', 'port', 'net', 'email', 'mail', 'box'],
                 hotword_weight: Optional[float] = 9.0,
                ) -> None:
        self.decoder = build_ctcdecoder(
            labels=vocab,
            kenlm_model_path=lm_path,
            alpha=alpha,
            beta=beta
        )

        self.beam_width = beam_width
        self.beam_prune_logp = beam_prune_logp
        self.hotwords = hotwords
        self.hotword_weight = hotword_weight

    def decode(self, logits: np.ndarray) -> str:
        return self.decoder.decode(
            logits,
            beam_width=self.beam_width,
            beam_prune_logp=self.beam_prune_logp,
            hotwords=self.hotwords,
            hotword_weight=self.hotword_weight
        )

    def decode_batch(self, logits: np.ndarray, lengths: Optional[np.ndarray] = None) -> List[str]:
        preds = []
        for index, probs in enumerate(logits):
            preds.append(
                self.decode(probs[: lengths[index], :] if lengths is not None else probs)
            )
        return preds