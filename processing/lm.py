from pyctcdecode import build_ctcdecoder
from typing import List, Union, Iterable, Optional, Callable
import numpy as np

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

    def decode_batch(self, logits: np.ndarray, lengths: Optional[np.ndarray] = None, decode_func: Callable[[str], str] = None) -> List[str]:
        preds = []
        for index, probs in enumerate(logits):
            text = self.decode(probs[: lengths[index], :] if lengths is not None else probs)
            if decode_func is not None:
                text = decode_func(text)
            preds.append(text)
        return preds