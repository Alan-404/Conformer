from pyctcdecode import build_ctcdecoder
from typing import List, Union, Iterable, Optional, Callable
import numpy as np
import re

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
        return self.post_process_s2t(
            self.decoder.decode(
                logits,
                beam_width=self.beam_width,
                beam_prune_logp=self.beam_prune_logp,
                hotwords=self.hotwords,
                hotword_weight=self.hotword_weight
            )
        )
    
    def post_process_s2t(self, raw: str) -> str:
        if not raw or len(raw)==1:
            return ''
        text = raw.replace('ti vi', 'tivi').replace('goai phoai', 'wifi')
        text = text.replace('côm bô', 'combo').replace("on lai", "online").replace("ốp lai", "offline")
        text = text.replace('ca cộng', 'k cộng').replace("gia lô", "zalo")
        text = text.replace('ốp thiết bị', 'off thiết bị')
        text = text.replace('cờ lao', 'cloud').replace("nhận mát", "nhận mac").replace("nhận mắc", "nhận mac")
        text = text.replace('in tơ net','internet').replace('in tơ nét','internet')
        text = text.replace('ca me ra', 'camera').replace('ca mê ra', 'camera').replace("cam mê ra", "camera").replace("cam me ra", "camera")
        text = text.replace("a bi quan", "ip1").replace("i bi quan", "ip1").replace("i bi a", "ip1").replace("con vật to", "converter")
        text = text.replace("vốt chơ", "voucher ").replace("lô gô", "logo")
        text = text.replace("lên áp", "lên app").replace("cái áp", "cái app").replace("meo", "mail").replace("thu lại bọt", "thu lại port").replace("thu bọt", "thu port")
        text = text.replace("hai fpt", "hi fpt").replace("tàu lâu", "tào lao").replace("gửi mai", "gửi mail")
        text = ' '.join([word for word in text.split(' ') if word not in ['n','d','h','g']])
        text = re.sub("\s\s+", " ", text)
        return text.strip()

    def decode_batch(self, logits: np.ndarray, lengths: Optional[np.ndarray] = None, decode_func: Optional[Callable[[str], str]] = None) -> List[str]:
        preds = []
        for index, probs in enumerate(logits):
            text = self.decode(probs[: lengths[index], :] if lengths is not None else probs)
            if decode_func is not None:
                text = decode_func(text)
            preds.append(text)
        return preds