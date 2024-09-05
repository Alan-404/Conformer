import torch
from flashlight.lib.text.decoder import CriterionType, LexiconDecoder, LexiconDecoderOptions, Trie, KenLM, SmearingMode, DecodeResult
from flashlight.lib.text.dictionary import create_word_dict, Dictionary, load_words
from processing.processor import ConformerProcessor
import re
from typing import Optional, List, Dict, Union

class KenCTCDecoder:
    def __init__(self,
                 processor: ConformerProcessor,
                 lexicon_path: str, lm_path: str,
                 nbest: int = 1, beam_size: int = 50, beam_size_token: Optional[int] = None, beam_threshold: float = 50,
                 lm_weight: float = 2,
                 word_score: float = 0, unk_score: float = float('-inf'), sil_score: float = 0,
                 log_add: bool = False,
                 hotwords: Optional[Union[Dict[str, float], List[str]]] = None, hotword_score: Optional[float] = None) -> None:
        self.hotwords_dict = dict()
        if hotwords is not None:
            if isinstance(hotwords, list):
                assert hotword_score is not None and hotword_score > 0
                for hotword in hotwords:
                    self.hotwords_dict[hotword] = hotword_score
            else:
                for score in hotwords.values():
                    assert score > 0
                self.hotwords_dict = hotwords
        
        self.vocab_size = len(processor.vocab)
        tokens_dict = Dictionary(processor.vocab)
        lexicon = load_words(lexicon_path)

        if beam_size_token is None:
            beam_size_token = tokens_dict.index_size()

        decoder_options = LexiconDecoderOptions(
            beam_size=beam_size,
            beam_size_token=beam_size_token,
            beam_threshold=beam_threshold,
            lm_weight=lm_weight,
            word_score=word_score,
            unk_score=unk_score,
            sil_score=sil_score,
            log_add=log_add,
            criterion_type=CriterionType.CTC,
        )

        self.word_dict = create_word_dict(lexicon)
        lm = KenLM(lm_path, self.word_dict)

        self.sil_idx = tokens_dict.get_index(processor.delim_token)
        self.unk_idx = self.word_dict.get_index(processor.unk_token)
        self.blank_idx = tokens_dict.get_index(processor.pad_token)

        trie = self.__construct_trie(tokens_dict, self.word_dict, lexicon, lm, self.sil_idx)

        token_lm = False
        transitions = []
        
        self.decoder = LexiconDecoder(
            decoder_options,
            trie,
            lm,
            self.sil_idx,
            self.blank_idx,
            self.unk_idx,
            transitions,
            token_lm,
        )

        self.nbest = nbest

        self.float_bytes = 4

    def __construct_trie(self, tokens_dict: Dictionary, word_dict: Dictionary, lexicon: Dict[str, List[List[str]]], lm: KenLM, silence: str) -> Trie:
        vocab_size = tokens_dict.index_size()
        trie = Trie(vocab_size, silence)
        start_state = lm.start(False)

        for word, spellings in lexicon.items():
            word_idx = word_dict.get_index(word)
            _, score = lm.score(start_state, word_idx)
            for spelling in spellings:
                spelling_idx = [tokens_dict.get_index(token) for token in spelling]
                trie.insert(spelling_idx, word_idx, score)

        trie.smear(SmearingMode.MAX)
        return trie
    
    def _to_hypo(self, results: List[DecodeResult]) -> List[str]:
        results_dict = dict()

        for result in results:
            pred = self.post_process_s2t(" ".join([self.word_dict.get_entry(x) for x in result.words if x >= 0]))
            pred_score = result.score
            if len(self.hotwords_dict) != 0:
                for hotword, hotword_score in self.hotwords_dict.items():
                    num_appearances = pred.count(hotword)
                    if num_appearances > 0:
                        pred_score += num_appearances + hotword_score
            results_dict[pred] = pred_score

        results_dict = {k: v for k, v in sorted(results_dict.items(), key=lambda item: item[1])}

        texts = list(results_dict.keys())[:self.nbest]
        return texts
    
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
    
    def __call__(self, emissions: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> List[str]:
        if emissions.ndim == 2:
            emissions = emissions.unsqueeze(0)
        
        if emissions.device != 'cpu':
            emissions = emissions.cpu()
            if lengths is not None:
                lengths = lengths.cpu()
        
        batch_size, time_steps, _ = emissions.size()

        if lengths is None:
            lengths = torch.full((batch_size, ), time_steps)

        hypos = []

        for batch_idx in range(batch_size):
            emissions_ptr = emissions.data_ptr() + self.float_bytes * batch_idx * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, lengths[batch_idx], self.vocab_size)
            hypos.append(self._to_hypo(results[: self.nbest]))
        
        if batch_size == 1:
            return hypos[0]
        return hypos

# class KenLanguageModel:
#     def __init__(self, 
#                  lm_path: str, 
#                  vocab: List[str], 
#                  alpha: float = 2.1, 
#                  beta: float = 9.2,
#                  beam_width: int = 190,
#                  beam_prune_logp: float = -20,
#                  hotwords: Union[List[str], Iterable[str]] =  ['fpt', 'wifi', 'modem', 'fpt telecom', 'internet',   'free',  'lag', 'tổng đài viên', 'reset', 'check', 'test', 'code', 'port', 'net', 'email', 'mail', 'box'],
#                  hotword_weight: Optional[float] = 9.0,
#                 ) -> None:
#         self.decoder = build_ctcdecoder(
#             labels=vocab,
#             kenlm_model_path=lm_path,
#             alpha=alpha,
#             beta=beta
#         )

#         self.beam_width = beam_width
#         self.beam_prune_logp = beam_prune_logp
#         self.hotwords = hotwords
#         self.hotword_weight = hotword_weight

#     def decode(self, logits: np.ndarray) -> str:
#         return self.post_process_s2t(
#             self.decoder.decode(
#                 logits,
#                 beam_width=self.beam_width,
#                 beam_prune_logp=self.beam_prune_logp,
#                 hotwords=self.hotwords,
#                 hotword_weight=self.hotword_weight
#             )
#         )
    
#     def post_process_s2t(self, raw: str) -> str:
#         if not raw or len(raw)==1:
#             return ''
#         text = raw.replace('ti vi', 'tivi').replace('goai phoai', 'wifi')
#         text = text.replace('côm bô', 'combo').replace("on lai", "online").replace("ốp lai", "offline")
#         text = text.replace('ca cộng', 'k cộng').replace("gia lô", "zalo")
#         text = text.replace('ốp thiết bị', 'off thiết bị')
#         text = text.replace('cờ lao', 'cloud').replace("nhận mát", "nhận mac").replace("nhận mắc", "nhận mac")
#         text = text.replace('in tơ net','internet').replace('in tơ nét','internet')
#         text = text.replace('ca me ra', 'camera').replace('ca mê ra', 'camera').replace("cam mê ra", "camera").replace("cam me ra", "camera")
#         text = text.replace("a bi quan", "ip1").replace("i bi quan", "ip1").replace("i bi a", "ip1").replace("con vật to", "converter")
#         text = text.replace("vốt chơ", "voucher ").replace("lô gô", "logo")
#         text = text.replace("lên áp", "lên app").replace("cái áp", "cái app").replace("meo", "mail").replace("thu lại bọt", "thu lại port").replace("thu bọt", "thu port")
#         text = text.replace("hai fpt", "hi fpt").replace("tàu lâu", "tào lao").replace("gửi mai", "gửi mail")
#         text = ' '.join([word for word in text.split(' ') if word not in ['n','d','h','g']])
#         text = re.sub("\s\s+", " ", text)
#         return text.strip()

#     def decode_batch(self, logits: np.ndarray, lengths: Optional[np.ndarray] = None, decode_func: Optional[Callable[[str], str]] = None) -> List[str]:
#         preds = []
#         for index, probs in enumerate(logits):
#             text = self.decode(probs[: lengths[index], :] if lengths is not None else probs)
#             if decode_func is not None:
#                 text = decode_func(text)
#             preds.append(text)
#         return preds