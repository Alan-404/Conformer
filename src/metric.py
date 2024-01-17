from torchmetrics.text import WordErrorRate
from typing import List

wer = WordErrorRate()

def WER_score(preds: List[str], labels: List[str]) -> float:
    return wer(
        preds=preds,
        target=labels
    ).item()