import torchmetrics.functional as F
from typing import List

def WER_score(preds: List[str], labels: List[str]) -> float:
    return F.word_error_rate(
        preds=preds,
        target=labels
    ).item()