import torch
import torch.nn as nn

import jiwer

from typing import Optional, Union, List

class ConformerCriterion:
    def __init__(self, blank_id: Optional[int] = None) -> None:
        if blank_id is not None:
            self.ctc_criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    def ctc_loss(self, outputs: torch.Tensor, targets: torch.Tensor, input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        outputs = outputs.float()
        targets = targets.float()
        
        return self.ctc_criterion(outputs.log_softmax(dim=-1).transpose(0,1), targets, input_lengths, target_lengths)

class ConformerMetric:
    def __init__(self) -> None:
        pass

    def wer_score(self, hypothesis: Union[str, List[str]], references: Union[str, List[str]]) -> float:
        return jiwer.wer(references, hypothesis)
    
    def cer_score(self, hypothesis: Union[str, List[str]], references: Union[str, List[str]]) -> float:
        return jiwer.cer(references, hypothesis)