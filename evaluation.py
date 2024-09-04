import torch
import torch.nn as nn

from torchmetrics import WordErrorRate, CharErrorRate

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
        self.wer_metric = WordErrorRate()
        self.cer_metric = CharErrorRate()

    def wer_score(self, prediction: Union[str, List[str]], target: Union[str, List[str]]) -> torch.Tensor:
        return self.wer_metric(prediction, target)
    
    def cer_score(self, prediction: Union[str, List[str]], target: Union[str, List[str]]) -> torch.Tensor:
        return self.cer_metric(prediction, target)
    