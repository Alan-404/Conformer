import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics.text import WordErrorRate

from typing import Optional, Union, List

class ConformerCriterion:
    def __init__(self, blank_id: Optional[int] = None) -> None:
        if blank_id is not None:
            self.ctc_criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    def ctc_loss(self, outputs: torch.Tensor, targets: torch.Tensor, input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        return self.ctc_criterion(outputs.log_softmax(dim=-1).transpose(0,1), targets, input_lengths, target_lengths)
    
    def contrastive_loss(self, context: torch.Tensor, target: torch.Tensor):
        k = context.size(0)

        context = context.flatten(start_dim=1)
        target = context.flatten(start_dim=1)

        loss = 0.0

        for i in range(k):
            similaries = []
            for j in range(k):
                similaries.append(F.cosine_similarity(context[i], target[j], dim=-1))
            
            loss += similaries[i] / torch.sum(torch.tensor(similaries))

        return -loss
    
    def __l2_norm(self, online: torch.Tensor, target: torch.Tensor):
        online = F.normalize(online, dim=-1, p=2)
        target = F.normalize(target, dim=-1, p=2)
        return 2 - 2 * (online * target).sum(dim=-1)
    
    def l2_norm(self, online: torch.Tensor, target: torch.Tensor):
        batch_size = online.size(0)
        loss = 0.0
        for idx in range(batch_size):
            loss += self.__l2_norm(online[idx], target[idx])

        return loss.mean()

class ConformerMetric:
    def __init__(self) -> None:
        self.wer_metric = WordErrorRate()

    def wer_score(self, pred: Union[List[str], str], label: Union[List[str], str]) -> torch.Tensor:
        return self.wer_metric(pred, label)