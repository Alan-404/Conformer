import torch
import torch.nn.functional as F

def ctc_loss(outputs: torch.Tensor, targets: torch.Tensor, output_lengths: torch.Tensor, target_lengths: torch.Tensor, blank_id: int, reduction: str = 'mean', zero_infinity: bool = True) -> torch.Tensor:
    return F.ctc_loss(
        log_probs=outputs.log_softmax(dim=-1).transpose(0,1),
        targets=targets,
        input_lengths=output_lengths,
        target_lengths=target_lengths,
        blank=blank_id,
        reduction=reduction,
        zero_infinity=zero_infinity
    )