import torch
import torch.nn as nn

class SpecAugment(nn.Module):
    def __init__(
        self,
        n_time_masks: int,
        time_mask_param: int,
        n_freq_masks: int,
        freq_mask_param: int,
        iid_masks: bool = True,
        p: float = 1.0,
        zero_masking: bool = False,
    ) -> None:
        super(SpecAugment, self).__init__()
        self.n_time_masks = n_time_masks
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.freq_mask_param = freq_mask_param
        self.iid_masks = iid_masks
        self.p = p
        self.zero_masking = zero_masking

    def forward(self, specgram: torch.Tensor) -> torch.Tensor:
        if self.zero_masking:
            mask_value = 0.0
        else:
            mask_value = specgram.mean()
        time_dim = specgram.dim() - 1
        freq_dim = time_dim - 1

        if specgram.dim() > 2 and self.iid_masks is True:
            for _ in range(self.n_time_masks):
                specgram = mask_along_axis_iid(specgram, self.time_mask_param, mask_value, time_dim, p=self.p)
            for _ in range(self.n_freq_masks):
                specgram = mask_along_axis_iid(specgram, self.freq_mask_param, mask_value, freq_dim, p=self.p)
        else:
            for _ in range(self.n_time_masks):
                specgram = mask_along_axis(specgram, self.time_mask_param, mask_value, time_dim, p=self.p)
            for _ in range(self.n_freq_masks):
                specgram = mask_along_axis(specgram, self.freq_mask_param, mask_value, freq_dim, p=self.p)

        return specgram
        

def mask_along_axis_iid(
    specgrams: torch.Tensor,
    mask_param: int,
    mask_value: float,
    axis: int,
    p: float = 1.0,
) -> torch.Tensor:

    dim = specgrams.dim()

    if dim < 3:
        raise ValueError(f"Spectrogram must have at least three dimensions ({dim} given).")

    if axis not in [dim - 2, dim - 1]:
        raise ValueError(
            f"Only Frequency and Time masking are supported (axis {dim-2} and axis {dim-1} supported; {axis} given)."
        )

    if not 0.0 <= p <= 1.0:
        raise ValueError(f"The value of p must be between 0.0 and 1.0 ({p} given).")

    mask_param = _get_mask_param(mask_param, p, specgrams.shape[axis])
    if mask_param < 1:
        return specgrams

    device = specgrams.device
    dtype = specgrams.dtype

    value = torch.rand(specgrams.shape[: (dim - 2)], device=device, dtype=dtype) * mask_param
    min_value = torch.rand(specgrams.shape[: (dim - 2)], device=device, dtype=dtype) * (specgrams.size(axis) - value)

    # Create broadcastable mask
    mask_start = min_value.long()[..., None, None]
    mask_end = (min_value.long() + value.long())[..., None, None]
    mask = torch.arange(0, specgrams.size(axis), device=device, dtype=dtype)

    # Per batch example masking
    specgrams = specgrams.transpose(axis, -1)
    specgrams = specgrams.masked_fill((mask >= mask_start) & (mask < mask_end), mask_value)
    specgrams = specgrams.transpose(axis, -1)

    return specgrams

def mask_along_axis(
    specgram: torch.Tensor,
    mask_param: int,
    mask_value: float,
    axis: int,
    p: float = 1.0,
) -> torch.Tensor:
    dim = specgram.dim()

    if dim < 2:
        raise ValueError(f"Spectrogram must have at least two dimensions (time and frequency) ({dim} given).")

    if axis not in [dim - 2, dim - 1]:
        raise ValueError(
            f"Only Frequency and Time masking are supported (axis {dim-2} and axis {dim-1} supported; {axis} given)."
        )

    if not 0.0 <= p <= 1.0:
        raise ValueError(f"The value of p must be between 0.0 and 1.0 ({p} given).")

    mask_param = _get_mask_param(mask_param, p, specgram.shape[axis])
    if mask_param < 1:
        return specgram

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape([-1] + list(shape[-2:]))
    # After packing, specgram is a 3D tensor, and the axis corresponding to the to-be-masked dimension
    # is now (axis - dim + 3), e.g. a tensor of shape (10, 2, 50, 10, 2) becomes a tensor of shape (1000, 10, 2).
    value = torch.rand(1) * mask_param
    min_value = torch.rand(1) * (specgram.size(axis - dim + 3) - value)

    mask_start = (min_value.long()).squeeze()
    mask_end = (min_value.long() + value.long()).squeeze()
    mask = torch.arange(0, specgram.shape[axis - dim + 3], device=specgram.device, dtype=specgram.dtype)
    mask = (mask >= mask_start) & (mask < mask_end)
    # unsqueeze the mask if the axis is frequency
    if axis == dim - 2:
        mask = mask.unsqueeze(-1)

    if mask_end - mask_start >= mask_param:
        raise ValueError("Number of columns to be masked should be less than mask_param")

    specgram = specgram.masked_fill(mask, mask_value)

    # unpack batch
    specgram = specgram.reshape(shape[:-2] + specgram.shape[-2:])

    return specgram

def _get_mask_param(mask_param: int, p: float, axis_length: int) -> int:
    if p == 1.0:
        return mask_param
    else:
        return min(mask_param, int(axis_length * p))