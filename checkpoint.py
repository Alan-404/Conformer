import torch
from model.conformer import Conformer
from typing import Union

from collections import OrderedDict

def change_format_single_gpu(dict: OrderedDict) -> OrderedDict:
    new_dict = OrderedDict()
    for key, value in dict.items():
        new_key = key.replace("module.", '', 1)
        new_dict[new_key] = value
    return new_dict

def load_model(state_dict: Union[str, OrderedDict], model: Conformer, world_size: int = 1) -> None:
    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict, map_location='cpu')['model']
    
    if world_size == 1:
        state_dict = change_format_single_gpu(state_dict)
    model.load_state_dict(state_dict)