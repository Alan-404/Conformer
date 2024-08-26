import torch
from model.conformer import Conformer
from typing import Union

from collections import OrderedDict

def is_ddp_checkpoint(state_dict: OrderedDict):
    for key in state_dict.keys():
        if key.startswith("module.") == False:
            return False
    return True

def change_format_single_gpu(dict: OrderedDict) -> OrderedDict:
    new_dict = OrderedDict()
    for key, value in dict.items():
        new_key = key.replace("module.", '', 1)
        new_dict[new_key] = value
    return new_dict

def change_format_multi_gpus(dict: OrderedDict) -> OrderedDict:
    new_dict = OrderedDict()
    for key, value in dict.items():
        new_key = f"module.{key}"
        new_dict[new_key] = value
    return new_dict

def load_model(state_dict: Union[str, OrderedDict], model: Conformer, world_size: int = 1) -> None:
    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict, map_location='cpu')['model']
    is_ddp = is_ddp_checkpoint(state_dict)
    if world_size == 1 and is_ddp:
        state_dict = change_format_single_gpu(state_dict)
    elif world_size > 1 and is_ddp == False:
        state_dict = change_format_multi_gpus(state_dict)
    model.load_state_dict(state_dict)