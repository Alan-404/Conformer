from collections import OrderedDict
from typing import Any

def map_weights(checkpoint: OrderedDict):
    checkpoint = OrderedDict((key.replace("model.", ""), value) for key, value in checkpoint.items())
    return checkpoint