from collections import OrderedDict

def map_weights(checkpoint: OrderedDict):
    checkpoint = OrderedDict((key.replace("model.", ""), value) for key, value in checkpoint.items())
    return checkpoint