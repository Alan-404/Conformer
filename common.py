from collections import OrderedDict

def change_format_single_gpu(dict: OrderedDict):
    new_dict = OrderedDict()
    for key, value in dict.items():
        new_key = key.replace("module.", '', 1)
        new_dict[new_key] = value
    return new_dict