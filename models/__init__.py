from .CMSLSTM import *


def get_convrnn_model(name, **kwargs):
    models = {
        'cmslstm': get_cmslstm,
    }
    return models[name.lower()](**kwargs)
