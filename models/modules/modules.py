import numpy as np
import paddle
import paddle.nn as nn

def spectral_norm(module):
    """ init & apply spectral norm """
    return nn.utils.spectral_norm(module)
