"""
#   Copyright (C) 2022 BAIDU CORPORATION. All rights reserved.
#   Author     :   tanglicheng@baidu.com
#   Date       :   2022-02-10
"""
import copy
from functools import partial
import paddle.nn as nn
from .modules import w_norm_dispatch
from .modules import ConvBlock, ResBlock


class ComponentEncoder(nn.Layer):
    """
    ComponentEncoder
    """
    def __init__(self, body, final_shape, sigmoid=False):
        super().__init__()

        #add condition layer
        self.body = nn.LayerList(body)
        self.final_shape = final_shape
        self.sigmoid = sigmoid

    def forward(self, x):
        ret_feats = {}
        for layer in self.body:
            x = layer(x)

        ret_feats["last"] = x
        if self.sigmoid:
            ret_feats = {k: nn.Sigmoid()(v) for k, v in ret_feats.items()}

        return ret_feats


def comp_enc_builder(C_in, C, norm='none', activ='relu', weight_init='xavier', skip_scale_var=False, sigmoid=True):
    """
    comp_enc_builder
    """
    ConvBlk = partial(ConvBlock, norm=norm, activ=activ, weight_init=weight_init)
    ResBlk = partial(ResBlock, norm=norm, activ=activ, weight_init=weight_init, scale_var=skip_scale_var)

    body = [
        ConvBlk(C_in, C, 3, 1, 1, norm='in', activ='relu'), #128x128x32
        ConvBlk(C * 1, C * 2, 3, 1, 1, downsample=True),  # 64x64x64
        ConvBlk(C * 2, C * 4, 3, 1, 1, downsample=True),  # 32x32x128
        ResBlk(C * 4, C * 4, 3, 1),
        ResBlk(C * 4, C * 4, 3, 1),
        ResBlk(C * 4, C * 8, 3, 1, downsample=True),  # 16x16x256
        ResBlk(C * 8, C * 8)
        ]
    
    final_shape = (C * 8, 16, 16)

    return ComponentEncoder(body, final_shape, sigmoid)

