"""
#   Copyright (C) 2022 BAIDU CORPORATION. All rights reserved.
#   Author     :   tanglicheng@baidu.com
#   Date       :   2022-02-10
"""
from functools import partial
import paddle.nn as nn
from .modules import ConvBlock, ResBlock


class ContentEncoder(nn.Layer):
    """
    ContentEncoder
    """
    def __init__(self, layers, sigmoid=False):
        super().__init__()
        self.net = nn.Sequential(*layers)
        self.sigmoid = sigmoid

    def forward(self, x):
        out = self.net(x)
        if self.sigmoid:
            out = nn.Sigmoid()(out)
        return out


def content_enc_builder(C_in, C, C_out, norm='none', activ='relu', weight_init='xavier', content_sigmoid=False):
    """
    content_enc_builder
    """
    ConvBlk = partial(ConvBlock, norm=norm, activ=activ, weight_init=weight_init)
    ResBlk = partial(ResBlock, norm=norm, activ=activ, weight_init=weight_init)

    layers = [
        ConvBlk(C_in, C, 3, 1, 1, norm='in', activ='relu'),
        ConvBlk(C*1, C*2, 3, 2, 1),  # 64x64
        ConvBlk(C*2, C*4, 3, 2, 1),  # 32x32
        ConvBlk(C*4, C*8, 3, 2, 1),  # 16x16
        ConvBlk(C*8, C_out, 3, 1, 1)
    ]

    return ContentEncoder(layers, content_sigmoid)
