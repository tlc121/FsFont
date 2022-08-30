"""
#   Copyright (C) 2022 BAIDU CORPORATION. All rights reserved.
#   Author     :   tanglicheng@baidu.com
#   Date       :   2022-02-10
"""
import copy
from functools import partial
import paddle
import paddle.nn as nn
# import torch
# import torch.nn as nn
from .modules import ConvBlock, ResBlock


class Integrator(nn.Layer):
    """
    Integrator
    """
    def __init__(self, C, norm='none', activ='none', weight_init='xavier', C_content=0):
        super().__init__()
        C_in = C + C_content
        self.integrate_layer = ConvBlock(C_in, C, 1, 1, 0, norm=norm, activ=activ, weight_init=weight_init)

    def forward(self, comps, content=None):
        """
        Args:
            comps [B, 3, mem_shape]: component features
        """
        inputs = paddle.concat([comps, content], axis=1)
        out = self.integrate_layer(inputs)
        return out


class Decoder(nn.Layer):
    """
    Decoder
    """
    def __init__(self, layers, skips=None, out='sigmoid'):
        super().__init__()
        self.layers = nn.LayerList(layers)

        if out == 'sigmoid':
            self.out = nn.Sigmoid()
        elif out == 'tanh':
            self.out = nn.Tanh()
        else:
            raise ValueError(out)

    def forward(self, x, content_feats=None):
        """
        forward
        """
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x, content=content_feats)
            else:
                x = layer(x)

        return self.out(x)


def dec_builder(C, C_out, norm='none', activ='relu', out='sigmoid', weight_init='xavier', C_content=0):
    """
    dec_builder
    """
    ConvBlk = partial(ConvBlock, norm=norm, activ=activ, weight_init=weight_init)
    ResBlk = partial(ResBlock, norm=norm, activ=activ, weight_init=weight_init)
    IntegrateBlk = partial(Integrator, norm='in', activ='relu', weight_init=weight_init)

    layers = [
        IntegrateBlk(C*8, C_content=C_content),
        ResBlk(C*8, C*8, 3, 1),
        ResBlk(C*8, C*8, 3, 1),
        ResBlk(C*8, C*8, 3, 1),
        ConvBlk(C*8, C*4, 3, 1, 1, upsample=True),   # 32x32
        ConvBlk(C*4, C*2, 3, 1, 1, upsample=True),   # 64x64
        ConvBlk(C*2, C*1, 3, 1, 1, upsample=True),   # 128x128
        ConvBlk(C*1, C_out, 3, 1, 1)
    ]

    return Decoder(layers, out=out)
