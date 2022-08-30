"""
#   Copyright (C) 2022 BAIDU CORPORATION. All rights reserved.
#   Author     :   tanglicheng@baidu.com
#   Date       :   2022-02-10
"""

import paddle
import paddle.nn as nn

def load_checkpoint(path, gen, disc, g_optim, d_optim, g_scheduler, d_scheduler):
    """
    load_checkpoint
    """
    ckpt = paddle.load(path)

    gen.set_state_dict(ckpt['generator'])
    g_optim.set_state_dict(ckpt['optimizer'])
    g_scheduler.set_state_dict(ckpt['g_scheduler'])

    if disc is not None:
        disc.set_state_dict(ckpt['discriminator'])
        d_optim.set_state_dict(ckpt['d_optimizer'])
        d_scheduler.set_state_dict(ckpt['d_scheduler'])

    st_epoch = ckpt['epoch'] + 1
    loss = ckpt['loss']

    return st_epoch, loss
