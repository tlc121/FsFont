from functools import partial
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .modules import spectral_norm


def dispatcher(dispatch_fn):
    def decorated(key, *args):
        if callable(key):
            return key

        if key is None:
            key = 'none'

        return dispatch_fn(key, *args)
    return decorated


@dispatcher
def norm_dispatch(norm):
    return {
        'none': nn.Identity,
        'in': nn.InstanceNorm2D,
        'bn': nn.BatchNorm2D
    }[norm.lower()]


@dispatcher
def w_norm_dispatch(w_norm):
    # NOTE Unlike other dispatcher, w_norm is function, not class.
    return {
        'spectral': spectral_norm,
        'none': lambda x: x
    }[w_norm.lower()]


@dispatcher
def activ_dispatch(activ, norm=None):
    return {
        "none": nn.Identity,
        "relu": nn.ReLU,
        "lrelu": partial(nn.LeakyReLU, negative_slope=0.2)
    }[activ.lower()]



class ConvBlock(nn.Layer):
    """ pre-active conv block """
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=1, norm='none',
                 activ='relu', bias=True, upsample=False, downsample=False, w_norm='none',
                 dropout=0., weight_init='xavier', size=None):
        # 1x1 conv assertion
        if kernel_size == 1:
            assert padding == 0
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        activ = activ_dispatch(activ, norm)
        norm = norm_dispatch(norm)
        w_norm = w_norm_dispatch(w_norm)
        self.upsample = upsample
        self.downsample = downsample
        if weight_init == 'gaussian':
            attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0))
        elif weight_init == 'xavier':
            attr = paddle.ParamAttr(initializer=paddle.nn.initializer.XavierNormal())
        elif weight_init == 'kaiming':
            attr = paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingNormal())

        self.norm = norm(C_in)
        self.activ = activ()
        
        if dropout > 0.:
            self.dropout = nn.Dropout2D(p=dropout)
            
        self.pad = nn.ZeroPad2D(padding)
        self.conv = w_norm(nn.Conv2D(C_in, C_out, kernel_size, stride, bias_attr=bias))

    def forward(self, x):
        _, c, h, w = x.shape
        if c != 1:
            x = self.norm(x)
        x = self.activ(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.conv(self.pad(x))
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x



class ResBlock(nn.Layer):
    """ Pre-activate ResBlock with spectral normalization """
    def __init__(self, C_in, C_out, kernel_size=3, padding=1, upsample=False, downsample=False,
                 norm='none', w_norm='none', activ='relu', weight_init='xavier', dropout=0., scale_var=False):
        assert not (upsample and downsample)
        super().__init__()
        w_norm = w_norm_dispatch(w_norm)
        self.C_in = C_in
        self.C_out = C_out
        self.upsample = upsample
        self.downsample = downsample
        self.scale_var = scale_var

        self.conv1 = ConvBlock(C_in, C_out, kernel_size, 1, padding, norm, activ,
                               upsample=upsample, w_norm=w_norm, weight_init=weight_init, 
                               dropout=dropout)
        self.conv2 = ConvBlock(C_out, C_out, kernel_size, 1, padding, norm, activ,
                               w_norm=w_norm, weight_init=weight_init, dropout=dropout)

        # XXX upsample / downsample needs skip conv
        if C_in != C_out or upsample or downsample:
            self.skip = w_norm(nn.Conv2D(C_in, C_out, 1))

    def forward(self, x):
        """
        normal: pre-activ + convs + skip-con
        upsample: pre-activ + upsample + convs + skip-con
        downsample: pre-activ + convs + downsample + skip-con
        => pre-activ + (upsample) + convs + (downsample) + skip-con
        """
        out = x
        out = self.conv1(out)
        out = self.conv2(out)

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        # skip-con
        if hasattr(self, 'skip'):
            if self.upsample:
                x = F.interpolate(x, scale_factor=2)
            x = self.skip(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)

        out = out + x
        if self.scale_var:
            out = out / np.sqrt(2)
        return out

