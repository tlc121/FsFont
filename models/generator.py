"""
#   Copyright (C) 2022 BAIDU CORPORATION. All rights reserved.
#   Author     :   tanglicheng@baidu.com
#   Date       :   2022-02-10
"""
import math
import copy
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .comp_encoder import comp_enc_builder
from .content_encoder import content_enc_builder
from .decoder import dec_builder
from .memory import Memory


class Generator(nn.Layer):
    """
    Generator
    """
    def __init__(self, C_in, C, C_out, cfg, comp_enc, dec, content_enc):
        super().__init__()
        self.component_encoder = comp_enc_builder(C_in, C, **comp_enc)
        self.mem_shape = self.component_encoder.final_shape
        assert self.mem_shape[-1] == self.mem_shape[-2]  # H == W
        
        #memory
        self.memory = Memory()

        self.num_heads = cfg.num_heads
        self.shot = cfg.kshot
        num_channels = 256
        
        self.linears_key = nn.Linear(num_channels, num_channels, bias_attr=False)
        self.linears_value = nn.Linear(num_channels, num_channels, bias_attr=False)
        self.linears_query = nn.Linear(num_channels, num_channels, bias_attr=False)

        self.fc = nn.Linear(num_channels, num_channels, bias_attr=False)
        self.layer_norm = nn.LayerNorm(num_channels,  epsilon=1e-6)

        C_content = content_enc['C_out']
        self.content_encoder = content_enc_builder(
            C_in, C, **content_enc
        )

        self.decoder = dec_builder(
            C, C_out, **dec, C_content=C_content
        )

    def reset_memory(self):
        """
        reset_memory
        """
        self.memory.reset_memory()

    def get_kqv_matrix(self, fm, linears):
        #matmul with style featuremaps and content featuremaps
        ret = linears(fm)
        return ret

    def encode_write_comb(self, style_ids, style_sample_index, style_imgs, reset_memory=True):
        """
        encode_write_comb
        """
        if reset_memory:
            self.reset_memory()

        feats = self.component_encoder(style_imgs)
        feat_scs = feats["last"]
        self.memory.write_comb(style_ids, style_sample_index, feat_scs)
        return feat_scs

    def read_memory(self, target_style_ids, trg_sample_index, reset_memory=True, \
                    reduction='mean'):
        """
        read_memory
        """
        feats = self.memory.read_chars(target_style_ids, trg_sample_index, reduction=reduction)

        feats = paddle.stack([x for x in feats]) #[B,3,C,H,W]
        batch, shot, channel, h, w = feats.shape
        feats = paddle.transpose(feats, perm=[0, 1, 3, 4, 2]) #B,3HW,C
        feats_reshape = paddle.reshape(feats, (batch, shot*h*w, channel)) #先只用最后一层做transformer B,3HW,C

        ######### attention ########
        d_channel = int(channel / self.num_heads)
        #size: [B, 3HW, num_heads, C/num_head]
        key_matrix = self.get_kqv_matrix(feats_reshape, self.linears_key)
        key_matrix = paddle.reshape(key_matrix, (batch, shot*h*w, self.num_heads, d_channel))
        value_matrix = self.get_kqv_matrix(feats_reshape, self.linears_value)
        value_matrix = paddle.reshape(value_matrix, (batch, shot*h*w, self.num_heads, d_channel))
        key_matrix = paddle.transpose(key_matrix, perm=[0, 2, 1, 3]) #[B, num_heads, 3HW, C/num_heads]
        value_matrix = paddle.transpose(value_matrix, perm=[0, 2, 1, 3])

        if reset_memory:
            self.reset_memory()

        return key_matrix, value_matrix

    def read_decode(self, target_style_ids, trg_sample_index, content_imgs, reset_memory=True, \
                    reduction='mean'):
        """
        read_decode
        """
        key_matrix, value_matrix = self.read_memory(target_style_ids, trg_sample_index, reset_memory, reduction=reduction)

        #[B,C,H,W]
        content_feats = self.content_encoder(content_imgs) #B,C,H,W
        content_feats_permute = paddle.transpose(content_feats, perm=[0, 2, 3, 1]) #B,H,W,C
        batch, h, w, channel = content_feats_permute.shape
        d_channel = int(channel / self.num_heads)
        content_feats_reshape = paddle.reshape(content_feats_permute, (batch, h*w, channel)) #B, HW, C
        query_matrix = self.get_kqv_matrix(content_feats_reshape, self.linears_query)
        residual = query_matrix
        query_matrix = paddle.reshape(query_matrix, (batch, h*w, self.num_heads, d_channel)) #[B, HW, num_heads, C/num_heads]
        query_matrix = paddle.transpose(query_matrix, perm=[0, 2, 3, 1]) #[B, num_heads, C/num_heads, HW]
        
        ######### attention ########
        #softmax & square root
        #[B, num_heads, HW, 3HW]
        attention_mask = paddle.matmul(key_matrix, query_matrix) #[B, num_heads, 3HW, HW]
        attention_mask = paddle.transpose(attention_mask, perm=[0, 1, 3, 2]) / math.sqrt(h*w)#[B, num_heads, HW, 3HW]
        attention_mask = F.softmax(attention_mask, axis = -1)

        #[B, num_heads, C/num_heads, HW]
        value_mask = paddle.matmul(attention_mask, value_matrix)     #[B, num_heads, HW, C/num_heads]
        value_mask = paddle.transpose(value_mask, perm=[0, 1, 3, 2]) #[B, num_heads, C/num_heads, HW]
        value_mask = paddle.reshape(value_mask, (batch, channel, -1))
        value_mask = paddle.transpose(value_mask, perm=[0, 2, 1]) #[B, HW, C]
        value_mask = self.fc(value_mask)
        value_mask += residual #[B, HW, C]
        value_mask = self.layer_norm(value_mask)
        value_mask = paddle.transpose(value_mask, perm=[0, 2, 1])
        feat_scs = paddle.reshape(value_mask, (batch, channel, h, w))
        out = self.decoder(feat_scs, content_feats=content_feats)

        if reset_memory:
            self.reset_memory()

        return out, feat_scs

    def infer(self, in_style_ids, in_imgs, trg_style_ids, style_sample_index, trg_sample_index, content_imgs, 
              reduction="mean"):
        """
        infer
        """
        in_style_ids = in_style_ids.cuda()
        in_imgs = in_imgs.cuda()
        trg_style_ids = trg_style_ids.cuda()
        content_imgs = content_imgs.cuda()

        self.encode_write_comb(in_style_ids, style_sample_index, in_imgs)

        out, feat_scs = self.read_decode(trg_style_ids, trg_sample_index, content_imgs=content_imgs,
                               reduction=reduction)

        return out, feat_scs
