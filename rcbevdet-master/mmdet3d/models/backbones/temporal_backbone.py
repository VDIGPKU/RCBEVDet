# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import warnings
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.cnn import (Linear, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.registry import (ATTENTION, FEEDFORWARD_NETWORK, POSITIONAL_ENCODING,
                       TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import build_positional_encoding, TransformerLayerSequence, build_transformer_layer_sequence
from mmcv.runner import force_fp32, auto_fp16
from ..builder import BACKBONES

class LayerNorm2d(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class TemporalDecoder(TransformerLayerSequence):
    def __init__(self, *args, **kwargs):
        super(TemporalDecoder, self).__init__(*args, **kwargs)

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='2d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        if dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = bev_query
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        prev_bev = prev_bev.permute(1, 0, 2)

        bs, len_bev, num_bev_level, _ = ref_2d.shape
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=ref_2d,
                ref_3d=None,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=None,
                level_start_index=None,
                reference_points_cam=None,
                bev_mask=None,
                prev_bev=prev_bev,
                **kwargs)
            bev_query = output
        return output


@BACKBONES.register_module()
class BiTemporalPredictor(BaseModule):
    """Implements feed-forward networks (FFNs) with identity connection.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
    """
    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 embed_dims=256,
                 feedforward_channels=512,
                 num_adj=4,
                 reduction=4,
                 num_short=2,
                 with_query=True,
                 with_sin_embedding=False,
                 decoder_long=None,
                 decoder_short=None,
                 bev_h=200,
                 bev_w=200,
                 positional_encoding_time=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 positional_encoding_long=dict(
                     type='SinePositionalEncoding',
                     num_feats=32,
                     normalize=True),
                 positional_encoding_short=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        positional_encoding_time['num_feats'] = embed_dims//2
        positional_encoding_long['num_feats'] = embed_dims//2//reduction
        positional_encoding_short['num_feats'] = embed_dims//2
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_adj = num_adj
        self.num_short = num_short
        self.reduction = reduction
        self.with_query = with_query
        self.with_sin_embedding = with_sin_embedding
        self.decoder_long = build_transformer_layer_sequence(decoder_long)
        self.decoder_short = build_transformer_layer_sequence(decoder_short)
        self.positional_encoding_long = build_positional_encoding(
            positional_encoding_long)
        self.positional_encoding_short = build_positional_encoding(
            positional_encoding_short)  

        if self.with_sin_embedding:
            self.frame_embeds_short = build_positional_encoding(positional_encoding_time)
            positional_encoding_time['num_feats'] = 32
            self.frame_embeds_long = build_positional_encoding(positional_encoding_time)
        else:
            self.frame_embeds = nn.Parameter(torch.Tensor(
                (self.num_adj+1), self.embed_dims))
            
        if self.with_query:
            self.queries_long = nn.Embedding(
                    bev_h*bev_w, self.embed_dims//self.reduction)
            self.queries_short = nn.Embedding(
                    bev_h*bev_w, self.embed_dims)
        self.input_proj = nn.Sequential(
                nn.Linear(embed_dims, embed_dims//self.reduction),
                nn.LayerNorm(embed_dims//self.reduction)
            )
        self.output_proj = nn.Sequential(
                nn.Conv2d(embed_dims+embed_dims//self.reduction, out_channels, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_channels)
            )
        if not self.in_channels==self.embed_dims:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, embed_dims, kernel_size=1, bias=False),
                LayerNorm2d(embed_dims)
            )

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if 'TemporalCrossAttention' in m.__class__.__name__:
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        if not self.with_sin_embedding:
            nn.init.normal_(self.frame_embeds)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def forward(
            self,
            bev_feat_list,
            **kwargs):
        """
        obtain bev features.
        """
        # T, T-2, T-3, T-4
        bs, C, bev_h, bev_w = bev_feat_list[0].shape
        dtype = bev_feat_list[0].dtype
        bev_mask = torch.zeros((bs, bev_h, bev_w),
                               device=bev_feat_list[0].device).to(dtype)
        bev_pos_long = self.positional_encoding_long(bev_mask).to(dtype)
        bev_pos_long = bev_pos_long.flatten(2).permute(2, 0, 1)
        bev_pos_short = self.positional_encoding_short(bev_mask).to(dtype)
        bev_pos_short = bev_pos_short.flatten(2).permute(2, 0, 1)

        if not self.in_channels==self.embed_dims:
            results = torch.stack(bev_feat_list, dim=1).reshape(bs*len(bev_feat_list), C, bev_h, bev_w)
            results = self.proj(results)
            C = self.embed_dims
            results = results.reshape(bs, len(bev_feat_list), C, bev_h, bev_w).transpose(1, 0).contiguous()
            bev_feat_list = [feat for feat in results]

        values = []
        if self.with_sin_embedding:
            time_mask = torch.zeros((bs, len(bev_feat_list)+1, 1),
                               device=bev_feat_list[0].device).to(dtype)
            time_pos_short = self.frame_embeds_short(time_mask).to(dtype).squeeze(-1)
            time_pos_short = time_pos_short.permute(2, 0, 1)
            time_pos_long = self.frame_embeds_long(time_mask).to(dtype).squeeze(-1)
            time_pos_long = time_pos_long.permute(2, 0, 1)
            frame_embeds_short = torch.cat((time_pos_short[:1], time_pos_short[2:3]), dim=0)
            frame_embeds_long = torch.cat((time_pos_long[:1], time_pos_long[2:]), dim=0)
            for i in range(len(bev_feat_list)):
                values.append(bev_feat_list[i])
        else:
            frame_embeds = self.frame_embeds
            for i in range(len(bev_feat_list)):
                values.append(frame_embeds[i].reshape(1, self.embed_dims, 1, 1).to(dtype) + bev_feat_list[i])
        values_short = values[:self.num_short]
        values_long = values

        values_long = torch.stack(values_long, dim=1).reshape(bs*(self.num_adj+1), C, bev_h*bev_w)
        values_long = self.input_proj(values_long.permute(0, 2, 1))
        values_long = values_long.permute(1, 0, 2)
        values_short = torch.stack(values_short, dim=1).reshape(bs*(self.num_short), C, bev_h*bev_w).permute(2, 0, 1)

        if self.with_query:
            bev_queries_long = self.queries_long.weight.to(dtype).unsqueeze(0).repeat(bs, 1, 1)
            bev_queries_long = bev_queries_long.permute(1, 0, 2).contiguous()
            bev_queries_short = self.queries_short.weight.to(dtype).unsqueeze(0).repeat(bs, 1, 1)
            bev_queries_short = bev_queries_short.permute(1, 0, 2).contiguous()
        else:
            bev_queries_long = values_long.reshape(bev_h*bev_w, bs, self.num_adj+1, C//self.reduction)[:, :, :1, :].reshape(bev_h*bev_w, bs, C//self.reduction)
            bev_queries_short = values_short.reshape(bev_h*bev_w, bs, self.num_short, C)[:, :, :1, :].reshape(bev_h*bev_w, bs, C)

        if self.with_sin_embedding:
            values_long = values_long + frame_embeds_long.permute(1, 0, 2).reshape(bs*(self.num_adj+1), C//self.reduction).unsqueeze(0)
            values_short = values_short + frame_embeds_short.permute(1, 0, 2).reshape(bs*self.num_short, C).unsqueeze(0)
            bev_queries_long = bev_queries_long + time_pos_long[1].unsqueeze(0)
            bev_queries_short = bev_queries_short + time_pos_short[1].unsqueeze(0)

        bev_embed = self.decoder_long(
            bev_queries_long,
            values_long,
            values_long,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos_long,
            prev_bev=values_long,
            **kwargs
        )
        bev_embed = bev_embed.reshape(bs, bev_h, bev_w, C//self.reduction).permute(0, 3, 1, 2)
        bev_embed_long = bev_embed
        bev_embed = self.decoder_short(
            bev_queries_short,
            values_short,
            values_short,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos_short,
            prev_bev=values_short,
            **kwargs
        )
        bev_embed = bev_embed.reshape(bs, bev_h, bev_w, C).permute(0, 3, 1, 2)
        bev_embed = self.output_proj(torch.cat((bev_embed, bev_embed_long), dim=1))
        return bev_embed