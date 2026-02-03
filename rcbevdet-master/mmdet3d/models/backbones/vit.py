import math
from functools import partial

import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.utils.checkpoint as cp

from detectron2.layers import CNNBlockBase, Conv2d, get_norm
from detectron2.modeling.backbone.fpn import _assert_strides_are_log2_contiguous
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.layers import ShapeSpec

from mmcv.runner import _load_checkpoint
from mmcv.runner import auto_fp16
from mmcv.runner import BaseModule
from ..builder import BACKBONES
from ...utils import get_root_logger

from einops import rearrange
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func
    from flash_attn.bert_padding import unpad_input
except:
    pass

try:
    from detectron2.modeling.backbone.utils import (
        PatchEmbed,
        add_decomposed_rel_pos,
        get_abs_pos,
        window_partition,
        window_unpartition,
        VisionRotaryEmbeddingFast,
    )
except:
    pass

try:
    import xformers.ops as xops
except:
    pass

try:
    from apex.normalization import FusedLayerNorm
except:
    pass


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0., 
                norm_layer=nn.LayerNorm, subln=False
            ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x
    

class Attention(nn.Module):
    def __init__(
            self, 
            dim, 
            num_heads=8, 
            qkv_bias=True, 
            qk_scale=None, 
            attn_head_dim=None, 
            rope=None,
            softmax_scale=None,
            attention_dropout=0.,
        ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
        self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
        self.v_proj = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.rope = rope
        # self.proj = nn.Linear(all_head_dim, dim) # 暂时不需要这个了

        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.fp16_enabled = True

    @auto_fp16(apply_to=('q', 'kv'), out_fp32=True)
    def forward(self, q, kv, 
                causal=False, 
                key_padding_mask=None):
        # flash attention
        assert q.dtype in [torch.float16, torch.bfloat16] and kv.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        assert q.shape[0] == kv.shape[0] and q.shape[-2] == kv.shape[-2] and q.shape[-1] == kv.shape[-1]

        batch_size = q.shape[0]
        seqlen_q, seqlen_k = q.shape[1], kv.shape[1]
        if key_padding_mask is None:
            q, kv = rearrange(q, 'b s ... -> (b s) ...'), rearrange(kv, 'b s ... -> (b s) ...')
            max_sq, max_sk = seqlen_q, seqlen_k 
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                                    device=kv.device)                    
            output = flash_attn_varlen_kvpacked_func( # cannot find function name same as streampetr, but find this which has same parameters
                q, kv, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )
            output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        else:
            nheads = kv.shape[-2]
            q = rearrange(q, 'b s ... -> (b s) ...')
            max_sq = seqlen_q
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)
            x = rearrange(kv, 'b s two h d -> b s (two h d)')
            x_unpad, indices, cu_seqlens_k, max_sk = unpad_input(x, key_padding_mask)
            x_unpad = rearrange(x_unpad, 'nnz (two h d) -> nnz two h d', two=2, h=nheads)
            output_unpad = flash_attn_varlen_kvpacked_func(
                q, x_unpad, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )
            output = rearrange(output_unpad, '(b s) ... -> b s ...', b=batch_size)

        # if self.xattn:
        #     q = q.permute(0, 2, 1, 3)   # B, num_heads, N, C -> B, N, num_heads, C
        #     k = k.permute(0, 2, 1, 3)
        #     v = v.permute(0, 2, 1, 3)
        #     x = xops.memory_efficient_attention(q, k, v)
        #     x = x.reshape(B, N, -1)
        # else:
            # q = q * self.scale
            # attn = (q @ k.transpose(-2, -1))
            # attn = attn.softmax(dim=-1).type_as(x)
            # x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        # x = self.proj(x)
        # x = x.view(B, H, W, C)

        # return x
        return output


class ResBottleneckBlock(CNNBlockBase):
    """
    The standard bottleneck residual block without the last activation layer.
    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        norm="LN",
        act_layer=nn.GELU,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            act_layer (callable): activation for all conv layers.
        """
        super().__init__(in_channels, out_channels, 1)

        self.conv1 = Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = get_norm(norm, bottleneck_channels)
        self.act1 = act_layer()

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            padding=1,
            bias=False,
        )
        self.norm2 = get_norm(norm, bottleneck_channels)
        self.act2 = act_layer()

        self.conv3 = Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = get_norm(norm, out_channels)

        for layer in [self.conv1, self.conv2, self.conv3]:
            weight_init.c2_msra_fill(layer)
        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        # zero init last norm layer.
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()

    def forward(self, x):
        out = x
        for layer in self.children():
            out = layer(out)

        out = x + out
        return out


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4*2/3,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        window_size=0,
        use_residual_block=False,
        rope=None,
        xattn=True,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            rope=rope,
            attention_dropout=0.1,
        )

        from timm.models.layers import DropPath

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = SwiGLU(
                in_features=dim, 
                hidden_features=int(dim * mlp_ratio), 
                subln=True,
                norm_layer=norm_layer,
            )

        self.window_size = window_size

        self.use_residual_block = use_residual_block
        if use_residual_block:
            # Use a residual block with bottleneck channel as dim // 2
            self.residual = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
                norm="LN",
            )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)

        # Window partition
        if self.window_size > 0:
            ori_H, ori_W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        B, H, W, C = x.shape
        x = x.view(B, -1, C)
        N = H * W

        q = F.linear(input=x, weight=self.attn.q_proj.weight, bias=self.attn.q_bias)
        k = F.linear(input=x, weight=self.attn.k_proj.weight, bias=None)
        v = F.linear(input=x, weight=self.attn.v_proj.weight, bias=self.attn.v_bias)

        q = q.reshape(B, N, self.attn.num_heads, -1).permute(0, 2, 1, 3)     # B, num_heads, N, C
        k = k.reshape(B, N, self.attn.num_heads, -1).permute(0, 2, 1, 3)  
        v = v.reshape(B, N, self.attn.num_heads, -1).permute(0, 2, 1, 3) 

        ## rope
        q = self.attn.rope(q).type_as(v)
        k = self.attn.rope(k).type_as(v)

        k = torch.unsqueeze(k, dim=2)
        v = torch.unsqueeze(v, dim=2)
        kv = torch.cat((k,v), 2)

        x = self.attn(q, kv)
        # x = self.attn.proj(x)
        x = x.view(B, H, W, C)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (ori_H, ori_W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.use_residual_block:
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return x


@BACKBONES.register_module()
class ViT(BaseModule):
    """
    This module implements Vision Transformer (ViT) backbone in :paper:`vitdet`.
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size=(640, 1600),
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4*2/3,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=False,
        rope=True,
        pt_hw_seq_len=16,
        intp_freq=True,
        window_size=0,
        global_window_size=0,
        window_block_indexes=(),
        residual_block_indexes=(),
        use_act_checkpoint=False,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        out_feature="last_feat",
        with_cp = False,
        xattn=False, # TODO: xformers0.0.13版本有bug，不给q的dim大于3，0.0.16版本无法用
        # pretrained=None,
    ):
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            window_block_indexes (list): Indexes for blocks using window attention.
            residual_block_indexes (list): Indexes for blocks using conv propagation.
            use_act_checkpoint (bool): If True, use activation checkpointing.
            pretrain_img_size (int): input image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretrainig models use class token.
            out_feature (str): name of the feature from the last block.
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token
        # 这里不加载预训练参数，预训练参数在包含VIT的PyramidForVIT中加载（default不单独使用VIT）
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        half_head_dim = embed_dim // num_heads // 2
        h_seq_len = img_size[0] // patch_size
        w_seq_len = img_size[1] // patch_size
        self.rope_win = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=pt_hw_seq_len,
            ft_seq_len=window_size if intp_freq else None,
        )
        self.rope_glb = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=pt_hw_seq_len,
            ft_seq_len=global_window_size if intp_freq else None,
        )

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                window_size=window_size if i in window_block_indexes else global_window_size, # use a bigger window size for global attention
                use_residual_block=i in residual_block_indexes,
                rope=self.rope_win if i in window_block_indexes else self.rope_glb,
                xattn=xattn
            )
            if use_act_checkpoint:
                # TODO: use torch.utils.checkpoint
                from fairscale.nn.checkpoint import checkpoint_wrapper

                block = checkpoint_wrapper(block)
            self.blocks.append(block)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.with_cp = with_cp
        # self.pretrained = pretrained
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    # def init_weights(self):
    #     """Initialize the weights in backbone."""

    #     def _init_weights(m):
    #         if isinstance(m, nn.Linear):
    #             trunc_normal_(m.weight, std=.02)
    #             if isinstance(m, nn.Linear) and m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.LayerNorm):
    #             nn.init.constant_(m.bias, 0)
    #             nn.init.constant_(m.weight, 1.0)

    #     if isinstance(self.pretrained, str):
    #         self.apply(_init_weights)
    #         logger = get_root_logger()
    #         ckpt = _load_checkpoint(
    #             self.pretrained, logger=logger, map_location='cpu')
    #         if 'state_dict' in ckpt:
    #             state_dict = ckpt['state_dict']
    #         elif 'model' in ckpt:
    #             state_dict = ckpt['model']
    #         else:
    #             state_dict = ckpt
    #         # load state_dict
    #         self.load_state_dict(state_dict, False)
    #     elif self.pretrained is None:
    #         self.apply(_init_weights)
    #     else:
    #         raise TypeError('pretrained must be a str or None')

    def output_shape(self): # detetron specified
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
    
    def forward(self, x):
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(self.patch_embed, x)
        else:
            x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )

        for blk in self.blocks:
            if self.with_cp and x.requires_grad:
                x = cp.checkpoint(blk, x)
            else:
                x = blk(x)

        # 如果要直接用VIT 这里要修改，从dict改成list，default不单独使用VIT
        # ret = [] # change dict to list
        # for key in outputs.keys():
        #     ret.append(outputs[key])
        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}
        return outputs


@BACKBONES.register_module()
class SimpleFeaturePyramidForViT(BaseModule):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        net_config,
        in_feature,
        out_channels,
        scale_factors,
        top_block=None,
        norm="LN",
        square_pad=0,
        checkpoint=None,
        out_layers='p4',
        with_cp=False,
    ):
        """
        Args:
            net (BaseModule): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            in_feature (str): names of the input feature maps coming
                from the net.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            norm (str): the normalization to use.
            square_pad (int): If > 0, require input images to be padded to specific square size.
        """
        super(SimpleFeaturePyramidForViT, self).__init__()
        assert(net_config['type'] == 'ViT')
        net = ViT(
            img_size=net_config['img_size'],
            patch_size=net_config['patch_size'],
            window_size=net_config['window_size'],
            global_window_size=net_config['global_window_size'],
            embed_dim=net_config['embed_dim'],
            depth=net_config['depth'],
            num_heads=net_config['num_heads'],
            mlp_ratio=net_config['mlp_ratio'],
            use_act_checkpoint=net_config['use_act_checkpoint'],
            drop_path_rate=net_config['drop_path_rate'],
            qkv_bias=net_config['qkv_bias'],
            residual_block_indexes=net_config['residual_block_indexes'],
            use_rel_pos=net_config['use_rel_pos'], 
            out_feature=net_config['out_feature'],
            window_block_indexes=net_config['window_block_indexes'],
            with_cp = with_cp
        )
        self.net = net
        self.scale_factors = scale_factors

        input_shapes = net.output_shape()
        strides = [int(input_shapes[in_feature].stride / scale) for scale in scale_factors]
        _assert_strides_are_log2_contiguous(strides)

        dim = input_shapes[in_feature].channels
        self.stages = []
        use_bias = norm == ""
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(norm, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                    Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

        
        self.in_feature = in_feature
        if top_block['type'] == 'LastLevelMaxPool':
            self.top_block = LastLevelMaxPool()
        else:
            assert False, TypeError
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad
        self.out_layers = out_layers

        if checkpoint:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f)
                self.load_state_dict(state_dict, strict=False)
        
        self.with_cp = with_cp
        # Attention！！！！
        # 由于我们只用了res['p4']和res['p2']，因此下面的module都没有用到
        # 需要把他们freeze住  不然会报错
        # 不能用find_unused_parameters = True, 不然也会报错
        module_name = ['simfp_2', 'simfp_3', 'simfp_5']
        for name in module_name:
            m = getattr(self, name)
            for p in m.parameters():
                p.requires_grad = False

    @property
    def padding_constraints(self):
        return {
            "size_divisiblity": self._size_divisibility,
            "square_size": self._square_pad,
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.net(x)
        features = bottom_up_features[self.in_feature]
        results = []

        for stage in self.stages:
            if self.with_cp and x.requires_grad:
                results.append(cp.checkpoint(stage, features))
            else:
                results.append(stage(features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            if self.with_cp and x.requires_grad:
                results.extend(cp.checkpoint(self.top_block, top_block_in_feature))
            else:
                results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        res =  {f: res for f, res in zip(self._out_features, results)}
        if isinstance(self.out_layers, str):
            return res[self.out_layers]
        elif isinstance(self.out_layers, list):
            return [res[i] for i in self.out_layers]
        else:
            raise TypeError("SimpleFeaturePyramidForViT(out_layers) can not be" + str(self.out_layers))



