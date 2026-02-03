# Copyright (c) Youngwan Lee (ETRI) All Rights Reserved.
# Copyright 2021 Toyota Research Institute.  All rights reserved.
from collections import OrderedDict
import math
import torch
from torch import nn
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init

import numpy as np
import torch.utils.checkpoint as cp
from mmdet.models import BACKBONES
from typing import Optional
from torch.nn import BatchNorm2d
from detectron2.layers import FrozenBatchNorm2d, ShapeSpec, get_norm, Conv2d

_NORM = False

VoVNet19_slim_dw_eSE = {
    'stem': [64, 64, 64],
    'stage_conv_ch': [64, 80, 96, 112],
    'stage_out_ch': [112, 256, 384, 512],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": True
}

VoVNet19_dw_eSE = {
    'stem': [64, 64, 64],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": True
}

VoVNet19_slim_eSE = {
    'stem': [64, 64, 128],
    'stage_conv_ch': [64, 80, 96, 112],
    'stage_out_ch': [112, 256, 384, 512],
    'layer_per_block': 3,
    'block_per_stage': [1, 1, 1, 1],
    'eSE': True,
    "dw": False
}

VoVNet19_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": False
}

VoVNet39_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 1, 2, 2],
    "eSE": True,
    "dw": False
}

VoVNet57_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 1, 4, 3],
    "eSE": True,
    "dw": False
}

VoVNet99_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 3, 9, 3],
    "eSE": True,
    "dw": False
}

_STAGE_SPECS = {
    "V-19-slim-dw-eSE": VoVNet19_slim_dw_eSE,
    "V-19-dw-eSE": VoVNet19_dw_eSE,
    "V-19-slim-eSE": VoVNet19_slim_eSE,
    "V-19-eSE": VoVNet19_eSE,
    "V-39-eSE": VoVNet39_eSE,
    "V-57-eSE": VoVNet57_eSE,
    "V-99-eSE": VoVNet99_eSE,
}

def dw_conv3x3(in_channels, out_channels, module_name, postfix, stride=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        (
            '{}_{}/dw_conv3x3'.format(module_name, postfix),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=out_channels,
                bias=False
            )
        ),
        (
            '{}_{}/pw_conv1x1'.format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        ),
        ('{}_{}/pw_norm'.format(module_name, postfix), get_norm(_NORM, out_channels)),
        ('{}_{}/pw_relu'.format(module_name, postfix), nn.ReLU(inplace=True)),
    ]


def conv3x3(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", get_norm(_NORM, out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


def conv1x1(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", get_norm(_NORM, out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class eSEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.hsigmoid = Hsigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x


class _OSA_module(nn.Module):
    def __init__(
        self, in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE=False, identity=False, depthwise=False
    ):

        super(_OSA_module, self).__init__()

        self.identity = identity
        self.depthwise = depthwise
        self.isReduced = False
        self.layers = nn.ModuleList()
        in_channel = in_ch
        if self.depthwise and in_channel != stage_ch:
            self.isReduced = True
            self.conv_reduction = nn.Sequential(
                OrderedDict(conv1x1(in_channel, stage_ch, "{}_reduction".format(module_name), "0"))
            )
        for i in range(layer_per_block):
            if self.depthwise:
                self.layers.append(nn.Sequential(OrderedDict(dw_conv3x3(stage_ch, stage_ch, module_name, i))))
            else:
                self.layers.append(nn.Sequential(OrderedDict(conv3x3(in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(OrderedDict(conv1x1(in_channel, concat_ch, module_name, "concat")))

        self.ese = eSEModule(concat_ch)

    def forward(self, x):

        identity_feat = x

        output = []
        output.append(x)
        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        xt = self.ese(xt)

        if self.identity:
            xt = xt + identity_feat

        return xt


class _OSA_stage(nn.Sequential):
    def __init__(
        self, in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num, SE=False, depthwise=False
    ):

        super(_OSA_stage, self).__init__()

        if not stage_num == 2:
            self.add_module("Pooling", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        if block_per_stage != 1:
            SE = False
        module_name = f"OSA{stage_num}_1"
        self.add_module(
            module_name, _OSA_module(in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, depthwise=depthwise)
        )
        for i in range(block_per_stage - 1):
            if i != block_per_stage - 2:  # last block
                SE = False
            module_name = f"OSA{stage_num}_{i + 2}"
            self.add_module(
                module_name,
                _OSA_module(
                    concat_ch,
                    stage_ch,
                    concat_ch,
                    layer_per_block,
                    module_name,
                    SE,
                    identity=True,
                    depthwise=depthwise
                ),
            )


@BACKBONES.register_module()
class VoVNet(nn.Module):
    def __init__(
            self, 
            norm,
            name, 
            input_ch, 
            out_features=None,
            with_cp=False,
    ):
        """
        Args:
            input_ch(int) : the number of input channel
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "stage2" ...
        """
        super(VoVNet, self).__init__()

        global _NORM
        _NORM = norm

        stage_specs = _STAGE_SPECS[name]

        stem_ch = stage_specs["stem"]
        config_stage_ch = stage_specs["stage_conv_ch"]
        config_concat_ch = stage_specs["stage_out_ch"]
        block_per_stage = stage_specs["block_per_stage"]
        layer_per_block = stage_specs["layer_per_block"]
        SE = stage_specs["eSE"]
        depthwise = stage_specs["dw"]

        self._out_features = out_features

        # Stem module
        conv_type = dw_conv3x3 if depthwise else conv3x3
        stem = conv3x3(input_ch, stem_ch[0], "stem", "1", 2)
        stem += conv_type(stem_ch[0], stem_ch[1], "stem", "2", 1)
        stem += conv_type(stem_ch[1], stem_ch[2], "stem", "3", 2)
        self.add_module("stem", nn.Sequential((OrderedDict(stem))))
        current_stirde = 4
        self._out_feature_strides = {"stem": current_stirde, "stage2": current_stirde}
        self._out_feature_channels = {"stem": stem_ch[2]}

        stem_out_ch = [stem_ch[2]]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]
        # OSA stages
        self.stage_names = []
        for i in range(4):  # num_stages
            name = "stage%d" % (i + 2)  # stage 2 ... stage 5
            self.stage_names.append(name)
            self.add_module(
                name,
                _OSA_stage(
                    in_ch_list[i],
                    config_stage_ch[i],
                    config_concat_ch[i],
                    block_per_stage[i],
                    layer_per_block,
                    i + 2,
                    SE,
                    depthwise,
                ),
            )

            self._out_feature_channels[name] = config_concat_ch[i]
            if not i == 0:
                self._out_feature_strides[name] = current_stirde = int(current_stirde * 2)
        self.with_cp = with_cp
        # initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return

        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "stage" + str(stage_index + 1))
            for p in m.parameters():
                p.requires_grad = False
                FrozenBatchNorm2d.convert_frozen_batchnorm(self)

    def forward(self, x):
        outputs = {}
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(self.stem, x)
        else:
            x = self.stem(x)

        if "stem" in self._out_features:
            outputs["stem"] = x

        for name in self.stage_names:
            if self.with_cp and x.requires_grad:
                x = cp.checkpoint(getattr(self, name), x)
            else:
                x = getattr(self, name)(x)
                
            if name in self._out_features:
                outputs[name] = x

        # 如果要直接用Vovnet 这里要修改，从dict改成list
        # ret = [] # change dict to list
        # for key in outputs.keys():
        #     ret.append(outputs[key])
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
            for name in self._out_features
        }


def build_vovnet_backbone(cfg, input_shape):
    """
    Create a VoVNet instance from config.

    Returns:
        VoVNet: a :class:`VoVNet` instance.
    """
    out_features = cfg.OUT_FEATURES
    return VoVNet(cfg, input_shape.channels, out_features=out_features)


def build_vovnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_vovnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = VovNetFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.FE.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.FE.FPN.FUSE_TYPE,
    )
    return backbone


def build_fcos_vovnet_fpn_backbone_p6(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_vovnet_backbone(cfg.FE.BACKBONE, input_shape)
    in_features = cfg.FE.FPN.IN_FEATURES
    out_channels = cfg.FE.FPN.OUT_CHANNELS
    in_channels_top = out_channels
    top_block = LastLevelP6(in_channels_top, out_channels, "p5")
    backbone = VovNetFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.FE.FPN.NORM,
        # top_block=LastLevelMaxPool(),
        top_block=top_block,
        fuse_type=cfg.FE.FPN.FUSE_TYPE,
    )

    backbone._size_divisibility *= 2

    return backbone



class LastLevelP6(nn.Module):
    """
    This module is used in FCOS to generate extra layers
    """
    def __init__(self, in_channels, out_channels, in_features="res5"):
        super().__init__()
        self.num_levels = 1
        self.in_feature = in_features
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        for module in [self.p6]:
            weight_init.c2_xavier_fill(module)

    def forward(self, x):
        p6 = self.p6(x)
        return [p6]


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )


@BACKBONES.register_module()
class VovNetFPN(nn.Module):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    _fuse_type: torch.jit.Final[str]

    def __init__(
        self,
        bottom_up_config,
        in_features,
        out_channels,
        out_layers='p4',
        norm="",
        top_block=None,
        fuse_type="sum",
        _size_divisibility_mul_2=False,
        square_pad=0,
        checkpoint=None,
        with_cp=False,
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
            square_pad (int): If > 0, require input images to be padded to specific square size.
        """
        super(VovNetFPN, self).__init__()
        assert(bottom_up_config['type'] == 'VovNet')
        if 'with_cp' not in bottom_up_config:
            backbone_with_cp = False
        else:
            backbone_with_cp = bottom_up_config['with_cp']
        bottom_up = VoVNet(
            norm=bottom_up_config['norm'], 
            name=bottom_up_config['name'], 
            input_ch=bottom_up_config['input_ch'],
            out_features=bottom_up_config['out_features'],
            with_cp=backbone_with_cp,
            )
        assert isinstance(bottom_up, nn.Module)
    
        assert in_features, in_features

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(strides)
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        if top_block['type'] == 'LastLevelMaxPool':
            self.top_block = LastLevelMaxPool()
        elif top_block['type'] == 'LastLevelP6':
            self.top_block = LastLevelP6(
                top_block['in_channels_top'], 
                top_block['out_channels'],
                top_block['in_features']
            )
        else:
            assert False, TypeError
    
        self.in_features = tuple(in_features)
        self.bottom_up = bottom_up
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
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

        self.out_layers = out_layers

        if _size_divisibility_mul_2:
            self._size_divisibility *= 2
        
        self.with_cp = with_cp
        self.out_layers = out_layers

        if checkpoint:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f)
                self.load_state_dict(state_dict)

    @property
    def size_divisibility(self):
        return self._size_divisibility

    @property
    def padding_constraints(self):
        return {"square_size": self._square_pad}

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """

        bottom_up_features = self.bottom_up(x)
        results = []

        if self.with_cp and bottom_up_features[self.in_features[-1]].requires_grad:
            prev_features = cp.checkpoint(self.lateral_convs[0], bottom_up_features[self.in_features[-1]])
        else:
            prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])

        if self.with_cp and prev_features.requires_grad:
            results.append(cp.checkpoint(self.output_convs[0], prev_features))
        else:
            results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                
                if self.with_cp and features.requires_grad:
                    lateral_features = cp.checkpoint(lateral_conv, features)
                else:
                    lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                
                if self.with_cp and prev_features.requires_grad:
                    results.insert(0, cp.checkpoint(output_conv, prev_features))
                else:
                    results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            
            if self.with_cp and top_block_in_feature.requires_grad:
                results.extend(cp.checkpoint(self.top_block, top_block_in_feature))
            else:
                results.extend(self.top_block(top_block_in_feature))

        assert len(self._out_features) == len(results)
        res = {f: res for f, res in zip(self._out_features, results)}
        if False: # 也可以通过上采样+concat+过linear融合特征图
            layers = ['p5', 'p4', 'p3', 'p2']
            prev_features = res['p6']
            for layer in layers:
                features = res['p5']
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode='nearest')
                features = torch.cat((features, top_down_features), dim=1)
        # TODO: 以后可以进行多尺度BEV检测
        # p2 1/4 downsample  p3 1/8  p4 1/16  p5 1/32  p6 1/64
        # 使用不同shape的res，view_transformer中的downsample要进行对应的修改
        if isinstance(self.out_layers, str):
            return res[self.out_layers]
        elif isinstance(self.out_layers, list):
            return [res[i] for i in self.out_layers]
        else:
            raise TypeError("VovNetFPN(out_layers) can not be" + str(self.out_layers))
        # return res['p4']

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


