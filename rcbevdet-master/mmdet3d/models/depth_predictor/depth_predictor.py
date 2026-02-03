import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthPredictor(nn.Module):

    def __init__(self, model_cfg):
        """
        Initialize depth predictor and depth encoder
        Args:
            model_cfg [EasyDict]: Depth classification network config
        """
        super().__init__()
        depth_num_bins = int(model_cfg["num_depth_bins"])

        # Create modules
        input_dim = model_cfg["hidden_dim"]
        d_model = 256   # default value

        '''Deprecated
        depth_min = float(model_cfg["depth_min"])
        depth_max = float(model_cfg["depth_max"])
        self.depth_max = depth_max
        bin_size = 2 * (depth_max - depth_min) / (depth_num_bins * (1 + depth_num_bins))
        bin_indice = torch.linspace(0, depth_num_bins - 1, depth_num_bins)
        bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + depth_min
        bin_value = torch.cat([bin_value, torch.tensor([depth_max])], dim=0)
        self.depth_bin_values = nn.Parameter(bin_value, requires_grad=False)

        self.downsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GroupNorm(32, d_model))
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))
        self.upsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))
        '''
        conv_layers = []
        _build_proj_layer = lambda dim0, dim1: nn.Sequential(nn.Conv2d(dim0, dim1, kernel_size=(1, 1)), nn.GroupNorm(32, dim1))
        _build_conv_layer = lambda dim0, dim1: nn.Sequential(
            nn.Conv2d(dim0, dim1, kernel_size=(3, 3), padding=1), nn.GroupNorm(32, dim1), nn.ReLU())

        self.multi_level_fusion = ('multi_level_fusion' in model_cfg and model_cfg['multi_level_fusion'])
        if self.multi_level_fusion:
            self.proj_8 = _build_proj_layer(input_dim, d_model)
            self.proj_16 = _build_proj_layer(input_dim, d_model)
            self.proj_32 = _build_proj_layer(input_dim, d_model)
            conv_layers.append(_build_conv_layer(d_model, d_model))
        else:
            conv_layers.append(_build_conv_layer(input_dim, d_model))

        self.conv_layer_num = 2-1     # default value
        if 'conv_layer_num' in model_cfg:
            self.conv_layer_num = model_cfg['conv_layer_num'] - 1
        conv_layers += [_build_conv_layer(d_model, d_model) for _ in range(self.conv_layer_num)]
        self.depth_head = nn.Sequential(*conv_layers)
        self.depth_classifier = nn.Conv2d(d_model, depth_num_bins + 1, kernel_size=(1, 1))

    def forward(self, feature):
        '''Deprecated, we donot need multilevel fusion since FPN has been employed, thus we use only one level depth such as p3/p4
        assert len(feature) == 4
        # foreground depth map
        src_16 = self.proj(feature[1])
        src_32 = self.upsample(F.interpolate(feature[2], size=src_16.shape[-2:]))
        src_8 = self.downsample(feature[0])
        src = (src_8 + src_16 + src_32) / 3
        '''
        if self.multi_level_fusion:
           # input multi-level feature
           src_8 = self.proj_8(feature[0].flatten(0, 1))
           src_16 = self.proj_16(F.interpolate(feature[1].flatten(0, 1), size=src_8.shape[-2:]))
           src_32 = self.proj_32(F.interpolate(feature[2].flatten(0, 1), size=src_8.shape[-2:]))
           src = (src_8 + src_16 + src_32) / 3
        else:
            src = feature

        src = self.depth_head(src)
        depth_logits = self.depth_classifier(src)

        # depth_probs = F.softmax(depth_logits, dim=1)
        # weighted_depth = (depth_probs * self.depth_bin_values.reshape(1, -1, 1, 1)).sum(dim=1)

        return depth_logits
