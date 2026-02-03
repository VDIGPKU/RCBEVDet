from typing import Optional
import torch.nn as nn
import torch
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.transformer import FFN
from mmcv.utils import build_from_cfg
from mmcv.cnn import xavier_init, constant_init, Scale, bias_init_with_prob
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
import numpy as np
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
)
from mmcv.runner import auto_fp16
@PLUGIN_LAYERS.register_module()
class SparseBox3DRefinementModule(BaseModule):
    def __init__(
            self,
            embed_dims: int = 256,
            output_dim: int = 8,
            num_cls: int = 26,
            normalize_yaw=False,
            with_cls_branch=True,
            ):
        super(SparseBox3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.output_dim = output_dim
        self.num_cls = num_cls
        self.normalize_yaw = normalize_yaw
        self.layers = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.embed_dims),

            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.embed_dims),

            nn.Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim)
        )

        self.with_cls_branch = with_cls_branch
        if with_cls_branch:
            self.cls_layers = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(self.embed_dims),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(self.embed_dims),
                nn.Linear(self.embed_dims, self.num_cls)
            )
    
    def init_weight(self):
        if self.with_cls_branch:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    def forward(self, instance_feature, reference_points, anchor_embed):
        output = self.layers(instance_feature + anchor_embed)
        output[..., 0:6] = output[..., 0:6] + reference_points[..., 0:6]
        if self.normalize_yaw:
            output[..., 6:8] = torch.nn.functional.normalize(output[..., 6:8], dim=-1)
        return output
    
    def cls_forward(self, instance_feature):
        assert self.with_cls_branch, "Without classification layers !!!"
        return self.cls_layers(instance_feature)
    
@PLUGIN_LAYERS.register_module()
class DepthReweightModule(BaseModule):
    def __init__(
            self,
            embed_dims=256,
            min_depth=1.0,
            max_depth=150.0,
            depth_interval=5,
            ffn_layers=2,
            ):
        super(DepthReweightModule, self).__init__()
        self.embed_dims = embed_dims
        self.min_depth = min_depth
        self.depth_interval = depth_interval
        self.depth = np.arange(min_depth, max_depth + 1e-5, depth_interval)
        self.max_depth = max(self.depth)

        layers = []
        for i in range(ffn_layers):
            layers.append(
                FFN(
                embed_dims=embed_dims,
                feedforward_channels=embed_dims,
                num_fcs=2,
                act_cfg=dict(type="ReLU", inplace=True),
                dropout=0.0,
                add_residual=True,
                )
            )
        layers.append(nn.Linear(embed_dims, embed_dims))
        self.depth_fc = nn.Sequential(*layers)

    def forward(self, features, points_3d, output_conf=False):
        reference_depths = torch.norm(points_3d[..., :2], dim=-1, p=2, keepdim=True)
        reference_depths = torch.clip(
            reference_depths,
            max=self.max_depth - 1e-5,
            min=self.min_depth + 1e-5,
        )
        weights = (1 - torch.abs(reference_depths - points_3d.new_tensor(self.depth)) / self.depth_interval)

        top2 = weights.topk(2, dim=-1)[0]
        weights = torch.where(weights >= top2[..., 1:2], weights, weights.new_tensor(0.0))
        scale = torch.pow(top2[..., 0:1], 2) + torch.pow(top2[..., 1:2], 2)
        confidence = self.depth_fc(features).softmax(dim=-1)
        confidence = torch.sum(weights * confidence, dim=-1, keepdim=True)
        confidence = confidence / scale

        if output_conf:
            return confidence
        return confidence * features

@PLUGIN_LAYERS.register_module()
class SparseBox3DKeyPointsGenerator(BaseModule):
    def __init__(
            self,
            embed_dims=256,
            num_learnable_pts=6,
            fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
            ):
        super(SparseBox3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.fix_scale = fix_scale
        self.num_learnable_points = num_learnable_pts
        self.num_pts = len(self.fix_scale) + self.num_learnable_points
        if self.num_learnable_points > 0:
            self.learnable_fc = nn.Linear(self.embed_dims, self.num_learnable_points * 3)

    def init_weight(self):
        if self.num_learnable_points > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(self, anchor, instance_feature):
        bs, num_anchor = anchor.shape[:2]
        fix_scale = anchor.new_tensor(self.fix_scale)
        scale = fix_scale[None, None].repeat(bs, num_anchor, 1, 1)
        if self.num_learnable_points > 0:
            learnable_scale = self.learnable_fc(instance_feature).reshape(bs, num_anchor, self.num_learnable_points, 3).sigmoid() - 0.5
            scale = torch.cat([scale, learnable_scale], dim=2)
        key_points = scale * anchor[..., None, 3:6].exp()
        rotation_mat = anchor.new_zeros([bs, num_anchor, 3, 3])
        rotation_mat[:, :, 0, 0] = anchor[:, :, 7]
        rotation_mat[:, :, 0, 1] = -anchor[:, :, 6]
        rotation_mat[:, :, 1, 0] = anchor[:, :, 6]
        rotation_mat[:, :, 1, 1] = anchor[:, :, 7]
        rotation_mat[:, :, 2, 2] = 1
        key_points = torch.matmul(rotation_mat[:, :, None], key_points[..., None]).squeeze(-1)
        key_points = key_points + anchor[..., None, :3]

        return key_points

@ATTENTION.register_module()
class DeformableFeatureAggregation(BaseModule):
    def __init__(
            self,
            embed_dims=256,
            num_groups=8,
            num_levels=4,
            num_cams=6,
            dropout=0.1,
            kps_generator=None,
            ):
        super(DeformableFeatureAggregation, self).__init__()
        self.embed_dims = embed_dims
        self.num_groups = num_groups
        self.group_dims = (self.embed_dims // self.num_groups)
        self.num_levels = num_levels
        self.num_cams = num_cams
        self.kps_generator = build_from_cfg(kps_generator, PLUGIN_LAYERS)
        self.num_pts = self.kps_generator.num_pts
        self.weights_fc = nn.Linear(self.embed_dims, self.num_groups * self.num_cams * self.num_levels * self.num_pts)
        self.output_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.drop = nn.Dropout(dropout)
        # self.fp16_enabled = True

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)


    # @auto_fp16(out_fp32=True)
    def forward(self, instance_feature, reference_points, anchor_embed, feature_maps, img_metas, lidar2img_mat, depth_module=None):
        key_points = self.kps_generator(reference_points, instance_feature)

        weights = self._get_weights(instance_feature, anchor_embed)

        features = self.feature_sampling(feature_maps, key_points, lidar2img_mat, img_metas)

        features = self.multi_view_level_fusion(features, weights)

        if depth_module is not None:
            features = depth_module(features, reference_points[:, :, None])
        
        features = features.sum(dim=2)
        output = self.output_proj(features)
        output = self.drop(output) + instance_feature
        return output

    def _get_weights(self, instance_feature, anchor_embed):
        bs, num_anchor = instance_feature.shape[:2]
        weights = (self.weights_fc(instance_feature + anchor_embed)).reshape(bs, num_anchor, -1, self.num_groups).softmax(dim=-2)
        return weights.reshape(bs, num_anchor, self.num_cams, self.num_levels, self.num_pts, self.num_groups, 1)
    
    def multi_view_level_fusion(self, features, weights):
        features = weights * features.reshape(features.shape[:-1] + (self.num_groups, self.group_dims))
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(features.shape[:-2] + (self.embed_dims,))
        return features

    @staticmethod
    def feature_sampling(feature_maps, key_points, lidar2img_mat, img_metas):
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extand = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)
        points_2d = torch.matmul(lidar2img_mat[:, :, None, None], pts_extand[:, None, ..., None]).squeeze(-1)
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        points_2d[..., 0:1] = points_2d[..., 0:1] / img_metas[0]['pad_shape'][0][1]
        points_2d[..., 1:2] = points_2d[..., 1:2] / img_metas[0]['pad_shape'][0][0]

        points_2d = points_2d * 2 - 1 
        points_2d = points_2d.flatten(end_dim=1)

        feature = []
        for fm in feature_maps:
            feature.append(nn.functional.grid_sample(fm.flatten(end_dim=1), points_2d))
        feature = torch.stack(feature, dim=1)
        feature = feature.reshape(bs, num_cams, num_levels, -1, num_anchor, num_pts).permute(0, 4, 1, 2, 5, 3)

        return feature

@ATTENTION.register_module()
class DeformableFeatureAggregationCuda(BaseModule):
    def __init__(
            self,
            embed_dims=256,
            num_groups=8,
            num_levels=4,
            num_cams=6,
            dropout=0.1,
            im2col_step=64,
            kps_generator=None,
            ):
        super(DeformableFeatureAggregationCuda, self).__init__()
        self.embed_dims = embed_dims
        self.num_groups = num_groups
        self.group_dims = (self.embed_dims // self.num_groups)
        self.num_levels = num_levels
        self.num_cams = num_cams
        self.kps_generator = build_from_cfg(kps_generator, PLUGIN_LAYERS)
        self.num_pts = self.kps_generator.num_pts
        self.weights_fc = nn.Linear(self.embed_dims, self.num_groups * self.num_cams * self.num_levels * self.num_pts)
        self.output_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.drop = nn.Dropout(dropout)
        self.im2col_step = im2col_step
        # self.fp16_enabled = True

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    # @auto_fp16(out_fp32=True)
    def forward(self, instance_feature, reference_points, anchor_embed, feat_flatten, spatial_flatten, level_start_index, img_metas, lidar2img_mat):
        key_points = self.kps_generator(reference_points, instance_feature)

        weights = self._get_weights(instance_feature, anchor_embed)

        features = self.feature_sampling(feat_flatten, spatial_flatten, level_start_index, key_points, weights, lidar2img_mat, img_metas)

        output = self.output_proj(features)
        output = self.drop(output) + instance_feature
        return output

    def _get_weights(self, instance_feature, anchor_embed):
        bs, num_anchor = instance_feature.shape[:2]
        weights = self.weights_fc(instance_feature + anchor_embed).reshape(bs, num_anchor, -1, self.num_groups).softmax(dim=-2)
        weights = weights.reshape(bs, num_anchor, self.num_cams, -1, self.num_groups).permute(0, 2, 1, 4, 3).contiguous()
        return weights.flatten(end_dim=1)
    
    def multi_view_level_fusion(self, features, weights):
        features = weights * features.reshape(features.shape[:-1] + (self.num_groups, self.group_dims))
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(features.shape[:-2] + (self.embed_dims,))
        return features

    def feature_sampling(self, feat_flatten, spatial_flatten, level_start_index, key_points, weights, lidar2img_mat, img_metas):
        bs, num_anchor, _ = key_points.shape[:3]

        pts_extand = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)
        points_2d = torch.matmul(lidar2img_mat[:, :, None, None], pts_extand[:, None, ..., None]).squeeze(-1)
        # eps = 1e-5
        # mask = (points_2d[..., 2:3] > eps)
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        points_2d[..., 0:1] = points_2d[..., 0:1] / img_metas[0]['pad_shape'][0][1]
        points_2d[..., 1:2] = points_2d[..., 1:2] / img_metas[0]['pad_shape'][0][0]
        # mask = (mask & (points_2d[..., 0:1] > 0.0)
        #         & (points_2d[..., 0:1] < 1.0)
        #         & (points_2d[..., 1:2] > 0.0)
        #         & (points_2d[..., 1:2] < 1.0))
        # nan_mask = torch.isnan(mask)
        # mask[nan_mask] = 0.0

        points_2d = points_2d.flatten(end_dim=1) #[b*7, 900, 13, 2]
        points_2d = points_2d[:, :, None, None, :, :].repeat(1, 1, self.num_groups, self.num_levels, 1, 1)

        bn, num_value, _ = feat_flatten.size()
        feat_flatten = feat_flatten.reshape(bn, num_value, self.num_groups, -1)
        # attention_weights = weights * mask
        output = MultiScaleDeformableAttnFunction.apply(
                feat_flatten, spatial_flatten, level_start_index, points_2d,
                weights, self.im2col_step)
        
        output = output.reshape(bs, self.num_cams, num_anchor, -1)

        return output.sum(1)

@POSITIONAL_ENCODING.register_module()
class SparseBox3DEncoder(BaseModule):
    def __init__(self, embed_dims=256, vel_dims=0):
        super().__init__()
        self.embed_dims = embed_dims
        self.vel_dim = vel_dims
        def embedding_layer(input_dims):
            return nn.Sequential(
                nn.Linear(input_dims, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(self.embed_dims),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(self.embed_dims)
            )
        self.pos_fc = embedding_layer(3)
        self.size_fc = embedding_layer(3)
        self.yaw_fc = embedding_layer(2)
        if self.vel_dim > 0:
            self.vel_fc = embedding_layer(self.vel_dim)
        self.output_fc = embedding_layer(self.embed_dims)
    def forward(self, box_3d):
        pos_feat = self.pos_fc(box_3d[..., :3])
        size_feat = self.size_fc(box_3d[..., 3:6])
        yaw_feat = self.yaw_fc(box_3d[..., 6:8])
        output = pos_feat + size_feat + yaw_feat
        if self.vel_dim > 0:
            vel_feat = self.vel_fc(box_3d[..., 8:8+self.vel_dim])
            output = output + vel_feat
        output = self.output_fc(output)
        return output
    
def get_global_pos(points, pc_range):
    points = points * (pc_range[3:6] - pc_range[0:3]) + pc_range[0:3]
    return points

