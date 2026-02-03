# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

from torch import nn as nn

from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
from mmdet.models.backbones.resnet import ResNet
from mmcv.ops import Voxelization
from mmcv.cnn import ConvModule, xavier_init, normal_init
from torch.nn.init import xavier_uniform_, constant_
from einops import rearrange, repeat
from ..model_utils.ops.modules.ms_deform_attn import MSDeformAttn, LearnedPositionalEncoding3D, SinePositionalEncoding3D
import math
from mmcv.runner import BaseModule, auto_fp16

import numpy as np
import cv2
import time

from copy import deepcopy
import mmcv

class RadarConvFuser(BaseModule):
    def __init__(self, in_channels: int, out_channels: int, deconv_blocks: int) -> None:
        super(RadarConvFuser, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        deconv = []
        deconv_in = [sum(in_channels) + out_channels]
        deconv_out = [out_channels]
        for i in range(deconv_blocks - 1):
            deconv_in.append(out_channels)
            deconv_out.append(out_channels)
        for i in range(deconv_blocks):
            deconv.append(nn.Sequential(
                nn.Conv2d(deconv_in[i], deconv_out[i], 3, padding=1, bias=False),
                nn.BatchNorm2d(deconv_out[i]),
                nn.ReLU(True))
            )
        self.deconv = nn.ModuleList(deconv)

    def init_weights(self):
        super().init_weights()
        normal_init(self.fuse_conv, mean=0, std=0.001)
        for i in enumerate(self.deconv):
            normal_init(i, mean=0, std=0.001)

    def forward(self, input1, input2) -> torch.Tensor:
        res = torch.cat((input1, input2), dim=1)
        res2 = res.clone()
        out = self.fuse_conv(res)
        out = torch.cat([out, res2], dim=1)
        for layer in self.deconv:
            out = layer(out)
        return out
    
@DETECTORS.register_module()
class BEVDet_RC(CenterPoint):
    r"""BEVDet paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2112.11790>`_

    Args:
        img_view_transformer (dict): Configuration dict of view transformer.
        img_bev_encoder_backbone (dict): Configuration dict of the BEV encoder
            backbone.
        img_bev_encoder_neck (dict): Configuration dict of the BEV encoder neck.
        imc (int): channel dimension of camera BEV feature.
        lic (int): channel dimension of radar BEV feature.
    """

    def __init__(
                self,
                img_view_transformer, 
                img_bev_encoder_backbone,
                img_bev_encoder_neck,
                radar_voxel_layer=None,
                pts_pillar_layer=None,
                radar_voxel_encoder=None,
                pts_voxel_encoder=None,
                radar_middle_encoder=None,
                radar_bev_backbone=None,
                radar_bev_neck=None,
                imgpts_neck=None,
                imc=256, rac=64, #im ra 特征维度
                freeze_img=False,
                freeze_radar=False,
                bev_size=128,
                **kwargs):
        super(BEVDet_RC, self).__init__(**kwargs)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)
        #new
        if radar_voxel_layer!=None:
            self.radar_voxel_layer = Voxelization(**radar_voxel_layer)
        if pts_pillar_layer!=None:
            self.pts_pillar_layer = Voxelization(**pts_pillar_layer)
        if pts_voxel_encoder is not None:
            self.pts_voxel_encoder = builder.build_voxel_encoder(pts_voxel_encoder)
        if radar_voxel_encoder!=None:
            self.radar_voxel_encoder = builder.build_voxel_encoder(radar_voxel_encoder)
        if radar_middle_encoder!=None:
            self.radar_middle_encoder = builder.build_middle_encoder(radar_middle_encoder)
        if radar_bev_backbone is not None:
            self.radar_bev_backbone = builder.build_backbone(radar_bev_backbone)
        if radar_bev_neck is not None:
            self.radar_bev_neck = builder.build_neck(radar_bev_neck)
        if imgpts_neck is not None:
            self.imgpts_neck = builder.build_neck(imgpts_neck)
        
        self.bev_size = bev_size

        self.DeformAttn1 = MSDeformAttn( d_model=256, n_levels=1, n_heads=8, n_points=8) #d_model=256, n_levels=1, n_heads=8, n_points=4
        self.DeformAttn2 = MSDeformAttn( d_model=256, n_levels=1, n_heads=8, n_points=8) #d_model=256, n_levels=1, n_heads=8, n_points=4
        patch_row = bev_size #// 4 pretrain_img_size[0] patch_size
        patch_col = bev_size #// 4
        num_patches = patch_row * patch_col
        self.LearnedPositionalEncoding1=LearnedPositionalEncoding3D(num_feats=imc//2, row_num_embed=bev_size, col_num_embed=bev_size)
        self.LearnedPositionalEncoding2=LearnedPositionalEncoding3D(num_feats=imc//2, row_num_embed=bev_size, col_num_embed=bev_size)
            
        self.radar_reduc_conv = ConvModule(
                rac,
                imc,  #rac change imc
                kernel_size=3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False) 
        self.RadarConvFuser_fuse = RadarConvFuser(in_channels = (imc,imc),out_channels = imc,deconv_blocks = 3)

        self.freeze_img=freeze_img
        self.freeze_radar=freeze_radar

    def image_encoder(self, img, stereo=False):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        # print("imgs!!", imgs.shape)
        x = self.img_backbone(imgs)
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        # if stereo:
        #     print("stereo_feat!!", stereo_feat.shape)
        # print("x!!!!!!", x.shape)
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat
    

    @torch.no_grad()
    @force_fp32()
    def radar_voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            # print(res.shape)
            res_voxels, res_coors, res_num_points = self.radar_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch



    @force_fp32()
    def bev_encoder(self, x):
        if hasattr(self, 'img_down2top_encoder_backbone'):
            x = self.img_down2top_encoder_backbone(x)
        x = self.img_bev_encoder_backbone(x)
        
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]

    def extract_img_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""
        img = self.prepare_inputs(img)
        x, _ = self.image_encoder(img[0])
        x, depth = self.img_view_transformer([x] + img[1:7])
        # print(x.shape)
        x = self.bev_encoder(x)
        # print(x.shape)
        return [x], depth
    
    

    def extract_radar_feat(self, radar, img_metas):
        """Extract features of points."""

        voxels, num_points, coors = self.radar_voxelize(radar)
        voxel_features = self.radar_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.radar_middle_encoder(voxel_features, coors, batch_size)

        if hasattr(self, 'radar_bev_backbone') and self.radar_bev_backbone is not None:
            x = self.radar_bev_backbone(x) # 8, 64, h/2, w/2
        
        if hasattr(self, 'radar_bev_neck') and self.radar_bev_neck is not None:
            x = self.radar_bev_neck(x) # 8, 64, h/4, w/4
            x = x[0]

        return [x]
    
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    
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

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
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
        
    def extract_feat(self, points, img, img_metas, radar, gt_bboxes_3d = None , **kwargs):  #add gt
        """Extract features from images and points."""

        img_feats, depth = self.extract_img_feat(img, img_metas, **kwargs) 
        pts_feats = None
        radar_feats = self.extract_radar_feat(radar, img_metas) #new

        fusion_feats = []
        bev_height = img_feats[0].shape[2]
        bev_width = img_feats[0].shape[3]

        for i in range(0,len(img_feats)):

            if hasattr(self, 'DeformAttn1') and self.DeformAttn1 is not None:
                radar_feats[i] = self.radar_reduc_conv(radar_feats[i])
                radar_feats = rearrange(radar_feats[i], 'b c h w -> b (h w) c')
                img_feats = rearrange(img_feats[i], 'b c h w -> b (h w) c')
                
                device = torch.device("cuda")  # Get the CUDA device
                mask = torch.zeros(1, 1, self.bev_size, self.bev_size).to(device)
                pos1 = self.LearnedPositionalEncoding1(mask)
                pos2 = self.LearnedPositionalEncoding2(mask)

                reference_point1=self.get_reference_points(self.bev_size,self.bev_size)
                reference_point2=self.get_reference_points(self.bev_size,self.bev_size)

                fusion_f1 = self.DeformAttn1(query=self.with_pos_embed(radar_feats, pos1), 
                                             reference_points = reference_point1, 
                                             input_flatten = self.with_pos_embed(img_feats, pos2), 
                                             input_spatial_shapes=torch.tensor([(self.bev_size, self.bev_size)]).to(device), 
                                             input_level_start_index=torch.tensor([0, self.bev_size*self.bev_size]).to(device), 
                                             input_padding_mask=None)
                fusion_f2 = self.DeformAttn2(query=self.with_pos_embed(img_feats, pos2), 
                                             reference_points = reference_point2, 
                                             input_flatten = self.with_pos_embed(radar_feats, pos1), 
                                             input_spatial_shapes=torch.tensor([(self.bev_size, self.bev_size)]).to(device), 
                                             input_level_start_index=torch.tensor([0, self.bev_size*self.bev_size]).to(device) , 
                                             input_padding_mask=None)
                fusion_f1 = rearrange(fusion_f1, 'b (h w) c -> b c h w', h=self.bev_size, w=self.bev_size) 
                fusion_f2 = rearrange(fusion_f2, 'b (h w) c -> b c h w', h=self.bev_size, w=self.bev_size)
                fusion_f = self.RadarConvFuser_fuse(fusion_f1,fusion_f2)
                """
                :param query                       (N, Length_{query}, C)
                :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                                or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
                :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
                :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
                :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
                :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

                :return output                     (N, Length_{query}, C)
                """
            fusion_feats.append(fusion_f) #cat  

        return fusion_feats, pts_feats, depth

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_masks_bev=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats,gt_bboxes_3d,
                                            gt_labels_3d, gt_masks_bev,
                                            img_metas, gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     gt_masks_bev=None,
                     radar=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0],radar[0], img_inputs[0], gt_masks_bev, **kwargs)
        else:
            return self.aug_test(None, img_metas, img_inputs[0], **kwargs)

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_masks_bev,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_masks_bev (list[torch.Tensor]): Ground truth labels for bev segmentation
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        loss_dict = dict()
        if self.pts_bbox_head:
            outs = self.pts_bbox_head(pts_feats)
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.pts_bbox_head.loss(*loss_inputs)
            loss_dict.update(losses)
        if self.pts_seg_head:
            losses = self.pts_seg_head(pts_feats, gt_masks_bev)
            loss_dict.update(losses)
        # print(loss_dict)
        return loss_dict

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        assert False

    def simple_test(self,
                    points,
                    img_metas,
                    radar,
                    img=None,
                    gt_masks_bev=None,
                    rescale=False,
                    return_feat=False,
                    **kwargs):
        """Test function without augmentaiton."""
        if return_feat:
            return self.extract_img_feat(img=img, pred_prev=True, img_metas=None)
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas,radar=radar, **kwargs) 
        bbox_list = [dict() for _ in range(len(img_metas))]

        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale) 
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        img_feats, _, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(img_feats)
        return outs



@DETECTORS.register_module()
class BEVDet4D_RC(BEVDet_RC):
    r"""BEVDet4D paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2203.17054>`_

    Args:
        pre_process (dict | None): Configuration dict of BEV pre-process net.
        align_after_view_transfromation (bool): Whether to align the BEV
            Feature after view transformation. By default, the BEV feature of
            the previous frame is aligned during the view transformation.
        num_adj (int): Number of adjacent frames.
        with_prev (bool): Whether to set the BEV feature of previous frame as
            all zero. By default, False.
    """
    def __init__(self,
                 pre_process=None,
                 align_after_view_transfromation=False,
                 num_adj=1,
                 with_prev=True,
                 **kwargs):
        super(BEVDet4D_RC, self).__init__(**kwargs)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)
        self.align_after_view_transfromation = align_after_view_transfromation
        self.num_frame = num_adj + 1

        self.with_prev = with_prev
        self.grid = None

    def init_weights(self):
        """Initialize model weights."""
        super(BEVDet4D_RC, self).init_weights()
        if self.freeze_img:
            
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False
            
            
            for name, param in self.named_parameters():
                if 'img_view_transformer' in name:
                    param.requires_grad = False
                if 'img_bev_encoder_backbone' in name:
                    param.requires_grad = False
                if 'img_bev_encoder_neck' in name:
                    param.requires_grad = False
                if 'pre_process' in name:
                    param.requires_grad = False
            def fix_bn(m):
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False
            self.img_view_transformer.apply(fix_bn)
            self.img_bev_encoder_backbone.apply(fix_bn)
            self.img_bev_encoder_neck.apply(fix_bn)
            
            self.img_backbone.apply(fix_bn)
            self.img_neck.apply(fix_bn)

            self.pre_process_net.apply(fix_bn)

        if self.freeze_radar:
            True

    def gen_grid(self, input, sensor2keyegos, bda, bda_adj=None):
        n, c, h, w = input.shape
        _, v, _, _ = sensor2keyegos[0].shape
        if self.grid is None:
            # generate grid
            xs = torch.linspace(
                0, w - 1, w, dtype=input.dtype,
                device=input.device).view(1, w).expand(h, w)
            ys = torch.linspace(
                0, h - 1, h, dtype=input.dtype,
                device=input.device).view(h, 1).expand(h, w)
            grid = torch.stack((xs, ys, torch.ones_like(xs)), -1)
            self.grid = grid
        else:
            grid = self.grid
        grid = grid.view(1, h, w, 3).expand(n, h, w, 3).view(n, h, w, 3, 1)

        # get transformation from current ego frame to adjacent ego frame
        # transformation from current camera frame to current ego frame
        c02l0 = sensor2keyegos[0][:, 0:1, :, :]

        # transformation from adjacent camera frame to current ego frame
        c12l0 = sensor2keyegos[1][:, 0:1, :, :]

        # add bev data augmentation
        bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
        bda_[:, :, :3, :3] = bda.unsqueeze(1)
        bda_[:, :, 3, 3] = 1
        c02l0 = bda_.matmul(c02l0)
        if bda_adj is not None:
            bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
            bda_[:, :, :3, :3] = bda_adj.unsqueeze(1)
            bda_[:, :, 3, 3] = 1
        c12l0 = bda_.matmul(c12l0)

        # transformation from current ego frame to adjacent ego frame
        # tmp = torch.inverse(c12l0)
        l02l1 = c02l0.matmul(torch.inverse(c12l0))[:, 0, :, :].view(
            n, 1, 1, 4, 4)
        '''
          c02l0 * inv(c12l0)
        = c02l0 * inv(l12l0 * c12l1)
        = c02l0 * inv(c12l1) * inv(l12l0)
        = l02l1 # c02l0==c12l1
        '''

        l02l1 = l02l1[:, :, :,
                      [True, True, False, True], :][:, :, :, :,
                                                    [True, True, False, True]]

        feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.grid_interval[0]
        feat2bev[1, 1] = self.img_view_transformer.grid_interval[1]
        feat2bev[0, 2] = self.img_view_transformer.grid_lower_bound[0]
        feat2bev[1, 2] = self.img_view_transformer.grid_lower_bound[1]
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1, 3, 3)
        tf = torch.inverse(feat2bev).matmul(l02l1).matmul(feat2bev)

        # transform and normalize
        grid = tf.matmul(grid)
        normalize_factor = torch.tensor([w - 1.0, h - 1.0],
                                        dtype=input.dtype,
                                        device=input.device)
        grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1,
                                                            2) * 2.0 - 1.0
        return grid

    @force_fp32()
    def shift_feature(self, input, sensor2keyegos, bda, bda_adj=None):
        grid = self.gen_grid(input, sensor2keyegos, bda, bda_adj=bda_adj)
        output = F.grid_sample(input, grid.to(input.dtype), align_corners=True)
        return output

    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran,
                         bda, mlp_input):
        x, _ = self.image_encoder(img)
        bev_feat, depth = self.img_view_transformer(
            [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input])
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth

    def extract_img_feat_sequential(self, inputs, feat_prev):
        imgs, sensor2keyegos_curr, ego2globals_curr, intrins = inputs[:4]
        sensor2keyegos_prev, _, post_rots, post_trans, bda = inputs[4:]
        bev_feat_list = []
        mlp_input = self.img_view_transformer.get_mlp_input(
            sensor2keyegos_curr[0:1, ...], ego2globals_curr[0:1, ...],
            intrins, post_rots, post_trans, bda[0:1, ...])
        inputs_curr = (imgs, sensor2keyegos_curr[0:1, ...],
                       ego2globals_curr[0:1, ...], intrins, post_rots,
                       post_trans, bda[0:1, ...], mlp_input)
        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
        bev_feat_list.append(bev_feat)

        # align the feat_prev
        _, C, H, W = feat_prev.shape
        feat_prev = self.shift_feature(feat_prev,
                               [sensor2keyegos_curr, sensor2keyegos_prev], bda)
        bev_feat_list.append(feat_prev.view(1, (self.num_frame - 1) * C, H, W))

        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth

    def prepare_inputs(self, inputs, stereo=False):
        # split the inputs into each frame
        B, N, C, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, C, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs[1:7]

        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        curr2adjsensor = None
        if stereo:
            sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
            sensor2egos_curr = \
                sensor2egos_cv[:, :self.temporal_frame, ...].double()
            ego2globals_curr = \
                ego2globals_cv[:, :self.temporal_frame, ...].double()
            sensor2egos_adj = \
                sensor2egos_cv[:, 1:self.temporal_frame + 1, ...].double()
            ego2globals_adj = \
                ego2globals_cv[:, 1:self.temporal_frame + 1, ...].double()
            curr2adjsensor = \
                torch.inverse(ego2globals_adj @ sensor2egos_adj) \
                @ ego2globals_curr @ sensor2egos_curr
            curr2adjsensor = curr2adjsensor.float()
            curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
            curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
            curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
            assert len(curr2adjsensor) == self.num_frame

        extra = [
            sensor2keyegos,
            ego2globals,
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra
        return imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda, curr2adjsensor

    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            return self.extract_img_feat_sequential(img, kwargs['feat_prev'])
        imgs, sensor2keyegos, ego2globals, intrins, \
        post_rots, post_trans, bda, _ = self.prepare_inputs(img)
        """Extract features of images."""
        bev_feat_list = []
        depth_list = []
        key_frame = True  # back propagation for key frame only
        for img, sensor2keyego, ego2global, intrin, post_rot, post_tran in zip(
                imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans):
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin, post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin, post_rot,
                               post_tran, bda, mlp_input)
                if key_frame:
                    bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
                else:
                    with torch.no_grad():
                        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            key_frame = False
        if pred_prev:
            assert self.align_after_view_transfromation
            assert sensor2keyegos[0].shape[0] == 1
            feat_prev = torch.cat(bev_feat_list[1:], dim=0)
            ego2globals_curr = \
                ego2globals[0].repeat(self.num_frame - 1, 1, 1, 1)
            sensor2keyegos_curr = \
                sensor2keyegos[0].repeat(self.num_frame - 1, 1, 1, 1)
            ego2globals_prev = torch.cat(ego2globals[1:], dim=0)
            sensor2keyegos_prev = torch.cat(sensor2keyegos[1:], dim=0)
            bda_curr = bda.repeat(self.num_frame - 1, 1, 1)
            return feat_prev, [imgs[0],
                               sensor2keyegos_curr, ego2globals_curr,
                               intrins[0],
                               sensor2keyegos_prev, ego2globals_prev,
                               post_rots[0], post_trans[0],
                               bda_curr]
        if self.align_after_view_transfromation:
            for adj_id in range(1, self.num_frame):
                bev_feat_list[adj_id] = self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0], sensor2keyegos[adj_id]], bda)
        if not hasattr(self,'img_down2top_encoder_backbone'):
            bev_feat = torch.cat(bev_feat_list, dim=1)
            x = self.bev_encoder(bev_feat)
        else:
            x = self.bev_encoder(bev_feat_list)
        return [x], depth_list[0]


@DETECTORS.register_module()
class BEVDepth4D_RC(BEVDet4D_RC):

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      radar=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_masks_bev=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs,radar=radar, gt_bboxes_3d=gt_bboxes_3d)
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, gt_masks_bev,
                                            img_metas, gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

@DETECTORS.register_module()
class BEVDepth4D_RC_d2t(BEVDepth4D_RC):
    def __init__(self,img_down2top_encoder_backbone, **kwargs):
        super(BEVDepth4D_RC_d2t, self).__init__(**kwargs)
        self.img_down2top_encoder_backbone = builder.build_backbone(img_down2top_encoder_backbone)


@DETECTORS.register_module()
class BEVStereo4D_RC(BEVDepth4D_RC):
    def __init__(self, **kwargs):
        super(BEVStereo4D_RC, self).__init__(**kwargs)
        self.extra_ref_frames = 1
        self.temporal_frame = self.num_frame
        self.num_frame += self.extra_ref_frames

    def extract_stereo_ref_feat(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        if isinstance(self.img_backbone,ResNet):
            if self.img_backbone.deep_stem:
                x = self.img_backbone.stem(x)
            else:
                x = self.img_backbone.conv1(x)
                x = self.img_backbone.norm1(x)
                x = self.img_backbone.relu(x)
            x = self.img_backbone.maxpool(x)
            for i, layer_name in enumerate(self.img_backbone.res_layers):
                res_layer = getattr(self.img_backbone, layer_name)
                x = res_layer(x)
                return x
        else:
            x = self.img_backbone.patch_embed(x)
            hw_shape = (self.img_backbone.patch_embed.DH,
                        self.img_backbone.patch_embed.DW)
            if self.img_backbone.use_abs_pos_embed:
                x = x + self.img_backbone.absolute_pos_embed
            x = self.img_backbone.drop_after_pos(x)

            for i, stage in enumerate(self.img_backbone.stages):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
                out = out.view(-1,  *out_hw_shape,
                               self.img_backbone.num_features[i])
                out = out.permute(0, 3, 1, 2).contiguous()
                return out

    def prepare_bev_feat(self, img, sensor2keyego, ego2global, intrin,
                         post_rot, post_tran, bda, mlp_input, feat_prev_iv,
                         k2s_sensor, extra_ref_frame):
        if extra_ref_frame:
            stereo_feat = self.extract_stereo_ref_feat(img)
            return None, None, stereo_feat
        x, stereo_feat = self.image_encoder(img, stereo=True)
        metas = dict(k2s_sensor=k2s_sensor,
                     intrins=intrin,
                     post_rots=post_rot,
                     post_trans=post_tran,
                     frustum=self.img_view_transformer.cv_frustum.to(x),
                     cv_downsample=4,
                     downsample=self.img_view_transformer.downsample,
                     grid_config=self.img_view_transformer.grid_config,
                     cv_feat_list=[feat_prev_iv, stereo_feat])
        bev_feat, depth = self.img_view_transformer(
            [x, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda,
             mlp_input], metas)
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth, stereo_feat

    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            # Todo
            assert False
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, curr2adjsensor = self.prepare_inputs(img, stereo=True)
        """Extract features of images."""
        bev_feat_list = []
        depth_key_frame = None
        feat_prev_iv = None
        for fid in range(self.num_frame-1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame-self.extra_ref_frames
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin,
                    post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda, mlp_input,
                               feat_prev_iv, curr2adjsensor[fid],
                               extra_ref_frame)
                if key_frame:
                    bev_feat, depth, feat_curr_iv = \
                        self.prepare_bev_feat(*inputs_curr)
                    depth_key_frame = depth
                else:
                    with torch.no_grad():
                        bev_feat, depth, feat_curr_iv = \
                            self.prepare_bev_feat(*inputs_curr)
                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)
                feat_prev_iv = feat_curr_iv
        if pred_prev:
            # Todo
            assert False
        if not self.with_prev:
            bev_feat_key = bev_feat_list[0]
            if len(bev_feat_key.shape) ==4:
                b,c,h,w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1),
                                  h, w]).to(bev_feat_key), bev_feat_key]
            else:
                b, c, z, h, w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1), z,
                                  h, w]).to(bev_feat_key), bev_feat_key]
        if self.align_after_view_transfromation:
            for adj_id in range(self.num_frame-2):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[self.num_frame-2-adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth_key_frame
