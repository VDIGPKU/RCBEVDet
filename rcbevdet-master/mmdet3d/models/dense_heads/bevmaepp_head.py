# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import BaseModule
from torch import nn

from mmdet3d.models.builder import HEADS



@HEADS.register_module()
class SGKDProjMaskNormCosHead(BaseModule):
    def __init__(self,
                 img_channels=256,
                 pts_channels=256,
                 out_channels=1024,
                #  loss_distill=dict(
                #      type='MSELoss', reduction='sum', loss_weight=2e-5),
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 init_cfg=None,
                 loss_prefix='',
                 train_cfg=None,
                 test_cfg=None,):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(SGKDProjMaskNormCosHead, self).__init__(init_cfg=init_cfg)

        # self.in_channels = in_channels

        self.loss_distill = nn.CosineSimilarity(dim=1)
        self.fp16_enabled = False

        dim = out_channels//4
        # dim = max(img_channels, pts_channels)

        self.img_head = nn.Sequential(
                    ConvModule(
                        img_channels,
                        img_channels,
                        kernel_size=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'),
                        bias=bias),
                    ConvModule(
                        img_channels,
                        img_channels,
                        kernel_size=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'),
                        bias=bias),
                    ConvModule(
                        img_channels,
                        out_channels,
                        kernel_size=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=None,
                        bias=bias,
                        ),
        )

        self.pts_head = nn.Sequential(
                    ConvModule(
                        pts_channels,
                        pts_channels,
                        kernel_size=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'),
                        bias=bias),
                    ConvModule(
                        pts_channels,
                        pts_channels,
                        kernel_size=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'),
                        bias=bias),
                    ConvModule(
                        pts_channels,
                        out_channels,
                        kernel_size=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=None,
                        bias=bias,
                        ),
        )

        self.proj = nn.Sequential(
                        ConvModule(
                            out_channels,
                            dim,
                            kernel_size=1,
                            padding=0,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=dict(type='ReLU'),
                            bias=bias),
                        build_conv_layer(
                            conv_cfg,
                            dim,
                            out_channels,
                            kernel_size=1,
                            stride=1,
                            bias=True),
        )

        # self.transform = nn.Sequential(
        #     build_conv_layer(
        #             conv_cfg,
        #             pts_channels,
        #             pts_channels,
        #             kernel_size=1,
        #             stride=1,
        #             bias=True),
        #     nn.ReLU(inplace=True),
        #     build_conv_layer(
        #             conv_cfg,
        #             pts_channels,
        #             pts_channels,
        #             kernel_size=1,
        #             stride=1,
        #             bias=True),
        # )

        # a shared convolution
        

        self.loss_prefix = loss_prefix

    def forward(self, img_feats, pts_feats, p_step, mask_tensor=None):

        img_feats = img_feats[0]
        pts_feats = pts_feats[0]
        # print(img_feats.mean(), img_feats.max(), img_feats.min())
        # print(torch.std(img_feats, dim=1).mean(), torch.std(img_feats, dim=1).min(), torch.std(img_feats, dim=1).max(), torch.std(img_feats, dim=1).std())
        img_feats = self.img_head(img_feats)
        # print(img_feats.mean(), img_feats.max(), img_feats.min())
        # print(mask_tensor)

        assert mask_tensor is not None
        # if mask_tensor is None:
        #     mask_tensor = torch.zeros_like(pts_feats).detach()
        #     # print(mask_tensor)
        #     # print(before_bev_feat)
        #     mask_tensor = (mask_tensor != pts_feats)
        #     mask_tensor = mask_tensor.detach()
            # mask_tensor = torch.zeros_like(pts_feats).detach()
            # mask_tensor = (mask_tensor == pts_feats).max(dim=1)[0]
            # mask_tensor = mask_tensor.detach()
        # print(pts_feats.mean(), pts_feats.max(), pts_feats.min())
        # print(torch.std(pts_feats, dim=1).mean(), torch.std(pts_feats, dim=1).min(), torch.std(pts_feats, dim=1).max(), torch.std(pts_feats, dim=1).std())
        pts_feats = self.pts_head(pts_feats)
        # print(pts_feats.mean(), pts_feats.max(), pts_feats.min())

        z_pts, z_img = self.proj(pts_feats), self.proj(img_feats)
        # print(z_img.mean(), z_img.max(), z_img.min())
        # print(z_pts.mean(), z_pts.max(), z_pts.min())
        # print(torch.std(z_img, dim=1).mean(), torch.std(z_img, dim=1).min(), torch.std(z_img, dim=1).max(), torch.std(z_img, dim=1).std())
        # print(torch.std(z_pts, dim=1).mean(), torch.std(z_pts, dim=1).min(), torch.std(z_pts, dim=1).max(), torch.std(z_pts, dim=1).std())

        loss_distill = dict()
        if p_step:
            N, C, H, W = z_pts.shape
            
            # z_pts = nn.functional.normalize(z_pts, dim=1)
            # img_feats = nn.functional.normalize(img_feats, dim=1)
            
            dis_loss = self.loss_distill(z_pts, img_feats.detach())
            ratio = N*H*W / mask_tensor.sum()
            # print(ratio)

            dis_loss = dis_loss * mask_tensor
            # print(dis_loss.shape, mask_tensor.shape)
            # dis_loss = dis_loss * mask_tensor
            dis_loss = dis_loss.mean() * ratio
            # print(dis_loss)

            loss_distill['loss_distill_cam'] = -dis_loss
            loss_distill['loss_distill_lidar'] = torch.zeros(1).cuda(dis_loss.device)
            # exit(0)
        else:
            # outs = self.pts_bbox_head(img_feats, pts_feats.detach())
            # losses_pts = self.pts_bbox_head.loss(*outs)
            N, C, H, W = z_pts.shape

            # z_img = nn.functional.normalize(z_img, dim=1)
            # pts_feats = nn.functional.normalize(pts_feats, dim=1)

            dis_loss = self.loss_distill(z_img, pts_feats.detach())

            dis_loss = dis_loss.mean()

            loss_distill['loss_distill_cam'] = torch.zeros(1).cuda(dis_loss.device)
            loss_distill['loss_distill_lidar'] = -dis_loss

        return loss_distill


    def loss(self, pred, target, **kwargs):
        
        N, C, H, W = pred.shape
        pred = self.proj(pred)
        dis_loss = self.loss_distill(pred, target) / N

        return dict(loss_distill=dis_loss)

    
    def norm(self, feat):
        """Normalize the feature maps to have zero mean and unit variances.
        Args:
            feat (torch.Tensor): The original feature map with shape
                (N, C, H, W).
        """
        assert len(feat.shape) == 4
        N, C, H, W = feat.shape
        feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    