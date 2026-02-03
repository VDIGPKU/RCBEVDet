# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
from mmdet.models.backbones.resnet import ResNet
# from mmdet3d.models.backbones import VovNetFPN, SwinTransformer, SwinTransformerV1, ConvNeXt, SimpleFeaturePyramidForViT
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d

@DETECTORS.register_module()
class BEVDet(CenterPoint):
    r"""BEVDet paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2112.11790>`_

    Args:
        img_view_transformer (dict): Configuration dict of view transformer.
        img_bev_encoder_backbone (dict): Configuration dict of the BEV encoder
            backbone.
        img_bev_encoder_neck (dict): Configuration dict of the BEV encoder neck.
    """

    def __init__(self, img_view_transformer, img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None, img_bev_encoder_backbone_forseg=None,
                 img_bev_encoder_neck_forseg=None, heatmap2seg=False, with_bevencoder=True,**kwargs):
        super(BEVDet, self).__init__(**kwargs)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        if img_bev_encoder_backbone is not None:
            self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
        if img_bev_encoder_neck is not None:
            self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)
        if img_bev_encoder_backbone_forseg is not None:
            self.img_bev_encoder_backbone_forseg = builder.build_backbone(img_bev_encoder_backbone_forseg)
        if img_bev_encoder_neck_forseg is not None:
            self.img_bev_encoder_neck_forseg = builder.build_neck(img_bev_encoder_neck_forseg)
        self.with_bevencoder = with_bevencoder
        self.heatmap2seg = heatmap2seg

    def image_encoder(self, img, stereo=False):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.contiguous().view(B * N, C, imH, imW)
        # print("imgs!!", imgs.shape)
        x = self.img_backbone(imgs)
        # print("bboutput", len(x))
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
            # print("neckoutput", len(x), x[0].shape, x[1].shape, x[2].shape)
        if type(x) in [list, tuple]:
            x = x[0]
        # if stereo:
        #     print("stereo_feat!!", stereo_feat.shape)
        # print("x!!!!!!", x.shape)
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat

    @force_fp32()
    def bev_encoder(self, x):
        if hasattr(self, 'img_down2top_encoder_backbone'):
            x = self.img_down2top_encoder_backbone(x)
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    @force_fp32()
    def bev_encoder_forseg(self, x):
        if hasattr(self, 'img_down2top_encoder_backbone'):
            x = self.img_down2top_encoder_backbone(x)
        x = self.img_bev_encoder_backbone_forseg(x)
        x = self.img_bev_encoder_neck_forseg(x)
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
        keyego2global = ego2globals[:, 0, ...].unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]

    def extract_img_feat(self, img, img_metas, with_bevencoder=True, **kwargs):
        """Extract features of images."""
        img = self.prepare_inputs(img)

        x, _ = self.image_encoder(img[0])
        # debug
        if False:  # visualize segment anything
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
            import argparse
            import json
            import os
            import cv2
            from typing import Any, Dict, List
            from mmcv.image.photometric import imdenormalize
            import numpy as np

            def image_Tensor2ndarray(image_tensor: torch.Tensor):
                """
                将tensor转化为cv2格式
                """
                assert (len(image_tensor.shape) == 3)
                # 复制一份
                image_tensor = image_tensor.clone().detach()
                # 到cpu
                image_tensor = image_tensor.to(torch.device('cpu'))
                # 先从CHW转为HWC，转为numpy，然后反归一化(包括RGB转BRG), 最后转成uint8
                mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
                std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
                image_cv2 = image_tensor.permute(1, 2, 0).numpy()
                image_cv2 = imdenormalize(image_cv2, mean, std, to_bgr=False)
                image_cv2 = image_cv2.astype(np.uint8)
                return image_cv2

            def write_masks_and_images_to_folder(masks: List[Dict[str, Any]], path: str, image: np.array) -> None:
                header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
                metadata = [header]
                cv2.imwrite(os.path.join(path, 'image.png'), image)
                image = np.zeros(image.shape)  # 避免没有seg的部分直接当原图进来
                for i, mask_data in enumerate(masks):
                    mask = mask_data["segmentation"]
                    filename = f"{i}.png"
                    image[mask] = np.array(
                        [235 // len(masks) * i + 20, 235 // len(masks) * i + 20, 235 // len(masks) * i + 20])
                    cv2.imwrite(os.path.join(path, 'mask' + filename), mask * 255)
                    mask_metadata = [
                        str(i),
                        str(mask_data["area"]),
                        *[str(x) for x in mask_data["bbox"]],
                        *[str(x) for x in mask_data["point_coords"][0]],
                        str(mask_data["predicted_iou"]),
                        str(mask_data["stability_score"]),
                        *[str(x) for x in mask_data["crop_box"]],
                    ]
                    row = ",".join(mask_metadata)
                    metadata.append(row)
                cv2.imwrite(os.path.join(path, 'masked-image.png'), image)
                metadata_path = os.path.join(path, "metadata.csv")
                with open(metadata_path, "w") as f:
                    f.write("\n".join(metadata))
                return

            images = img[0][0]  # B = 1
            model_type = 'vit_b'
            checkpoint = '/home/wangxinhao/segment-anything/checkpoint/sam_vit_b_01ec64.pth'
            device = 'cuda'
            convert_to_rle = False
            output = '/home/wangxinhao/test'
            print("Loading model...")
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            _ = sam.to(device=device)
            output_mode = "coco_rle" if convert_to_rle else "binary_mask"
            amg_kwargs = {}
            generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)
            os.makedirs(output, exist_ok=True)
            new_x = []
            for i in range(images.shape[0]):
                print(f"Processing '{i}'...")
                image = image_Tensor2ndarray(images[i])  # 先转成np.array
                if image is None:
                    print(f"Could not load '{i}' as an image, skipping...")
                    continue
                masks = generator.generate(image)
                new_x.append(generator.predictor.features[0])
                # below is to show segmentation results in visualization
                # base = str(i)+'-'+img_metas[0]['sample_idx']
                # save_base = os.path.join(output, base)
                # if output_mode == "binary_mask":
                #     os.makedirs(save_base, exist_ok=True)
                #     write_masks_and_images_to_folder(masks, save_base, image)
                # else:
                #     save_file = save_base + ".json"
                #     with open(save_file, "w") as f:
                #         json.dump(masks, f)
            # below is to compare image feature
            new_x = [torch.stack(new_x)]
            x, new_x = x[0].cpu().detach(), new_x[0].cpu().detach()
            import ipdb
            ipdb.set_trace()
            assert (x.shape == new_x.shape)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    for k in range(x.shape[2]):
                        for l in range(x.shape[3]):
                            if abs(x[i][j][k][l] - new_x[i][j][k][l]) > 1e-3:  # 检测表明数据之差小于5e-4
                                print('******** DIFFERENT IN', i, j, k, l, '**********')
            print("Done!")
            exit(0)

        if False:  # directly use segment anything
            from mmcv.image.photometric import imdenormalize
            import numpy as np

            def image_Tensor2ndarray(image_tensor: torch.Tensor):
                """
                将tensor转化为cv2格式
                """
                assert (len(image_tensor.shape) == 3)
                # 复制一份
                image_tensor = image_tensor.clone().detach()
                # 到cpu
                image_tensor = image_tensor.to(torch.device('cpu'))
                # 先从CHW转为HWC，转为numpy，然后反归一化(包括RGB转BRG), 最后转成uint8
                mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
                std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
                image_cv2 = image_tensor.permute(1, 2, 0).numpy()
                image_cv2 = imdenormalize(image_cv2, mean, std, to_bgr=False)
                image_cv2 = image_cv2.astype(np.uint8)
                return image_cv2

            new_x = []
            from segment_anything import SamPredictor, sam_model_registry
            model_type = 'vit_b'
            checkpoint = '/home/wangxinhao/segment-anything/checkpoint/sam_vit_b_01ec64.pth'
            image = img[0][0]  # B = 1
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            predictor = SamPredictor(sam)
            for i in range(image.shape[0]):
                predictor.set_image(image_Tensor2ndarray(image[i]))
                new_x.append(predictor.features[0])
                predictor.reset_image()
            new_x = [torch.stack(new_x)]
            x, new_x = x[0].cpu().detach(), new_x[0].cpu().detach()
            assert (x.shape == new_x.shape)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    for k in range(x.shape[2]):
                        for l in range(x.shape[3]):
                            if abs(x[i][j][k][l] - new_x[i][j][k][l]) > 5e-4:  # 检测表明数据之差小于5e-4
                                print('******** DIFFERENT IN', i, j, k, l, '**********')
            print('\n******** END **********\n')

        if False:  # print image feature
            import matplotlib.pyplot as plt
            import numpy as np
            fig = plt.figure(figsize=(16, 16))
            print('\n******** BEGIN PRINT **********\n')
            pts_feats = x[0]
            for idx in range(pts_feats.shape[0]):
                pts_feat = pts_feats[idx].cpu().detach().numpy()  # 放到cpu上 去掉梯度 转换成numpy
                print('pts_feat.shape =', pts_feat.shape)
                feat_2d = np.zeros(pts_feat.shape[1:])
                print('feat_2d.shape =', feat_2d.shape)
                for h in range(feat_2d.shape[0]):
                    for w in range(feat_2d.shape[1]):
                        for c in range(pts_feat.shape[0]):
                            feat_2d[h][w] += abs(pts_feat[c][h][w])
                plt.imshow(feat_2d, cmap=plt.cm.gray, vmin=0, vmax=255)
                plt.savefig("/home/wangxinhao/bevperception/utils/std_feat/" + img_metas[0]['sample_idx'] + '-' + str(
                    idx) + ".png")
            print('\n******** END PRINT **********\n')

        x, depth = self.img_view_transformer([x] + img[1:7])
        if with_bevencoder:
            x = self.bev_encoder(x)
        return [x], depth

    def extract_feat(self, points, img, img_metas, with_bevencoder=True, **kwargs):
        """Extract features from images and points."""
        img_feats, depth = self.extract_img_feat(img, img_metas, with_bevencoder=with_bevencoder, **kwargs)
        pts_feats = None
        return img_feats, pts_feats, depth

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
            points, img=img_inputs, img_metas=img_metas, with_bevencoder=self.with_bevencoder, **kwargs)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, gt_masks_bev,
                                            img_metas, gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     gt_masks_bev=None,
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
            return self.simple_test(points[0], img_metas[0], img_inputs[0], gt_masks_bev, **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_masks_bev,
                          img_metas,
                          gt_bboxes_ignore=None,
                          pts_feats_forseg=None,):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            pts_feats_forseg (list[torch.Tensor]): For diff bevencoder
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
            if self.heatmap2seg:
                pts_feats_list = []
                if self.pts_bbox_head:
                    for task_id, out in enumerate(outs):
                        # print("HEATMAP", out[0]['heatmap'].shape)
                        # print("BEV", pts_feats[0].shape)
                        pts_feats_list.append(out[0]['heatmap'])
                    if pts_feats_forseg is not None:
                        pts_feats_list.append(pts_feats_forseg[0])
                        pts_feats_forseg = torch.cat(pts_feats_list, dim=1)
                        # print("NEW BEV", pts_feats_forseg.shape)
                    else:
                        pts_feats_list.append(pts_feats[0])
                        pts_feats = torch.cat(pts_feats_list, dim=1)
                        # print("NEW BEV", pts_feats.shape)
                else:
                    raise TypeError("heatmap2seg is true but doesn't have a pts_bbox_head.")
            if pts_feats_forseg is not None:
                losses = self.pts_seg_head(pts_feats_forseg, gt_masks_bev)
                loss_dict.update(losses)
            else:
                losses = self.pts_seg_head(pts_feats, gt_masks_bev)
                loss_dict.update(losses)
        # print(loss_dict)
        return loss_dict

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        assert False

    def simple_test_pts(self, x, img_metas, rescale=False, return_outs=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        if return_outs:
            return bbox_results, outs
        else:
            return bbox_results

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    gt_masks_bev=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""

        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, with_bevencoder=True, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]

        # import numpy as np
        # import matplotlib.pyplot as plt
        # img_to_show = img_feats[0][0][:, 14:-14, 14:-14]
        # print(img_to_show.shape)
        # img_to_show_std = img_to_show.std(dim=0)
        # img_to_show_mean = torch.abs(img_to_show).mean(dim=0)
        # x = np.arange(-40, 40, 0.8)
        # y = np.arange(-40, 40, 0.8)
        # plt.figure(figsize=(10, 10))
        # plt.pcolormesh(x, y, img_to_show_mean.cpu().numpy(), cmap='viridis')
        # plt.savefig("/home/xiazhongyu/vis/det_feat_mean.png")
        # plt.figure(figsize=(10, 10))
        # plt.pcolormesh(x, y, img_to_show_std.cpu().numpy(), cmap='viridis')
        # plt.savefig("/home/xiazhongyu/vis/det_feat_std.png")
        # exit(0)

        if self.pts_bbox_head:
            bbox_pts, outs = self.simple_test_pts(img_feats, img_metas, rescale=rescale, return_outs=True)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox

        if self.pts_seg_head:
            if self.heatmap2seg:
                pts_feats_list = []
                if self.pts_bbox_head:
                    for task_id, out in enumerate(outs):
                        pts_feats_list.append(out[0]['heatmap'])
                    pts_feats_list.append(img_feats[0])
                    img_feats = torch.cat(pts_feats_list, dim=1)
                else:
                    raise TypeError("heatmap2seg is true but doesn't have a pts_bbox_head.")

            bbox_segs = self.pts_seg_head(img_feats, gt_masks_bev)
            for result_dict, pts_seg, gt in zip(bbox_list, bbox_segs, gt_masks_bev):
                result_dict['pts_seg'] = pts_seg
                result_dict['gt_masks_bev'] = gt
        return bbox_list

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        img_feats, _, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, with_bevencoder=True, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(img_feats)
        return outs


@DETECTORS.register_module()
class BEVDetTRT(BEVDet):

    def result_serialize(self, outs):
        outs_ = []
        for out in outs:
            for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                outs_.append(out[0][key])
        return outs_

    def result_deserialize(self, outs):
        outs_ = []
        keys = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
        for head_id in range(len(outs) // 6):
            outs_head = [dict()]
            for kid, key in enumerate(keys):
                outs_head[0][key] = outs[head_id * 6 + kid]
            outs_.append(outs_head)
        return outs_

    def forward(
            self,
            img,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_starts,
            interval_lengths,
    ):
        x = self.img_backbone(img)
        x = self.img_neck(x)
        x = self.img_view_transformer.depth_net(x)
        depth = x[:, :self.img_view_transformer.D].softmax(dim=1)
        tran_feat = x[:, self.img_view_transformer.D:(
                self.img_view_transformer.D +
                self.img_view_transformer.out_channels)]
        tran_feat = tran_feat.permute(0, 2, 3, 1)
        x = TRTBEVPoolv2.apply(depth.contiguous(), tran_feat.contiguous(),
                               ranks_depth, ranks_feat, ranks_bev,
                               interval_starts, interval_lengths)
        x = x.permute(0, 3, 1, 2).contiguous()
        bev_feat = self.bev_encoder(x)
        outs = self.pts_bbox_head([bev_feat])
        outs = self.result_serialize(outs)
        return outs

    def get_bev_pool_input(self, input):
        input = self.prepare_inputs(input)
        coor = self.img_view_transformer.get_lidar_coor(*input[1:7])
        return self.img_view_transformer.voxel_pooling_prepare_v2(coor)


@DETECTORS.register_module()
class BEVDet4D(BEVDet):
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
        super(BEVDet4D, self).__init__(**kwargs)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)
        self.align_after_view_transfromation = align_after_view_transfromation
        self.num_frame = num_adj + 1

        self.with_prev = with_prev
        self.grid = None

    def gen_grid(self, input, sensor2keyegos, bda, bda_adj=None, flag=False):
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
        if flag: # 部署的时候只有这么写不会出错
            tf = torch.inverse(feat2bev.cpu()).cuda().matmul(l02l1).matmul(feat2bev)
        else:
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

    def extract_img_feat_sequential(self, inputs, feat_prev, with_bevencoder=True):
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
        if with_bevencoder:
            x = self.bev_encoder(bev_feat)
        else:
            x = bev_feat
        return [x], depth

    def prepare_inputs(self, inputs, stereo=False, flag=False):
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
        if flag: # 部署的时候只有这么写不会出错
            device = keyego2global.device
            global2keyego = torch.inverse(keyego2global.double().to('cpu')).to(device)
        else:
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
                         with_bevencoder=True,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            return self.extract_img_feat_sequential(img, kwargs['feat_prev'], with_bevencoder=with_bevencoder)
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
        if not hasattr(self, 'img_down2top_encoder_backbone'):
            bev_feat = torch.cat(bev_feat_list, dim=1)
            if with_bevencoder:
                x = self.bev_encoder(bev_feat)
            else:
                x = bev_feat
        else:
            if with_bevencoder:
                x = self.bev_encoder(bev_feat_list)
            else:
                raise ValueError('Define img_down2top_encoder_backbone but set with_bevencoder'
                                 '=False. You should skip bev_encoder when defining a mix'
                                 'backbone model.')
        return [x], depth_list[0]


@DETECTORS.register_module()
class BEVDepth4D(BEVDet4D):

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
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, with_bevencoder=True, **kwargs)
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, gt_masks_bev,
                                            img_metas, gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses


@DETECTORS.register_module()
class BEVDepth4DTRT(BEVDepth4D):
    def result_serialize(self, outs):
        outs_ = []
        for out in outs:
            for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                outs_.append(out[0][key])
        return outs_

    def result_deserialize(self, outs):
        outs_ = []
        keys = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
        for head_id in range(len(outs) // 6):
            outs_head = [dict()]
            for kid, key in enumerate(keys):
                outs_head[0][key] = outs[head_id * 6 + kid]
            outs_.append(outs_head)
        return outs_

    def forward(
            self,
            imgs,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_starts,
            interval_lengths,
            mlp_input,
            feat_prev,
    ):
        bev_feat_list = []
        x = self.img_backbone(imgs)
        x = self.img_neck(x)
        x = self.img_view_transformer.depth_net(x, mlp_input, None)
        depth = x[:, :self.img_view_transformer.D].softmax(dim=1)
        tran_feat = x[:, self.img_view_transformer.D:(
                self.img_view_transformer.D +
                self.img_view_transformer.out_channels)]
        tran_feat = tran_feat.permute(0, 2, 3, 1)
        x = TRTBEVPoolv2.apply(depth.contiguous(), tran_feat.contiguous(),
                            ranks_depth, ranks_feat, ranks_bev,
                            interval_starts, interval_lengths)
        x = x.permute(0, 3, 1, 2).contiguous()
        if self.pre_process:
            x = self.pre_process_net(x)[0]
        bev_feat_list.append(x)
        # align the feat_prev
        _, C, H, W = feat_prev.shape
        bev_feat_list.append(feat_prev.view(1, (self.num_frame - 1) * C, H, W))
        bev_feat = torch.cat(bev_feat_list, dim=1)
        bev_feat = self.bev_encoder(bev_feat)
        outs = self.pts_bbox_head([bev_feat])
        outs = self.result_serialize(outs)
        return outs, bev_feat_list[0] # 用来保存起来作为下一帧使用

    def get_bev_feat_sequential( # 提前计算出bev特征图，align after view transformation
            self,
            imgs,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_starts,
            interval_lengths,
            mlp_input,
            ego2globals,
            sensor2keyegos,
            bda,
            img_metas=None,
    ):
        bev_feat_list = []
        for idx in range(imgs.shape[0]):
            if idx == 0:
                continue
            x = self.img_backbone(imgs[idx,:,:,:,:].squeeze(0))
            x = self.img_neck(x)
            x = self.img_view_transformer.depth_net(x, mlp_input[idx:idx+1,:,:], None)
            depth = x[:, :self.img_view_transformer.D, ...].softmax(dim=1)
            tran_feat = x[:, self.img_view_transformer.D:(
                    self.img_view_transformer.D +
                    self.img_view_transformer.out_channels), ...]
            tran_feat = tran_feat.permute(0, 2, 3, 1)
            x = TRTBEVPoolv2.apply(depth.contiguous(), tran_feat.contiguous(),
                                ranks_depth[idx], ranks_feat[idx], ranks_bev[idx],
                                interval_starts[idx], interval_lengths[idx])
            x = x.permute(0, 3, 1, 2).contiguous()
            if self.pre_process:
                x = self.pre_process_net(x)[0]
            bev_feat_list.append(x)

        assert sensor2keyegos.shape[0] == self.num_frame # 1 * 9 = 9
        feat_prev = torch.cat(bev_feat_list, dim=0)
        ego2globals_curr = \
            ego2globals[0:1,:,:,:].repeat(self.num_frame - 1, 1, 1, 1)
        sensor2keyegos_curr = \
            sensor2keyegos[0:1,:,:,:].repeat(self.num_frame - 1, 1, 1, 1)
        ego2globals_prev = ego2globals[1:,:,:,:]
        sensor2keyegos_prev = sensor2keyegos[1:,:,:,:]
        bda_curr = bda.repeat(self.num_frame - 1, 1, 1)
        return feat_prev, sensor2keyegos_curr, ego2globals_curr, \
            sensor2keyegos_prev, ego2globals_prev, bda_curr


    def get_bev_pool_input(self, input):
        imgs, sensor2keyegos, ego2globals, intrins, \
        post_rots, post_trans, bda, _ = self.prepare_inputs(input, flag=True)
        mlp_input_list = []
        metas_list = []
        for img, sensor2keyego, ego2global, intrin, post_rot, post_tran in zip(
                imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans):
            # sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0] # assert align_after_view_transformation==True
            mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin, post_rot, post_tran, bda)
            mlp_input_list.append(mlp_input)
            coor = self.img_view_transformer.get_lidar_coor(sensor2keyego, ego2global, intrin, post_rot, post_tran, bda)
            metas = self.img_view_transformer.voxel_pooling_prepare_v2(coor)
            metas_list.append(metas)
        return imgs, mlp_input_list, metas_list, sensor2keyegos, ego2globals, bda


@DETECTORS.register_module()
class BEVDepth4D_d2t(BEVDepth4D):
    def __init__(self, img_down2top_encoder_backbone, **kwargs):
        super(BEVDepth4D, self).__init__(**kwargs)
        self.img_down2top_encoder_backbone = builder.build_backbone(img_down2top_encoder_backbone)


@DETECTORS.register_module()
class BEVStereo4D(BEVDepth4D):
    def __init__(self, **kwargs):
        super(BEVStereo4D, self).__init__(**kwargs)
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
                         with_bevencoder=True,
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
        if pred_prev:
            fid = 0
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame - self.extra_ref_frames
            if self.align_after_view_transfromation:
                sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
            mlp_input = self.img_view_transformer.get_mlp_input(
                sensor2keyegos[0], ego2globals[0], intrin,
                post_rot, post_tran, bda)
            inputs_curr = (img, sensor2keyego, ego2global, intrin,
                           post_rot, post_tran, bda, mlp_input,
                           feat_prev_iv, curr2adjsensor[fid],
                           extra_ref_frame)
            bev_feat, depth, feat_curr_iv = \
                self.prepare_bev_feat(*inputs_curr)
            print("############", with_bevencoder)
            if with_bevencoder:
                x = self.bev_encoder(torch.cat([bev_feat, bev_feat], dim=1)) # sim prev
                return [x], depth_key_frame
            else:
                return [torch.cat([bev_feat, bev_feat], dim=1)], depth
        for fid in range(self.num_frame - 1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame - self.extra_ref_frames
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
        if not self.with_prev:
            bev_feat_key = bev_feat_list[0]
            if len(bev_feat_key.shape) == 4:
                b, c, h, w = bev_feat_key.shape
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
            for adj_id in range(self.num_frame - 2):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[self.num_frame - 2 - adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        if with_bevencoder:
            x = self.bev_encoder(bev_feat)
            return [x], depth_key_frame
        else:
            return [bev_feat], depth_key_frame



@DETECTORS.register_module()
class HoPBEVDet4D(BEVDet4D):
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
                 with_hop=False,
                 history_decoder=None,
                 loss_weight_aux=1.,
                 aux_bbox_head=None,
                 aux_train_cfg=None,
                 aux_test_cfg=None,
                 det_multi_frame=False,
                 **kwargs):
        super(HoPBEVDet4D, self).__init__(**kwargs)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)
        self.align_after_view_transfromation = align_after_view_transfromation
        self.num_frame = num_adj + 1
        self.with_hop = with_hop
        self.with_prev = with_prev
        self.loss_weight_aux = loss_weight_aux
        self.det_multi_frame = det_multi_frame

        if aux_bbox_head is not None:
            self.aux_bbox_head = nn.ModuleList()
            for i, bbox_head in enumerate(aux_bbox_head):
                bbox_head.update(train_cfg=aux_train_cfg[i])
                bbox_head.update(test_cfg=aux_test_cfg[i])
                self.aux_bbox_head.append(builder.build_head(bbox_head))
                #self.aux_bbox_head[-1].voxel_size = voxel_size
        else:
            self.aux_bbox_head = None
        if self.with_hop:
            self.history_decoder = builder.build_backbone(history_decoder)
        else:
            self.history_decoder = None

    def extract_img_feat_sequential(self, inputs, feat_prev, with_bevencoder=True):
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
        if with_bevencoder:
            x = self.bev_encoder(bev_feat)
        else:
            x = bev_feat
        return [x], depth, bev_feat_list

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points.
        Return:
        (BEV Feature, None, depth)
        """
        img_feats, depth, prev_feats = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth, prev_feats)

    def extract_img_feat(self,
                         img,
                         img_metas,
                         with_bevencoder=True,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            return self.extract_img_feat_sequential(img, kwargs['feat_prev'], with_bevencoder=with_bevencoder)
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
        if not hasattr(self, 'img_down2top_encoder_backbone'):
            bev_feat = torch.cat(bev_feat_list, dim=1)
            if with_bevencoder:
                x = self.bev_encoder(bev_feat)
            else:
                x = bev_feat
        else:
            if with_bevencoder:
                x = self.bev_encoder(bev_feat_list)
            else:
                raise ValueError('Define img_down2top_encoder_backbone but set with_bevencoder'
                                 '=False. You should skip bev_encoder when defining a mix'
                                 'backbone model.')
        return [x], depth_list[0], bev_feat_list

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
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
        def upd_loss(losses, feat_idx, head_idx, weight=1):
            new_losses = dict()
            for k,v in losses.items():
                new_k = '{}-{}-{}'.format(k,feat_idx,head_idx)
                if isinstance(v,list) or isinstance(v,tuple):
                    new_losses[new_k] = [i*weight for i in v]
                else:new_losses[new_k] = v*weight
            return new_losses

        img_feats, pts_feats, _, prev_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        if self.with_hop and self.with_prev: # 注意前几epoch还没有prev frame呢
            if self.det_multi_frame:
                adj_gt_3d_list = list(range(len(img_metas[0]['adj_gt_3d'])-1)) # 从t-1到t-n+1
            else:
                adj_gt_3d_list = [0]
            img_metas_aux = img_metas
            for idx in adj_gt_3d_list:
                gt_bboxes_3d_aux = [img_meta['adj_gt_3d'][idx][0] for img_meta in img_metas]
                gt_labels_3d_aux = [img_meta['adj_gt_3d'][idx][1].to('cuda') for img_meta in img_metas]
                feature_bev_aux = [self.history_decoder(prev_feats[:idx+1]+prev_feats[idx+2:])]
                if self.aux_bbox_head is not None:
                    for i in range(len(self.aux_bbox_head)):
                        # feature_bev: [torch.Size([1, 384, 213, 125])]
                        # img_metas: dict_keys(['filename', 'ori_shape', 'img_shape', 'lidar2img', 'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'transformation_3d_flow', 'img_info'])
                        bbox_head = self.aux_bbox_head[i]
                        if 'Center' in bbox_head.__class__.__name__:
                            outs = bbox_head(feature_bev_aux)
                            loss_inputs = [gt_bboxes_3d_aux, gt_labels_3d_aux, outs]
                            loss_det = bbox_head.loss(*loss_inputs)
                        else:
                            x = bbox_head(feature_bev_aux)
                            loss_det = bbox_head.loss(*x, gt_bboxes_3d_aux, gt_labels_3d_aux, img_metas_aux)
                        loss_det = upd_loss(loss_det, idx, i, weight=self.loss_weight_aux)
                        losses.update(loss_det)
        return losses

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    gt_masks_bev=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        if self.pts_bbox_head:
            bbox_pts, outs = self.simple_test_pts(img_feats, img_metas, rescale=rescale, return_outs=True)
            # if len(bbox_pts[0]['boxes_3d']) > 500:
            #     print('DEBUG: find > 500 bboxes per sample', img_metas[0]['sample_idx'], len(bbox_pts[0]['boxes_3d']))
            #     exit(0)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox

        if self.pts_seg_head:

            if self.heatmap2seg:
                pts_feats_list = []
                if self.pts_bbox_head:
                    for task_id, out in enumerate(outs):
                        pts_feats_list.append(out[0]['heatmap'])
                    pts_feats_list.append(img_feats[0])
                    img_feats = torch.cat(pts_feats_list, dim=1)
                else:
                    raise TypeError("heatmap2seg is true but doesn't have a pts_bbox_head.")

            bbox_segs = self.pts_seg_head(img_feats, gt_masks_bev)
            for result_dict, pts_seg, gt in zip(bbox_list, bbox_segs, gt_masks_bev):
                result_dict['pts_seg'] = pts_seg
                result_dict['gt_masks_bev'] = gt

        return bbox_list


@DETECTORS.register_module()
class HoPBEVDepth4D(HoPBEVDet4D):

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
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
        def upd_loss(losses, feat_idx, head_idx, weight=1):
            new_losses = dict()
            for k,v in losses.items():
                new_k = '{}-{}-{}'.format(k,feat_idx,head_idx)
                if isinstance(v,list) or isinstance(v,tuple):
                    new_losses[new_k] = [i*weight for i in v]
                else:new_losses[new_k] = v*weight
            return new_losses

        img_feats, pts_feats, depth, prev_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        if self.with_hop and self.with_prev:
            if self.det_multi_frame:
                adj_gt_3d_list = list(range(len(img_metas[0]['adj_gt_3d'])-1)) # 从t-1到t-n+1
            else:
                adj_gt_3d_list = [0]
            img_metas_aux = img_metas
            for idx in adj_gt_3d_list:
                gt_bboxes_3d_aux = [img_meta['adj_gt_3d'][idx][0] for img_meta in img_metas]
                gt_labels_3d_aux = [img_meta['adj_gt_3d'][idx][1].to('cuda') for img_meta in img_metas]
                feature_bev_aux = [self.history_decoder(prev_feats[:idx+1]+prev_feats[idx+2:])]
                if self.aux_bbox_head is not None:
                    for i in range(len(self.aux_bbox_head)):
                        # feature_bev: [torch.Size([1, 384, 213, 125])]
                        # img_metas: dict_keys(['filename', 'ori_shape', 'img_shape', 'lidar2img', 'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'transformation_3d_flow', 'img_info'])
                        bbox_head = self.aux_bbox_head[i]
                        if 'Center' in bbox_head.__class__.__name__:
                            outs = bbox_head(feature_bev_aux)
                            loss_inputs = [gt_bboxes_3d_aux, gt_labels_3d_aux, outs]
                            loss_det = bbox_head.loss(*loss_inputs)
                        else:
                            x = bbox_head(feature_bev_aux)
                            loss_det = bbox_head.loss(*x, gt_bboxes_3d_aux, gt_labels_3d_aux, img_metas_aux)
                        loss_det = upd_loss(loss_det, idx, i, weight=self.loss_weight_aux)
                        losses.update(loss_det)
        return losses


@DETECTORS.register_module()
class BEVStereo4DHoP(BEVStereo4D):
    r"""BEVStereo4D paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2203.17054>`_
    """
    def __init__(self,
                 with_hop=False,
                 loss_weight_aux=1.,
                 aux_bbox_head=None,
                 aux_train_cfg=None,
                 aux_test_cfg=None,
                 **kwargs):
        super(BEVStereo4DHoP, self).__init__(**kwargs)
        self.with_hop = with_hop
        self.loss_weight_aux = loss_weight_aux

        if aux_bbox_head is not None:
            self.aux_bbox_head = nn.ModuleList()
            for i, bbox_head in enumerate(aux_bbox_head):
                bbox_head.update(train_cfg=aux_train_cfg[i])
                bbox_head.update(test_cfg=aux_test_cfg[i])
                self.aux_bbox_head.append(builder.build_head(bbox_head))
                #self.aux_bbox_head[-1].voxel_size = voxel_size
        else:
            self.aux_bbox_head = None


    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points.
        Return:
        (BEV Feature, None, depth)
        """
        img_feats, depth, prev_feats = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth, prev_feats)


    def extract_img_feat(self,
                         img,
                         img_metas,
                         with_bevencoder=True,
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
        for fid in range(self.num_frame - 1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame - self.extra_ref_frames
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
            if len(bev_feat_key.shape) == 4:
                b, c, h, w = bev_feat_key.shape
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
            for adj_id in range(self.num_frame - 2):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[self.num_frame - 2 - adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        if with_bevencoder:
            x = self.bev_encoder(bev_feat)
            return [x], depth_key_frame, bev_feat_list
        else:
            return [bev_feat], depth_key_frame, bev_feat_list

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
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
        def upd_loss(losses, idx, weight=1):
            new_losses = dict()
            for k,v in losses.items():
                new_k = '{}{}'.format(k,idx)
                if isinstance(v,list) or isinstance(v,tuple):
                    new_losses[new_k] = [i*weight for i in v]
                else:new_losses[new_k] = v*weight
            return new_losses

        img_feats, pts_feats, depth, prev_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        if self.with_hop:
            img_metas_aux = img_metas
            gt_bboxes_3d_aux = [img_meta['adj_gt_3d'][0][0] for img_meta in img_metas]
            gt_labels_3d_aux = [img_meta['adj_gt_3d'][0][1].to('cuda') for img_meta in img_metas]
            # feature_bev_aux = [self.history_decoder(prev_feats[:1]+prev_feats[2:])]
            feature_bev_aux = img_feats # 直接用这个concat的bev就行
        if self.aux_bbox_head is not None:
            for i in range(len(self.aux_bbox_head)):
                # feature_bev: [torch.Size([1, 384, 213, 125])]
                # img_metas: dict_keys(['filename', 'ori_shape', 'img_shape', 'lidar2img', 'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'transformation_3d_flow', 'img_info'])
                bbox_head = self.aux_bbox_head[i]
                if 'Center' in bbox_head.__class__.__name__:
                    outs = bbox_head(feature_bev_aux)
                    loss_inputs = [gt_bboxes_3d_aux, gt_labels_3d_aux, outs]
                    loss_det = bbox_head.loss(*loss_inputs)
                else:
                    x = bbox_head(feature_bev_aux)
                    loss_det = bbox_head.loss(*x, gt_bboxes_3d_aux, gt_labels_3d_aux, img_metas_aux)
                loss_det = upd_loss(loss_det, i, weight=self.loss_weight_aux)
                losses.update(loss_det)
        return losses

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    gt_masks_bev=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        if self.pts_bbox_head:
            bbox_pts, outs = self.simple_test_pts(img_feats, img_metas, rescale=rescale, return_outs=True)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox

        if self.pts_seg_head:

            if self.heatmap2seg:
                pts_feats_list = []
                if self.pts_bbox_head:
                    for task_id, out in enumerate(outs):
                        pts_feats_list.append(out[0]['heatmap'])
                    pts_feats_list.append(img_feats[0])
                    img_feats = torch.cat(pts_feats_list, dim=1)
                else:
                    raise TypeError("heatmap2seg is true but doesn't have a pts_bbox_head.")

            bbox_segs = self.pts_seg_head(img_feats, gt_masks_bev)
            for result_dict, pts_seg, gt in zip(bbox_list, bbox_segs, gt_masks_bev):
                result_dict['pts_seg'] = pts_seg
                result_dict['gt_masks_bev'] = gt
        return bbox_list