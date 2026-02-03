# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

# from ..builder import LOSSES
from mmdet3d.models import LOSSES


@LOSSES.register_module()
class UniDistillLoss(nn.Module):
    """
    """

    def __init__(
        self,
        tau: float = 1.0,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.tau = tau
        self.loss_weight = loss_weight

    def forward(self, preds_S: torch.Tensor, preds_T: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W).

        Return:
            torch.Tensor: The calculated loss value.
        """
      #  
        if isinstance(preds_S,list):
            preds_S = preds_S[-1]
            preds_T = preds_T[-1]
        if preds_S.shape[-2:] != preds_T.shape[-2:]:
            H_new, W_new = preds_T.shape[-2:]
            preds_S = nn.functional.interpolate(preds_S, size=(H_new, W_new), mode='bilinear', align_corners=False)
        
        N, C, H, W = preds_S.shape
        softmax_pred_T = F.softmax(preds_T.view(-1, W * H) / self.tau, dim=1)

        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum(softmax_pred_T *
                         logsoftmax(preds_T.view(-1, W * H) / self.tau) -
                         softmax_pred_T *
                         logsoftmax(preds_S.view(-1, W * H) / self.tau)) * (
                             self.tau**2)

        loss = self.loss_weight * loss / (C * N)

        return loss
    def FeatureDistillLoss(feature_lidar: torch.Tensor, feature_fuse: torch.Tensor, gt_boxes_bev_coords: torch.Tensor, gt_boxes_indices: torch.Tensor):
        h, w = feature_lidar.shape[-2:]
        gt_boxes_bev_center: torch.Tensor = torch.mean(gt_boxes_bev_coords, dim=2).unsqueeze(2)
        gt_boxes_bev_edge_1: torch.Tensor = torch.mean(
            gt_boxes_bev_coords[:, :, [0, 1], :], dim=2
        ).unsqueeze(2)
        gt_boxes_bev_edge_2: torch.Tensor = torch.mean(
            gt_boxes_bev_coords[:, :, [1, 2], :], dim=2
        ).unsqueeze(2)
        gt_boxes_bev_edge_3 = torch.mean(
            gt_boxes_bev_coords[:, :, [2, 3], :], dim=2
        ).unsqueeze(2)
        gt_boxes_bev_edge_4 = torch.mean(
            gt_boxes_bev_coords[:, :, [0, 3], :], dim=2
        ).unsqueeze(2)
        gt_boxes_bev_all = torch.cat(
            (
                gt_boxes_bev_coords,
                gt_boxes_bev_center,
                gt_boxes_bev_edge_1,
                gt_boxes_bev_edge_2,
                gt_boxes_bev_edge_3,
                gt_boxes_bev_edge_4,
            ),
            dim=2,
        )
        gt_boxes_bev_all[:, :, :, 0] = (gt_boxes_bev_all[:, :, :, 0] - w / 2) / (w / 2)
        gt_boxes_bev_all[:, :, :, 1] = (gt_boxes_bev_all[:, :, :, 1] - h / 2) / (h / 2)
        gt_boxes_bev_all[:, :, :, [0, 1]] = gt_boxes_bev_all[:, :, :, [1, 0]]
        feature_lidar_sample = torch.nn.functional.grid_sample(
            feature_lidar, gt_boxes_bev_all
        )
        feature_lidar_sample = feature_lidar_sample.permute(0, 2, 3, 1)
        feature_fuse_sample = torch.nn.functional.grid_sample(
            feature_fuse, gt_boxes_bev_all
        )
        feature_fuse_sample = feature_fuse_sample.permute(0, 2, 3, 1)
        criterion = nn.L1Loss(reduce=False)
        loss_feature_distill: torch.Tensor = criterion(
            feature_lidar_sample[gt_boxes_indices], feature_fuse_sample[gt_boxes_indices]
        )
        loss_feature_distill = torch.mean(loss_feature_distill, 2)
        loss_feature_distill = torch.mean(loss_feature_distill, 1)
        loss_feature_distill = torch.sum(loss_feature_distill)
        weight = gt_boxes_indices.float().sum()
        weight = reduce_mean(weight)
        loss_feature_distill = loss_feature_distill / (weight + 1e-4)
        return loss_feature_distill
    def BEVDistillLoss(bev_lidar, bev_fuse, gt_boxes_bev_coords, gt_boxes_indices):
        h, w = bev_lidar.shape[-2:]
        gt_boxes_bev_center = torch.mean(gt_boxes_bev_coords, dim=2).unsqueeze(2)
        gt_boxes_bev_edge_1 = torch.mean(
            gt_boxes_bev_coords[:, :, [0, 1], :], dim=2
        ).unsqueeze(2)
        gt_boxes_bev_edge_2 = torch.mean(
            gt_boxes_bev_coords[:, :, [1, 2], :], dim=2
        ).unsqueeze(2)
        gt_boxes_bev_edge_3 = torch.mean(
            gt_boxes_bev_coords[:, :, [2, 3], :], dim=2
        ).unsqueeze(2)
        gt_boxes_bev_edge_4 = torch.mean(
            gt_boxes_bev_coords[:, :, [0, 3], :], dim=2
        ).unsqueeze(2)
        gt_boxes_bev_all = torch.cat(
            (
                gt_boxes_bev_coords,
                gt_boxes_bev_center,
                gt_boxes_bev_edge_1,
                gt_boxes_bev_edge_2,
                gt_boxes_bev_edge_3,
                gt_boxes_bev_edge_4,
            ),
            dim=2,
        )
        gt_boxes_bev_all[:, :, :, 0] = (gt_boxes_bev_all[:, :, :, 0] - w / 2) / (w / 2)
        gt_boxes_bev_all[:, :, :, 1] = (gt_boxes_bev_all[:, :, :, 1] - h / 2) / (h / 2)
        gt_boxes_bev_all[:, :, :, [0, 1]] = gt_boxes_bev_all[:, :, :, [1, 0]]
        feature_lidar_sample = torch.nn.functional.grid_sample(bev_lidar, gt_boxes_bev_all)
        feature_lidar_sample = feature_lidar_sample.permute(0, 2, 3, 1)
        feature_fuse_sample = torch.nn.functional.grid_sample(bev_fuse, gt_boxes_bev_all)
        feature_fuse_sample = feature_fuse_sample.permute(0, 2, 3, 1)
        criterion = nn.L1Loss(reduce=False)
        weight = gt_boxes_indices.float().sum()
        weight = reduce_mean(weight)
        gt_boxes_sample_lidar_feature = feature_lidar_sample.contiguous().view(
            -1, feature_lidar_sample.shape[-2], feature_lidar_sample.shape[-1]
        )
        gt_boxes_sample_fuse_feature = feature_fuse_sample.contiguous().view(
            -1, feature_fuse_sample.shape[-2], feature_fuse_sample.shape[-1]
        )
        gt_boxes_sample_lidar_feature = gt_boxes_sample_lidar_feature / (
            torch.norm(gt_boxes_sample_lidar_feature, dim=-1, keepdim=True) + 1e-4
        )
        gt_boxes_sample_fuse_feature = gt_boxes_sample_fuse_feature / (
            torch.norm(gt_boxes_sample_fuse_feature, dim=-1, keepdim=True) + 1e-4
        )
        gt_boxes_lidar_rel = torch.bmm(
            gt_boxes_sample_lidar_feature,
            torch.transpose(gt_boxes_sample_lidar_feature, 1, 2),
        )
        gt_boxes_fuse_rel = torch.bmm(
            gt_boxes_sample_fuse_feature,
            torch.transpose(gt_boxes_sample_fuse_feature, 1, 2),
        )
        gt_boxes_lidar_rel = gt_boxes_lidar_rel.contiguous().view(
            gt_boxes_bev_coords.shape[0],
            gt_boxes_bev_coords.shape[1],
            gt_boxes_lidar_rel.shape[-2],
            gt_boxes_lidar_rel.shape[-1],
        )
        gt_boxes_fuse_rel = gt_boxes_fuse_rel.contiguous().view(
            gt_boxes_bev_coords.shape[0],
            gt_boxes_bev_coords.shape[1],
            gt_boxes_fuse_rel.shape[-2],
            gt_boxes_fuse_rel.shape[-1],
        )
        loss_rel = criterion(
            gt_boxes_lidar_rel[gt_boxes_indices], gt_boxes_fuse_rel[gt_boxes_indices]
        )
        loss_rel = torch.mean(loss_rel, 2)
        loss_rel = torch.mean(loss_rel, 1)
        loss_rel = torch.sum(loss_rel)
        loss_rel = loss_rel / (weight + 1e-4)
        return loss_rel

    def ResponseDistillLoss(resp_lidar, resp_fuse, gt_boxes, pc_range, voxel_size, out_size_scale):
        cls_lidar = []
        reg_lidar = []
        cls_fuse = []
        reg_fuse = []
        criterion = nn.L1Loss(reduce=False)
        for task_id, task_out in enumerate(resp_lidar):
            cls_lidar.append(task_out["hm"])
            cls_fuse.append(_sigmoid(resp_fuse[task_id]["hm"] / 2))
            reg_lidar.append(
                torch.cat(
                    [
                        task_out["reg"],
                        task_out["height"],
                        task_out["dim"],
                        task_out["rot"],
                        task_out["vel"],
                        task_out["iou"],
                    ],
                    dim=1,
                )
            )
            reg_fuse.append(
                torch.cat(
                    [
                        resp_fuse[task_id]["reg"],
                        resp_fuse[task_id]["height"],
                        resp_fuse[task_id]["dim"],
                        resp_fuse[task_id]["rot"],
                        resp_fuse[task_id]["vel"],
                        resp_fuse[task_id]["iou"],
                    ],
                    dim=1,
                )
            )
        cls_lidar = torch.cat(cls_lidar, dim=1)
        reg_lidar = torch.cat(reg_lidar, dim=1)
        cls_fuse = torch.cat(cls_fuse, dim=1)
        reg_fuse = torch.cat(reg_fuse, dim=1)
        cls_lidar_max, _ = torch.max(cls_lidar, dim=1)
        cls_fuse_max, _ = torch.max(cls_fuse, dim=1)
        gaussian_mask = calculate_box_mask_gaussian(
            reg_lidar.shape,
            gt_boxes.cpu().detach().numpy(),
            pc_range,
            voxel_size,
            out_size_scale,
        )
        diff_reg = criterion(reg_lidar, reg_fuse)
        diff_cls = criterion(cls_lidar_max, cls_fuse_max)
        diff_reg = torch.mean(diff_reg, dim=1)
        diff_reg = diff_reg * gaussian_mask
        diff_cls = diff_cls * gaussian_mask
        weight = gaussian_mask.sum()
        weight = reduce_mean(weight)
        loss_reg_distill = torch.sum(diff_reg) / (weight + 1e-4)
        loss_cls_distill = torch.sum(diff_cls) / (weight + 1e-4)
        return loss_cls_distill, loss_reg_distill




