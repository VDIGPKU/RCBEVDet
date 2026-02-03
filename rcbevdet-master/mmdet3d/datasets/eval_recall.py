import mmcv
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def bbox_cxcywh_to_xyxy(bbox):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    # factor = [960, 640]
    factor = [1, 1]
    cx = bbox[:, 0:1] * factor[0]
    cy = bbox[:, 1:2] * factor[1]
    w = bbox[:, 2:3] * factor[0]
    h = bbox[:, 3:4] * factor[1]
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)

def bbox_overlaps(bboxes1,
                  bboxes2,
                  mode='iou',
                  eps=1e-6,
                  use_legacy_coordinate=False):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.
    Args:
        bboxes1 (ndarray): Shape (n, 4)
        bboxes2 (ndarray): Shape (k, 4)
        mode (str): IOU (intersection over union) or IOF (intersection
            over foreground)
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Note when function is used in `VOCDataset`, it should be
            True to align with the official implementation
            `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar`
            Default: False.
    Returns:
        ious (ndarray): Shape (n, k)
    """

    assert mode in ['iou', 'iof']
    if not use_legacy_coordinate:
        extra_length = 0.
    else:
        extra_length = 1.
    bboxes1 = bboxes1.to(torch.float32)
    bboxes2 = bboxes2.to(torch.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols), dtype=torch.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows), dtype=torch.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + extra_length) * (
        bboxes1[:, 3] - bboxes1[:, 1] + extra_length)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + extra_length) * (
        bboxes2[:, 3] - bboxes2[:, 1] + extra_length)
    for i in range(bboxes1.shape[0]):
        x_start = torch.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = torch.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = torch.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = torch.minimum(bboxes1[i, 3], bboxes2[:, 3])
        
        overlap = torch.maximum(x_end - x_start + extra_length, torch.zeros_like(x_end)) * torch.maximum(
            y_end - y_start + extra_length, torch.zeros_like(y_end))
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = torch.maximum(union, torch.ones_like(union) * eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious

def eval_recall(gt_2dbboxes, gt_2d_depth, pred_bboxes, thr, range_m = [0, 150]):
    num_true_sample = 0
    num_gt_sample = 0
    num_pred_sample = 0
    for i in range(len(gt_2dbboxes)):
        pred_bbox = pred_bboxes[i]
        if len(gt_2dbboxes[i]) == 0:
            continue
        valid_gt_2d_depth = np.logical_and(gt_2d_depth[i] >= range_m[0], gt_2d_depth[i] < range_m[1])
        gt_bbox = gt_2dbboxes[i][valid_gt_2d_depth, :]
        if len(gt_bbox) == 0:
            continue
        
        gt_bbox = gt_bbox.reshape(-1, 4)
        if len(pred_bbox) == 0:
            num_gt_sample = num_gt_sample + len(gt_bbox)
            continue
        gt_bbox = torch.tensor(gt_bbox, device=pred_bbox.device)
        pred_bbox = bbox_cxcywh_to_xyxy(pred_bbox)
        ious = bbox_overlaps(gt_bbox, pred_bbox)
        gt_ious = torch.zeros((ious.shape[0]))
        for j in range(ious.shape[0]):
            gt_max_overlaps = ious.argmax(axis=1)
            max_ious = ious[torch.arange(0, ious.shape[0]), gt_max_overlaps]
            gt_idx = max_ious.argmax()
            gt_ious[j] = max_ious[gt_idx]
            box_idx = gt_max_overlaps[gt_idx]
            ious[gt_idx, :] = -1
            ious[:, box_idx] = -1
        num_true_sample = num_true_sample + (gt_ious >= thr).sum()
        num_gt_sample = num_gt_sample + len(gt_bbox)
        num_pred_sample = num_pred_sample + len(pred_bbox)
        
    return num_gt_sample, num_true_sample, num_pred_sample
    
def cal_recall_2d(results_2d):
    thr_ious = [0.5, 0.3]
    for thr_iou in thr_ious:
        ranges_m = [[0, 50], [50, 100], [100, 150], [0, 150]]
        # ranges_m = [[0, 25], [25,50], [50, 75], [75,100], [100, 150], [0, 150]]
        for range_m in ranges_m:
            num_gt_total = 0
            num_true_total = 0
            num_sample = 0
            num_pred_total = 0
            gt_2d = mmcv.load("data/av2/gt_2d_val_nopadding.pkl", file_format='pkl')
            for token in tqdm(gt_2d.keys()):
                if token in results_2d.keys():
                    pred_bboxes = results_2d[token]
                    gt_2d_bbox = gt_2d[token]['gt_bboxes']
                    gt_2d_depth = gt_2d[token]['depths']

                    num_gt_sample, num_true_sample, num_pred_sample = eval_recall(gt_2d_bbox, gt_2d_depth, pred_bboxes, thr_iou, range_m)
                    num_pred_total += num_pred_sample
                    num_gt_total = num_gt_total + num_gt_sample
                    num_true_total = num_true_total + num_true_sample
                    num_sample += 1
                    
            recall = num_true_total / num_gt_total
            
            print("num_sample is {}".format(num_sample))
            print("num_pred_total is {}".format(num_pred_total))
            print("num_gt_total is {}".format(num_gt_total))
            print("num_true_total is {}".format(num_true_total))
            print("recall with iou {} in range {} is {}".format(thr_iou, range_m, recall))
    
    return