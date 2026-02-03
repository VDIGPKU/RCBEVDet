import argparse
import mmcv
import os
import time
import pickle
import torch
from torch.nn import functional as F
import warnings
from os import path as osp
from mmcv import Config, DictAction

from mmdet3d.datasets import build_dataloader, build_dataset

from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
import numpy as np
from IPython import embed
from copy import deepcopy
import cv2

import matplotlib.pyplot as plt
from mmdet3d.ops import Voxelization
# from mmdet3d.ops import spconv as spconv
from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE
if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential



from mmdet3d.models.necks import LSSViewTransformerBEVDepth

def voxelize(points, pts_voxel_layer):
    """Apply dynamic voxelization to points.

    Args:
        points (list[torch.Tensor]): Points of each sample.

    Returns:
        tuple[torch.Tensor]: Concatenated points, number of points
            per voxel, and coordinates.
    """
    voxels, coors, num_points = [], [], []
    for res in points:

        res_voxels, res_coors, res_num_points = pts_voxel_layer(res)
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    # parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)


    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
    print('create bev transfomer...')
    view_transform = LSSViewTransformerBEVDepth(
                                grid_config=cfg.grid_config, 
                                input_size=cfg.data_config['input_size'],
                                out_channels=3,
                                downsample=1,)
    view_transform = view_transform.cuda()
    view_transform.eval()
    # build the dataloader
    print('create dataset...')
    dataset = build_dataset(cfg.data.train)

    for i in range(10):
        data = dataset[i+2]
        # embed()
        print(f'get data sample {i}')
        img_metas = data['img_metas'].data
        # img_metas = [img_metas]
        
        img = data['img_inputs']
        gt_depth = data['gt_depth']
        # print(len(img))
        x, rots, trans, intrins, post_rots, post_trans, bda = img
        x = x[:6, ...]
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1,-1,1,1)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1,-1,1,1)
        mean = torch.from_numpy(mean)
        std = torch.from_numpy(std)
        x = x * std + mean
        rots, trans, intrins, post_rots, post_trans, bda = \
            rots.unsqueeze(dim=0), trans.unsqueeze(dim=0), intrins.unsqueeze(dim=0), post_rots.unsqueeze(dim=0), post_trans.unsqueeze(dim=0), bda.unsqueeze(dim=0)
        x = x.unsqueeze(dim=0)
        gt_depth = gt_depth.unsqueeze(dim=0)
        # gt_depth = gt_depth.int()
        gt_depth = torch.clip(torch.floor(gt_depth), 0, 60).to(torch.long)
        print('gt depth shape', gt_depth.shape)
        tmp = np.array(gt_depth[0][0].cpu().numpy(), dtype=np.uint8)
        tmp[tmp>0] = 1
        mmcv.imwrite(tmp*255, f'vis/{i}_gt_depth.jpg')
        # print(gt_depth.shape)
        # print(gt_depth.device, x.device)
        B, N, C, H, W = x.shape
        # x = torch.nn.functional.interpolate(x, )

        x, rots, trans, intrins, post_rots, post_trans, bda = \
            x.cuda(), rots.cuda(), trans.cuda(), intrins.cuda(), post_rots.cuda(), post_trans.cuda(), bda.cuda()
        gt_depth = gt_depth.cuda()

        tran_feat = x.view(B * N, C, H, W)
        print('img shape', x.shape)
        # print(gt_depth.max(), gt_depth.min())
        # print((gt_depth>0).sum())
        # depth = view_transform.get_downsampled_gt_depth(gt_depth)
        depth = torch.nn.functional.one_hot(gt_depth, num_classes=view_transform.D+1)
        depth = depth.view(B * N, H, W,-1).permute(0, 3, 1, 2)
        depth = depth[:,1:,:,]
        depth = depth.long()
        print('gt depth shape logit', depth.shape)
        # print(depth, x)
        # print(x.max(), rots.max(), trans.max(), intrins.max(), post_rots.max(), post_trans.max(), bda.max(), depth.max(), tran_feat.max())
        # print(x.min(), rots.min(), trans.min(), intrins.min(), post_rots.min(), post_trans.min(), bda.min(), depth.min(), tran_feat.min())
        # tran_feat = torch.ones_like(x.view(B * N, C, H, W)).long()
        # print(tran_feat)
        out, _ = view_transform.view_transform([x, rots, trans, intrins, post_rots, post_trans, bda], depth, tran_feat)
        bev_img = out.detach()
        bev_img = bev_img[0]
        # x, rots, trans, intrins, post_rots, post_trans, depth_gt = img

        # view_transform

        # img_to_bev(img_metas, img, points[:,:3])
        # bev_img = img_to_bev(img)

        print(bev_img.size())
        c,h,w = bev_img.size()
        # bev_img[bev_img!=0] = 1
        bev_img = bev_img.view(c,h,w).permute(1,2,0)
        bev_img[bev_img==torch.nan] = 0
        # bev_img[bev_img!=0] = 1
        # bev_img = bev_img.sum(dim=-1)
        # bev_img = torch.clip(torch.floor(bev_img), 0, 1)
        # bev_img = bev_img.cpu().numpy()*255
        bev_img = bev_img.cpu().numpy()
        # bev_img = bev_img*std + mean
        print(bev_img.max(), bev_img.min())
        # np.savetxt('debug_img/test.txt', bev_img.reshape(-1))
        bev_img = np.array(bev_img, dtype=np.uint8)
        # print(bev_img.max())
        # print((bev_img>0).sum())
        mmcv.imwrite(bev_img, f'vis/{i}_bev.jpg')


        # ori_img = img[0][0].permute(1,2,0).numpy()*255
        ori_img = x[0][0].permute(1,2,0).cpu().numpy()
        # ori_img = ori_img*std + mean
        ori_img = np.array(ori_img, dtype=np.uint8)
        mmcv.imwrite(ori_img, f'vis/{i}_img.jpg')
        print(ori_img.shape)
        # break


        points = data['points'].data
        # points = points[:, :2]
        # plt.scatter(points[:, 0], points[:, 1])
        # plt.savefig('debug_img/points.jpg')
        voxel_layer = Voxelization(**cfg.model.pts_voxel_layer)
        sparse_shape=[41, 1024, 1024]
        voxels, num_points, coors_batch = voxelize([points], voxel_layer)
        print(voxels.size())
        voxels = voxels[:, 0, 2:3]
        batch_size = coors_batch[-1, 0] + 1
        pts_2d = SparseConvTensor(
                voxels,
                coors_batch.int(),
                sparse_shape,
                batch_size
            ).dense()
        print(pts_2d.size())
        pts_2d = pts_2d[0][0].permute(1,2,0)
        # pts_2d[pts_2d!=0] = pts_2d[pts_2d!=0] - pts_2d.min() + 1
        # pts_2d = pts_2d/pts_2d.max()
        pts_2d[pts_2d!=0] = 1
        # print(pts_2d.size(), pts_2d.max(), pts_2d.min())
        pts_2d = pts_2d.sum(dim=-1)
        # print(pts_2d.size(), pts_2d.max(), pts_2d.min())
        pts_2d = torch.clip(torch.floor(pts_2d), 0, 1)
        pts_2d = pts_2d.numpy()*255
        pts_2d = np.array(pts_2d, dtype=np.uint8)
        mmcv.imwrite(pts_2d, f'vis/{i}_points.jpg')

        # plt.figure(figsize=(50, 50))
        # ax = plt.gca()
        # if points.shape[1]<3:
        #     ax.scatter(points[:, 1], points[:, 0], s=0.5, c='b', alpha=0.5)
        # else:
        #     ax.scatter(points[:, 1], points[:, 0], s=0.5, c=points[:, 2], alpha=0.5)
        # plt.savefig('debug_img/points.jpg')


def img_to_bev(input):
    D = 59
    x, rots, trans, intrins, post_rots, post_trans, depth_gt = input
    rots = rots.unsqueeze(dim=0)
    trans = trans.unsqueeze(dim=0)
    intrins = intrins.unsqueeze(dim=0)
    post_rots = post_rots.unsqueeze(dim=0)
    post_trans = post_trans.unsqueeze(dim=0)
    # print(depth_gt.size())
    # print(x.size())
    depth_gt = depth_gt.unsqueeze(dim=0)
    B, N, H, W = depth_gt.shape
    depth_gt = (depth_gt - 1)/1
    depth_gt = torch.clip(torch.floor(depth_gt), 0, D).to(torch.long)
    depth_gt_logit = F.one_hot(depth_gt.reshape(-1),
                                num_classes=D)
    depth_gt_logit = depth_gt_logit.reshape(B, N, H, W, D).permute(
        0, 1, 4, 2, 3).to(torch.float32)
    depth_gt_logit = depth_gt_logit[0]
    # depth_gt_logit = depth_gt_logit.view(B*N, D, H, W)

    x = x.unsqueeze(dim=0)
    B, N, C, H, W = x.shape
    x = x.view(B * N, C, H, W)

    volume = depth_gt_logit.unsqueeze(1) * x.unsqueeze(2)
    volume = volume.view(B, N, 3, D, H, W)
    volume = volume.permute(0, 1, 3, 4, 5, 2)
    
    geom = get_geometry(rots, trans, intrins, post_rots, post_trans)
    # print(geom.size(), volume.size())
    # print(geom.__class__, volume.__class__)
    bev_feat = voxel_pooling(geom.cuda(), volume.cuda())
    return bev_feat


def lidar_to_img(
    img_metas,
    img,
    points,
):
    
    img_feat, rots, trans, intrins, post_rots, post_trans, _ = img
    img_feats = img_feat.permute(0, 2, 3, 1)*255
    # print(img_feats.size())
    for i in range(img_feats.size(0)):
        img_feat = img_feats[i]
        print(img_feat.size())
        lidar2cam_r = torch.inverse(rots[i])
        lidar2cam_t = trans[i] @ lidar2cam_r.t()
        lidar2cam_rt = torch.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.t()
        lidar2cam_rt[3, :3] = -lidar2cam_t
        intrinsic = intrins[i]
        viewpad = torch.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        lidar2img_rt = (viewpad @ lidar2cam_rt.t())

        num_points = points.shape[0]
        pts_4d = torch.cat([points, points.new_ones(size=(num_points, 1))], dim=-1)
        pts_2d = pts_4d @ lidar2img_rt.t()

        pts_2d_cam_z = (pts_4d @ lidar2cam_rt)[:,2]

        pts_2d[:, 2] = torch.clamp(pts_2d[:, 2], min=1e-5)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]

        pts_2d[:, 2] = pts_2d_cam_z
        # print(post_trans[0].size())
        pts_2d =  post_rots[i] @ pts_2d[:,:3].t()
        # print(pts_2d.size())
        pts_2d = pts_2d + post_trans[i].view(3,1)
        pts_2d = pts_2d.t()
        print(pts_2d.size())

        h, w = img_feat.size()[:2]
        mask = (pts_2d_cam_z>0) & (pts_2d[:, 0] > 0) & (pts_2d[:, 0] < w) & (pts_2d[:, 1] > 0) & (pts_2d[:, 1] < h)
        pts_2d = pts_2d[mask, :]
        # embed()
        img_feat = img_feat.numpy()
        img_feat = np.array(img_feat, dtype=np.uint8)
        draw(img_feat, pts_2d, './debug_img/debug_{}.jpg'.format(i))


        # return points



def draw(img, pts_uv_filtered, path):
    depth = torch.pow(pts_uv_filtered[:, 2], 0.5).numpy()
    pts_uv_filtered = torch.floor(pts_uv_filtered)
    pts_uv_filtered = pts_uv_filtered.numpy().astype(np.uint)
    # embed()
    cur_rgb = img[pts_uv_filtered[:, 1], pts_uv_filtered[:, 0], :].astype(np.uint8)
    # depth -= depth.min()
    depth = depth/depth.max()
    depth *= 255
    print(depth.max(), depth.min())
    gray_values = np.arange(256, dtype=np.uint8)
    color_values = cv2.applyColorMap(gray_values, cv2.COLORMAP_JET).reshape(256, 3)
    color_values = np.array(color_values).tolist()
    # embed()

    img_test = np.zeros_like(img, dtype=np.uint8)
    img_test = np.ascontiguousarray(img_test)
    img_with_pts = deepcopy(img)
    img_with_pts = np.array(img_with_pts, dtype=np.uint8)
    img_with_pts = np.ascontiguousarray(img_with_pts)
    # print(img_test.shape)
    for i in range(pts_uv_filtered.shape[0]):
        # print(i)
        color = (int(cur_rgb[i, 0]), int(cur_rgb[i, 1]), int(cur_rgb[i, 2]))
        p = (int(pts_uv_filtered[i, 0]), int(pts_uv_filtered[i, 1]))
        # print(p)
        # print(color)
        cv2.circle(img_test, p, 1, color=color)
        # color2 = (int(pts_uv_filtered[i, 2]), int(pts_uv_filtered[i, 2]), int(pts_uv_filtered[i, 2]))
        # color2 = cv2.COLORMAP_JET[pts_uv_filtered[i, 2]]
        cv2.circle(img_with_pts, p, 1, color=color_values[int(depth[i])])
    img_show = np.concatenate([img, img_test, img_with_pts], axis=0)
    # img_show = cv2.resize(img_show, (800, 900))
    mmcv.imwrite(img_show, path)





if __name__ == '__main__':
    main()
