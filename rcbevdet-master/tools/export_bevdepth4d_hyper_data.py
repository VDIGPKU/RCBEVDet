import argparse
import pickle
import logging

import torch.onnx
from mmcv import Config
from mmdeploy.backend.tensorrt.utils import save, search_cuda_version

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg
import torch.nn.functional as F
import os

from typing import Dict, Optional, Sequence, Union

import h5py
import mmcv
import numpy as np
import onnx
import pycuda.driver as cuda
import tensorrt as trt
import torch
import tqdm
from mmcv.runner import load_checkpoint
from mmdeploy.apis.core import no_mp
from mmdeploy.backend.tensorrt.calib_utils import HDF5Calibrator
from mmdeploy.backend.tensorrt.init_plugins import load_tensorrt_plugin
from mmdeploy.utils import load_config
from packaging import version
from torch.utils.data import DataLoader

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
from tools.misc.fuse_conv_bn import fuse_module
from PIL import Image


logging.getLogger().setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Deploy BEVDet with Tensorrt')
    parser.add_argument('config', help='deploy config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('work_dir', help='work dir to save file')
    parser.add_argument(
        '--prefix', default='bevdepth4d', help='prefix of the save file name')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args
    

def main():
    args = parse_args()
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    model_prefix = args.prefix
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.type = cfg.model.type + 'TRT'

    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [0]

    # build the dataloader
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # BEV size
    assert model.img_view_transformer.grid_size[0] == 128
    assert model.img_view_transformer.grid_size[1] == 128
    assert model.img_view_transformer.grid_size[2] == 1
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model_prefix = model_prefix + '_fuse'
        model = fuse_module(model)
    model.cuda()
    model.eval()

    min_num_points_shape = 1e9
    max_num_points_shape = 0
    min_num_intervals_shape = 1e9
    max_num_intervals_shape = 0
    init = False

    data_dict = {}
    data_keys = ['imgs', 'ranks_depth', 'ranks_feat', 'ranks_bev', 'interval_starts', 'interval_lengths', 'mlp_input', 'feat_prev']
    for key in data_keys:
        data_dict[key] = []
        
    total_samples = len(data_loader)

    prog_bar = mmcv.ProgressBar(total_samples)

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            inputs = [t.cuda() for t in data['img_inputs'][0]]
            imgs, mlp_input_list, metas_list, \
            sensor2keyegos, ego2globals, bda = model.get_bev_pool_input(inputs) # bda = torch.Size([3, 3]), 不需要处理
            imgs = torch.cat(imgs, dim=0) # 注意，使用img的时候需要squeeze(0)而使用mlp_input的时候不需要 这是二者的差别. torch.Size([1, 6, 3, 256, 704]) * 9
            mlp_input = torch.cat(mlp_input_list, dim=0) # torch.Size([1, 6, 27]) * 9
            sensor2keyegos = torch.cat(sensor2keyegos, dim=0) # torch.Size([1, 6, 4, 4]) * 9
            ego2globals = torch.cat(ego2globals, dim=0) # torch.Size([1, 6, 4, 4]) * 9
            ranks_depth = []
            ranks_feat = []
            ranks_bev = []
            interval_starts = []
            interval_lengths = []
            
            for metas in metas_list:
                min_num_points_shape = min(min_num_points_shape, metas[1].shape[0])
                max_num_points_shape = max(max_num_points_shape, metas[1].shape[0])
                min_num_intervals_shape = min(min_num_intervals_shape, metas[3].shape[0])
                max_num_intervals_shape = max(max_num_intervals_shape, metas[3].shape[0])
                # 不能把他们合并到一起，因为会fliter掉box外的点，所以这些点的个数不一定是相同的
                ranks_depth.append(metas[1].int().contiguous())
                ranks_feat.append(metas[2].int().contiguous())
                ranks_bev.append(metas[0].int().contiguous())
                interval_starts.append(metas[3].int().contiguous())
                interval_lengths.append(metas[4].int().contiguous())
            # 得到过去bev feat
            # sensor2keyegos_prev = torch.Size([8, 6, 4, 4]), ego2globals_prev = torch.Size([8, 6, 4, 4]), bda_curr = torch.Size([8, 3, 3])
            # feat_prev = torch.Size([8, 80, 128, 128]), sensor2keyegos_curr = torch.Size([8, 6, 4, 4]), ego2globals_curr = torch.Size([8, 6, 4, 4])
            feat_prev, sensor2keyegos_curr, ego2globals_curr, \
                sensor2keyegos_prev, ego2globals_prev, bda_curr = model.get_bev_feat_sequential(imgs, ranks_depth, ranks_feat, ranks_bev,\
                    interval_starts, interval_lengths, mlp_input, ego2globals, sensor2keyegos, bda)
            # shift feature        
            # grid = model.gen_grid(feat_prev, [sensor2keyegos_curr, sensor2keyegos_prev], bda, bda_adj=None, flag=True)
            # feat_prev = F.grid_sample(feat_prev, grid.to(feat_prev.dtype), align_corners=True)

            imgs = imgs[0:1,:,:,:,:].squeeze(0).float().contiguous()
            ranks_depth = ranks_depth[0]
            ranks_feat = ranks_feat[0]
            ranks_bev = ranks_bev[0]
            interval_starts = interval_starts[0]
            interval_lengths = interval_lengths[0]
            mlp_input = mlp_input[0:1,:,:].float().contiguous()
            
            data_dict['imgs'].append(imgs.cpu())
            data_dict['ranks_depth'].append(ranks_depth.cpu())
            data_dict['ranks_feat'].append(ranks_feat.cpu())
            data_dict['ranks_bev'].append(ranks_bev.cpu())
            data_dict['interval_starts'].append(interval_starts.cpu())
            data_dict['interval_lengths'].append(interval_lengths.cpu())
            data_dict['mlp_input'].append(mlp_input.cpu())
            data_dict['feat_prev'].append(feat_prev.cpu())

            prog_bar.update()

    print('\nSave Hyper Data...')
    # convert to tensorrt
    num_points_shape = ranks_bev.shape
    num_intervals_shape = interval_starts.shape
    imgs_shape = imgs.shape
    mlp_input_shape = mlp_input.shape
    feat_prev_shape = feat_prev.shape
    input_shapes = dict(
        imgs=dict(
            min_shape=imgs_shape, 
            opt_shape=imgs_shape, 
            max_shape=imgs_shape),
        ranks_depth=dict(
            min_shape=[min_num_points_shape],
            opt_shape=[(min_num_points_shape+max_num_points_shape)//2],
            max_shape=[max_num_points_shape]),
        ranks_feat=dict(
            min_shape=[min_num_points_shape],
            opt_shape=[(min_num_points_shape+max_num_points_shape)//2],
            max_shape=[max_num_points_shape]),
        ranks_bev=dict(
            min_shape=[min_num_points_shape],
            opt_shape=[(min_num_points_shape+max_num_points_shape)//2],
            max_shape=[max_num_points_shape]),
        interval_starts=dict(
            min_shape=[min_num_intervals_shape],
            opt_shape=[(min_num_intervals_shape+max_num_intervals_shape)//2],
            max_shape=[max_num_intervals_shape]),
        interval_lengths=dict(
            min_shape=[min_num_intervals_shape],
            opt_shape=[(min_num_intervals_shape+max_num_intervals_shape)//2],
            max_shape=[max_num_intervals_shape]),
        mlp_input=dict(
            min_shape=mlp_input_shape,
            opt_shape=mlp_input_shape,
            max_shape=mlp_input_shape),
        feat_prev=dict(
            min_shape=feat_prev_shape,
            opt_shape=feat_prev_shape,
            max_shape=feat_prev_shape),
        )
    data_dict['input_shapes'] = input_shapes
    if args.fuse_conv_bn:
         data_path = os.path.join(args.work_dir, args.prefix + '_hyper_data' + '_fuse' + '.pkl')
    else:
        data_path = os.path.join(args.work_dir, args.prefix + '_hyper_data' + '.pkl')
    with open(data_path,'wb') as f:
        try:
            pickle.dump(data_dict, f)
            f.close()
        except ValueError:
            logging.error(f"{data_path} file write failed.")

    print('Done!')


if __name__ == '__main__':

    main()
