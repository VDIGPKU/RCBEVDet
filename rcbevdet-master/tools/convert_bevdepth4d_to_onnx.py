'''
please run export_bevdepth4d_hyper_data.py to generate hyper data first
python tools/convert_bevdepth4d_to_onnx.py configs/bevperception/detsegVAD-256x704-r50-BEV128-9kf-depth-circle60e-det.py work_dirs/detsegVAD-256x704-r50-BEV128-9kf-depth-circle60e/epoch_60.pth mmdeploy/tensorrt8522/detsegVAD-256x704-r50-BEV128-9kf-depth-circle60ev3/ --fuse-conv-bn --fp16 --int8
'''

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
        '--hyper_data_path', 
        '--h',
        type=str,
        default='bevdepth4d_hyper_data_fuse.pkl',
        help='path to hyper data')
    parser.add_argument(
        '--prefix', default='bevdepth4d', help='prefix of the save file name')
    parser.add_argument(
        '--fp16', action='store_true', help='Whether to use tensorrt fp16')
    parser.add_argument(
        '--int8', action='store_true', help='Whether to use tensorrt int8')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed. please stay consistent with param from export_bevdepth4d_hyper_data')
    args = parser.parse_args()
    return args


def get_plugin_names():
    return [pc.name for pc in trt.get_plugin_registry().plugin_creator_list]


def main():
    args = parse_args()
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    load_tensorrt_plugin()
    assert 'bev_pool_v2' in get_plugin_names(), \
        'bev_pool_v2 is not in the plugin list of tensorrt, ' \
        'please install mmdeploy from ' \
        'https://github.com/HuangJunJie2017/mmdeploy.git'

    if args.int8:
        assert args.fp16
    model_prefix = args.prefix
    if args.int8:
        model_prefix = model_prefix + '_int8'
    elif args.fp16:
        model_prefix = model_prefix + '_fp16'
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.type = cfg.model.type + 'TRT'

    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [0]


    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    assert model.img_view_transformer.grid_size[0] == 128
    assert model.img_view_transformer.grid_size[1] == 128
    assert model.img_view_transformer.grid_size[2] == 1
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model_prefix = model_prefix + '_fuse'
        model = fuse_module(model)
    model.cuda()
    model.eval()

    # load hyper data from path, only need one data from hyper data
    hyper_data_path = os.path.join(args.work_dir, args.hyper_data_path)
    try:
        with open(hyper_data_path,'rb') as f:
            data_dict = pickle.load(f)
    except FileNotFoundError:
        logging.error(f"{hyper_data_path}no such hyper data file.")
    
    imgs = data_dict['imgs'][0].cuda()
    ranks_depth = data_dict['ranks_depth'][0].cuda()
    ranks_feat = data_dict['ranks_feat'][0].cuda()
    ranks_bev = data_dict['ranks_bev'][0].cuda()
    interval_starts = data_dict['interval_starts'][0].cuda()
    interval_lengths = data_dict['interval_lengths'][0].cuda()
    mlp_input = data_dict['mlp_input'][0].cuda()
    feat_prev = data_dict['feat_prev'][0].cuda()

    # fix to orin
    if os.path.exists(args.work_dir + model_prefix + '.onnx'):
        print("Reading onnx from file {}".format(args.work_dir + model_prefix + '.onnx'))
    else:
        with torch.no_grad():
            torch.onnx.export(
                model,
                (imgs, ranks_depth, ranks_feat, ranks_bev, interval_starts, interval_lengths, \
                mlp_input, feat_prev.float().contiguous()),
                args.work_dir + model_prefix + '.onnx',
                opset_version=11, # grid_sampler函数在onnx opset 16 中才被加入
                input_names=[
                    'imgs', 'ranks_depth', 'ranks_feat', 'ranks_bev',
                    'interval_starts', 'interval_lengths', 'mlp_input',
                    'feat_prev',
                ],
                output_names=[f'output_{j}' for j in
                                range(6 * len(model.pts_bbox_head.task_heads))] + ['cur_feat'],
                dynamic_axes={'ranks_depth':{0: 'num_points_shape'},
                                'ranks_feat':{0: 'num_points_shape'},
                                'ranks_bev':{0: 'num_points_shape'},
                                'interval_starts':{0: 'num_intervals_shape'},
                                'interval_lengths':{0: 'num_intervals_shape'}})    # check onnx model
    
    onnx_model = onnx.load(args.work_dir + model_prefix + '.onnx')
    try:
        onnx.checker.check_model(onnx_model)
    except Exception:
        print('ONNX Model Incorrect', Exception)
    else:
        print('ONNX Model Correct')


if __name__ == '__main__':

    main()
