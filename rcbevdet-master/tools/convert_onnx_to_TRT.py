'''
please run export_bevdepth4d_hyper_data.py and convert_bevdepth4d_to_onnx.py to generate hyper data and onnx model first
ython ./tools/convert_onnx_to_TRT.py configs/bevperception/detsegVAD-256x704-r50-BEV128-9kf-depth-circle60e-det.py work_dirs/detsegVAD-256x704-r50-BEV128-9kf-depth-circle60e/epoch_60.pth mmdeploy/tensorrt8522/detsegVAD-256x704-r50-BEV128-9kf-depth-circle60ev3/ --fuse-conv-bn --fp16 --int8
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


TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) # this is necessary for generate dynamic range input
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
        '--onnx_path', 
        '--o',
        type=str,
        default='bevdepth4d_int8_fuse.onnx',
        help='path to onnx model')
    parser.add_argument(
        "--cache_file",
        "--c",
        type=str,
        default="bevdepth4d_int8_fuse.cache",
        help="calib file path",
    )
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
        'the inference speed. please stay consistent with param from export_bevdepth4d_hyper_data and convert_bevdepth4d_to_onnx')
    args = parser.parse_args()
    return args


def get_plugin_names():
    return [pc.name for pc in trt.get_plugin_registry().plugin_creator_list]


def create_dataset(data_path):
    try:
        with open(data_path,'rb') as f:
            data_dict = pickle.load(f)
            dataset = []
            for idx in range(len(data_dict['imgs'])):
                dataset.append(dict(
                    imgs=data_dict['imgs'][idx], 
                    ranks_depth=data_dict['ranks_depth'][idx],
                    ranks_feat=data_dict['ranks_feat'][idx],
                    ranks_bev=data_dict['ranks_bev'][idx],
                    interval_starts=data_dict['interval_starts'][idx],
                    interval_lengths=data_dict['interval_lengths'][idx],
                    mlp_input=data_dict['mlp_input'][idx],
                    feat_prev=data_dict['feat_prev'][idx]
                    ))
            return dataset, data_dict['input_shapes']

    except:
        logging.error(f"{data_path}no such hyper data file.")
        raise FileNotFoundError
        
    


class MyCalibratorV2(trt.IInt8EntropyCalibrator2): # can change Int8Entropy to other Calibrator type
    def __init__(self, data_path, cache_file, batch_size=1, data_length=500):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        # currently only support batch_size=1
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.batch_size = batch_size
        self.current_index = 0

        self.dataset, self.dynamic_ranges = create_dataset(data_path)
        self.data_length = min(data_length, len(self.dataset) // batch_size)
        self.dataiter = iter(self.dataset)
        self.buffers = {}

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data_length:
            logging.info(f"{self.data_length} calibration data have been used, return")
            return None
        try:
            data = next(self.dataiter)
        except StopIteration:
            logging.info(f"all calibration data has been used, return")
            return None
        ret = []
        for name in names:
            name_data = data[name].numpy()
            name_data_cuda_ptr = cuda.mem_alloc(name_data.nbytes)
            cuda.memcpy_htod(name_data_cuda_ptr, np.ascontiguousarray(name_data))
            self.buffers[name] = name_data_cuda_ptr
            ret.append(self.buffers[name])
        
        self.current_index += self.batch_size
        return ret

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def get_algorithm(self):
        return trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2


def onnx2trt(onnx_file_path, engine_file_path, calib=None, fp16=False):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        logging.info("Loading ONNX file from path {}...".format(onnx_file_path))
        with open(onnx_file_path, "rb") as model:
            logging.info("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                logging.error("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    logging.info(parser.get_error(error))
                return None
        profile = builder.create_optimization_profile()
        inputs = [network.get_input(i) for i in range(network.num_inputs)]

        if calib.dynamic_ranges: # set dynamic shape
            for i in inputs:
                profile.set_shape(i.name, calib.dynamic_ranges[i.name]['min_shape'], calib.dynamic_ranges[i.name]['opt_shape'], calib.dynamic_ranges[i.name]['max_shape'])
        else:
            for i in inputs:
                profile.set_shape(i.name, i.shape, i.shape, i.shape) 

        config.add_optimization_profile(profile)
        # last_layer = network.get_layer(network.num_layers - 1)
        # network.mark_output(last_layer.get_output(0))
        # config.set_memory_pool_limit = 4 * (1 << 30)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        config.int8_calibrator = calib
        config.clear_flag(trt.BuilderFlag.TF32)
        config.set_flag(trt.BuilderFlag.FP16) if fp16 else config.clear_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        logging.info("Completed parsing of ONNX file")
        logging.info(f"Building an engine from file {onnx_file_path}; this may take a while...")

        plan = builder.build_serialized_network(network, config)
        logging.info("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(plan)


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

    calib = MyCalibratorV2(os.path.join(args.work_dir, args.hyper_data_path), os.path.join(args.work_dir, args.cache_file))
    onnx2trt(os.path.join(args.work_dir, args.onnx_path), os.path.join(args.work_dir, args.onnx_path.replace('onnx', 'engine')), calib=calib, fp16=args.fp16)

if __name__ == '__main__':

    main()
