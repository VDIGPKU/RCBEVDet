import argparse

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
import common
def get_engine(onnx_model: Union[str, onnx.ModelProto],
              output_file_prefix: str,
              input_shapes: Dict[str, Sequence[int]],
              max_workspace_size: int = 0,
              fp16_mode: bool = False,
              int8_mode: bool = False,
              int8_param: Optional[dict] = None,
              device_id: int = 0,
              log_level: trt.Logger.Severity = trt.Logger.ERROR,
              **kwargs) -> trt.ICudaEngine:
    """
    Create a tensorrt engine from ONNX.
    Modified from TensorRT/samples/python/yolov3_onnx/onnx_to_tensorrt.py/get_engine
    Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it.
    """

    def build_engine(onnx_file_path, engine_file_path):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
            config.max_workspace_size = 1 << 32 # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # network.get_input(0).shape = [1, 3, 608, 608]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


# # Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# #
# class BEVDepth4DEntropyCalibrator(trt.IInt8EntropyCalibrator2):
#     def __init__(self, training_data, cache_file, batch_size=64):
#         # Whenever you specify a custom constructor for a TensorRT class,
#         # you MUST call the constructor of the parent explicitly.
#         trt.IInt8EntropyCalibrator2.__init__(self)

#         self.cache_file = cache_file

#         # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
#         self.data = load_mnist_data(training_data)
#         self.batch_size = batch_size
#         self.current_index = 0

#         # Allocate enough memory for a whole batch.
#         self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

#     def get_batch_size(self):
#         return self.batch_size

#     # TensorRT passes along the names of the engine bindings to the get_batch function.
#     # You don't necessarily have to use them, but they can be useful to understand the order of
#     # the inputs. The bindings list is expected to have the same ordering as 'names'.
#     def get_batch(self, names):
#         """Get batch data."""
#         if self.count < self.dataset_length:
#             # if self.count % 100 == 0:
#             print('%d/%d' % (self.count, self.dataset_length))
#             ret = []
#             for name in names:
#                 input_group = self.calib_data[name]
#                 if name == 'imgs':
#                     data_np = input_group[str(self.count)][...].astype(
#                         np.float32)
#                 else:
#                     data_np = input_group[str(self.count)][...].astype(
#                         np.int32)
#                 # tile the tensor so we can keep the same distribute
#                 opt_shape = self.input_shapes[name]['opt_shape']
#                 data_shape = data_np.shape

#                 reps = [
#                     int(np.ceil(opt_s / data_s))
#                     for opt_s, data_s in zip(opt_shape, data_shape)
#                 ]

#                 data_np = np.tile(data_np, reps)

#                 slice_list = tuple(slice(0, end) for end in opt_shape)
#                 data_np = data_np[slice_list]

#                 data_np_cuda_ptr = cuda.mem_alloc(data_np.nbytes)
#                 cuda.memcpy_htod(data_np_cuda_ptr,
#                                  np.ascontiguousarray(data_np))
#                 self.buffers[name] = data_np_cuda_ptr

#                 ret.append(self.buffers[name])
#             self.count += 1
#             return ret
#         else:
#             return None


#     def read_calibration_cache(self):
#         # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
#         if os.path.exists(self.cache_file):
#             with open(self.cache_file, "rb") as f:
#                 return f.read()

#     def write_calibration_cache(self, cache):
#         with open(self.cache_file, "wb") as f:
#             f.write(cache)



class HDF5CalibratorBEVDepth4D(HDF5Calibrator):

    def get_batch(self, names: Sequence[str], **kwargs) -> list:
        """Get batch data."""
        if self.count < self.dataset_length:
            # if self.count % 100 == 0:
            print('%d/%d' % (self.count, self.dataset_length))
            ret = []
            for name in names:
                input_group = self.calib_data[name]
                if name == 'imgs':
                    data_np = input_group[str(self.count)][...].astype(
                        np.float32)
                else:
                    data_np = input_group[str(self.count)][...].astype(
                        np.int32)
                # tile the tensor so we can keep the same distribute
                opt_shape = self.input_shapes[name]['opt_shape']
                data_shape = data_np.shape

                reps = [
                    int(np.ceil(opt_s / data_s))
                    for opt_s, data_s in zip(opt_shape, data_shape)
                ]

                data_np = np.tile(data_np, reps)

                slice_list = tuple(slice(0, end) for end in opt_shape)
                data_np = data_np[slice_list]

                data_np_cuda_ptr = cuda.mem_alloc(data_np.nbytes)
                cuda.memcpy_htod(data_np_cuda_ptr,
                                 np.ascontiguousarray(data_np))
                self.buffers[name] = data_np_cuda_ptr

                ret.append(self.buffers[name])
            self.count += 1
            return ret
        else:
            return None


def parse_args():
    parser = argparse.ArgumentParser(description='Deploy BEVDet with Tensorrt')
    parser.add_argument('config', help='deploy config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('work_dir', help='work dir to save file')
    parser.add_argument(
        '--prefix', default='bevdet', help='prefix of the save file name')
    parser.add_argument(
        '--fp16', action='store_true', help='Whether to use tensorrt fp16')
    parser.add_argument(
        '--int8', action='store_true', help='Whether to use tensorrt int8')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args


def get_plugin_names():
    return [pc.name for pc in trt.get_plugin_registry().plugin_creator_list]


def create_calib_input_data_impl(calib_file: str,
                                 dataloader: DataLoader,
                                 model_partition: bool = False,
                                 metas: list = []) -> None:
    with h5py.File(calib_file, mode='w') as file:
        calib_data_group = file.create_group('calib_data')
        assert not model_partition
        # create end2end group
        input_data_group = calib_data_group.create_group('end2end')
        input_group_img = input_data_group.create_group('imgs')
        input_keys = [
            'ranks_bev', 'ranks_depth', 'ranks_feat', 'interval_starts',
            'interval_lengths', 'mlp_input', 'feat_prev',
        ]
        input_groups = []
        for input_key in input_keys:
            input_groups.append(input_data_group.create_group(input_key))
        metas = [
            metas[i].int().detach().cpu().numpy() for i in range(len(metas))
        ]
        for data_id, input_data in enumerate(tqdm.tqdm(dataloader)):
            # save end2end data
            input_tensor = input_data['img_inputs'][0][0] 
            # 这里得到的结果是[frames * N, C, H, W]，要修改成我们需要的当前帧[N, C, H, W]
            input_tensor = input_tensor.view(1, 6, 9, 3, 256, 704) # B=1, N=6. self.num_frame=9, C=3, H=256, W=704
            input_tensor = torch.split(input_tensor, 1, 2)
            input_tensor = [t.squeeze(2) for t in input_tensor]
            input_tensor = input_tensor[0].squeeze(0)
            input_ndarray = input_tensor.detach().cpu().numpy()
            # print(input_ndarray.shape, input_ndarray.dtype)
            input_group_img.create_dataset(
                str(data_id),
                shape=input_ndarray.shape,
                compression='gzip',
                compression_opts=4,
                data=input_ndarray)
            for kid, input_key in enumerate(input_keys):
                input_groups[kid].create_dataset(
                    str(data_id),
                    shape=metas[kid].shape,
                    compression='gzip',
                    compression_opts=4,
                    data=metas[kid])
            file.flush()


def create_calib_input_data(calib_file: str,
                            deploy_cfg: Union[str, mmcv.Config],
                            model_cfg: Union[str, mmcv.Config],
                            model_checkpoint: Optional[str] = None,
                            dataset_cfg: Optional[Union[str,
                                                        mmcv.Config]] = None,
                            dataset_type: str = 'val',
                            device: str = 'cpu',
                            metas: list = [None]) -> None:
    """Create dataset for post-training quantization.

    Args:
        calib_file (str): The output calibration data file.
        deploy_cfg (str | mmcv.Config): Deployment config file or
            Config object.
        model_cfg (str | mmcv.Config): Model config file or Config object.
        model_checkpoint (str): A checkpoint path of PyTorch model,
            defaults to `None`.
        dataset_cfg (Optional[Union[str, mmcv.Config]], optional): Model
            config to provide calibration dataset. If none, use `model_cfg`
            as the dataset config. Defaults to None.
        dataset_type (str, optional): The dataset type. Defaults to 'val'.
        device (str, optional): Device to create dataset. Defaults to 'cpu'.
    """
    with no_mp():
        if dataset_cfg is None:
            dataset_cfg = model_cfg

        # load cfg if necessary
        deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

        if dataset_cfg is None:
            dataset_cfg = model_cfg

        # load dataset_cfg if necessary
        dataset_cfg = load_config(dataset_cfg)[0]

        from mmdeploy.apis.utils import build_task_processor
        task_processor = build_task_processor(model_cfg, deploy_cfg, device)

        dataset = task_processor.build_dataset(dataset_cfg, dataset_type)

        dataloader = task_processor.build_dataloader(
            dataset, 1, 1, dist=False, shuffle=False)

        create_calib_input_data_impl(
            calib_file, dataloader, model_partition=False, metas=metas)


def create_calib_input_datav2(
                            deploy_cfg: Union[str, mmcv.Config],
                            model_cfg: Union[str, mmcv.Config],
                            model,
                            dataset_cfg: Optional[Union[str,
                                                        mmcv.Config]] = None,
                            dataset_type: str = 'val',
                            device: str = 'cpu') -> None:
    """Create dataset for post-training quantization.
    Args:
        deploy_cfg (str | mmcv.Config): Deployment config file or
            Config object.
        model_cfg (str | mmcv.Config): Model config file or Config object.
        dataset_cfg (Optional[Union[str, mmcv.Config]], optional): Model
            config to provide calibration dataset. If none, use `model_cfg`
            as the dataset config. Defaults to None.
        dataset_type (str, optional): The dataset type. Defaults to 'val'.
        device (str, optional): Device to create dataset. Defaults to 'cpu'.
    """
    with no_mp():
        if dataset_cfg is None:
            dataset_cfg = model_cfg

        # load cfg if necessary
        deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

        if dataset_cfg is None:
            dataset_cfg = model_cfg

        # load dataset_cfg if necessary
        dataset_cfg = load_config(dataset_cfg)[0]

        from mmdeploy.apis.utils import build_task_processor
        task_processor = build_task_processor(model_cfg, deploy_cfg, device)

        dataset = task_processor.build_dataset(dataset_cfg, dataset_type)

        dataloader = task_processor.build_dataloader(
            dataset, 1, 1, dist=False, shuffle=False)
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                print('idx: ', i)
                inputs = [t.cuda() for t in data['img_inputs'][0]]
                imgs, mlp_input_list, metas_list, \
                sensor2keyegos, ego2globals, bda = model.get_bev_pool_input(inputs) # bda = torch.Size([3, 3]), 不需要处理
                imgs = torch.cat(imgs, dim=0) # 注意，使用img的时候需要squeeze(0)而使用mlp_input的时候不需要 这是二者的差别. torch.Size([1, 6, 3, 256, 704]) * 9
                mlp_input = torch.cat(mlp_input_list, dim=0) # torch.Size([1, 6, 27]) * 9
                print('idx: ', i)
                print('mlp_input: ', mlp_input)
                sensor2keyegos = torch.cat(sensor2keyegos, dim=0) # torch.Size([1, 6, 4, 4]) * 9
                ego2globals = torch.cat(ego2globals, dim=0) # torch.Size([1, 6, 4, 4]) * 9
                ranks_depth = []
                ranks_feat = []
                ranks_bev = []
                interval_starts = []
                interval_lengths = []
                for metas in metas_list:
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

                imgs = imgs[0:1,:,:,:,:].squeeze(0).float().contiguous().detach().cpu().numpy()
                ranks_depth = ranks_depth[0].int().detach().cpu().numpy()
                ranks_feat = ranks_feat[0].int().detach().cpu().numpy()
                ranks_bev = ranks_bev[0].int().detach().cpu().numpy()
                interval_starts = interval_starts[0].int().detach().cpu().numpy()
                interval_lengths = interval_lengths[0].int().detach().cpu().numpy()
                mlp_input = mlp_input[0:1,:,:].float().contiguous().int().detach().cpu().numpy()
                metas_list = [imgs, ranks_depth, ranks_feat, ranks_bev, interval_starts, interval_lengths, mlp_input, feat_prev]
                one_batch_data = {}
                metas_name = ['imgs', 'ranks_depth', 'ranks_feat', 'ranks_bev', 'interval_starts', 'interval_lengths', 'mlp_input', 'feat_prev'],
                for i in range(len(metas_list)):
                    one_batch_data[metas_name[0][i]] = metas_list[i] # 不知道为什么这里metas_name老是会自动变成tuple。。。
                yield one_batch_data  # 以生成器 generator 的形式输出数据


def from_onnx(onnx_model: Union[str, onnx.ModelProto],
              output_file_prefix: str,
              input_shapes: Dict[str, Sequence[int]],
              max_workspace_size: int = 0,
              fp16_mode: bool = False,
              int8_mode: bool = False,
              int8_param: Optional[dict] = None,
              device_id: int = 0,
              log_level: trt.Logger.Severity = trt.Logger.ERROR,
              **kwargs) -> trt.ICudaEngine:
    """Create a tensorrt engine from ONNX.

    Modified from mmdeploy.backend.tensorrt.utils.from_onnx
    """

    import os
    old_cuda_device = os.environ.get('CUDA_DEVICE', None)
    os.environ['CUDA_DEVICE'] = str(device_id)
    import pycuda.autoinit  # noqa:F401
    if old_cuda_device is not None:
        os.environ['CUDA_DEVICE'] = old_cuda_device
    else:
        os.environ.pop('CUDA_DEVICE')

    load_tensorrt_plugin()
    # create builder and network
    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)

    if isinstance(onnx_model, str):
        onnx_model = onnx.load(onnx_model)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    # config builder
    if version.parse(trt.__version__) < version.parse('8'):
        builder.max_workspace_size = max_workspace_size

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size

    cuda_version = search_cuda_version()
    if cuda_version is not None:
        version_major = int(cuda_version.split('.')[0])
        if version_major < 11:
            # cu11 support cublasLt, so cudnn heuristic tactic should disable CUBLAS_LT # noqa E501
            tactic_source = config.get_tactic_sources() - (
                1 << int(trt.TacticSource.CUBLAS_LT))
            config.set_tactic_sources(tactic_source)

    profile = builder.create_optimization_profile()

    for input_name, param in input_shapes.items():
        min_shape = param['min_shape']
        opt_shape = param['opt_shape']
        max_shape = param['max_shape']
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if fp16_mode:
        if version.parse(trt.__version__) < version.parse('8'):
            builder.fp16_mode = fp16_mode
        config.set_flag(trt.BuilderFlag.FP16)

    if int8_mode:
        config.set_flag(trt.BuilderFlag.INT8)
        assert int8_param is not None
        config.int8_calibrator = HDF5CalibratorBEVDepth4D(
            int8_param['calib_file'],
            input_shapes,
            model_type=int8_param['model_type'],
            device_id=device_id,
            algorithm=int8_param.get(
                'algorithm', trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2))
        if version.parse(trt.__version__) < version.parse('8'):
            builder.int8_mode = int8_mode
            builder.int8_calibrator = config.int8_calibrator
    # create engine
    engine = builder.build_engine(network, config)

    assert engine is not None, 'Failed to create TensorRT engine'

    save(engine, output_file_prefix + '.engine')
    return engine


from polygraphy.backend.trt import Calibrator
def from_onnxv2(onnx_model: Union[str, onnx.ModelProto],
              output_file_prefix: str,
              input_shapes: Dict[str, Sequence[int]],
              deploy_cfg: Union[str, mmcv.Config],
              model_cfg: Union[str, mmcv.Config],
              model,
              max_workspace_size: int = 0,
              fp16_mode: bool = False,
              int8_mode: bool = False,
              device_id: int = 0,
              log_level: trt.Logger.Severity = trt.Logger.ERROR,
              calibration_cache_file: str = None,
              dataset_cfg: Optional[Union[str, mmcv.Config]] = None,
              dataset_type: str = 'val',
              device: str = 'cpu',
              **kwargs) -> trt.ICudaEngine:
    """Create a tensorrt engine from ONNX.
    Modified from mmdeploy.backend.tensorrt.utils.from_onnx
    """

    import os
    old_cuda_device = os.environ.get('CUDA_DEVICE', None)
    os.environ['CUDA_DEVICE'] = str(device_id)
    import pycuda.autoinit  # noqa:F401
    if old_cuda_device is not None:
        os.environ['CUDA_DEVICE'] = old_cuda_device
    else:
        os.environ.pop('CUDA_DEVICE')

    load_tensorrt_plugin()
    # create builder and network
    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)

    if isinstance(onnx_model, str):
        onnx_model = onnx.load(onnx_model)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    # config builder
    if version.parse(trt.__version__) < version.parse('8'):
        builder.max_workspace_size = max_workspace_size

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size

    cuda_version = search_cuda_version()
    if cuda_version is not None:
        version_major = int(cuda_version.split('.')[0])
        if version_major < 11:
            # cu11 support cublasLt, so cudnn heuristic tactic should disable CUBLAS_LT # noqa E501
            tactic_source = config.get_tactic_sources() - (
                1 << int(trt.TacticSource.CUBLAS_LT))
            config.set_tactic_sources(tactic_source)

    profile = builder.create_optimization_profile()

    for input_name, param in input_shapes.items():
        min_shape = param['min_shape']
        opt_shape = param['opt_shape']
        max_shape = param['max_shape']
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if fp16_mode:
        if version.parse(trt.__version__) < version.parse('8'):
            builder.fp16_mode = fp16_mode
        config.set_flag(trt.BuilderFlag.FP16)

    if int8_mode:
        config.set_flag(trt.BuilderFlag.INT8)
        calibrator_class = trt.IInt8EntropyCalibrator2
        config.int8_calibrator = Calibrator(
            BaseClass=calibrator_class,
            data_loader=create_calib_input_datav2(
            deploy_cfg=deploy_cfg,
            model_cfg=model_cfg,
            model=model,
            dataset_cfg=dataset_cfg,
            dataset_type=dataset_type,
            device=device),
        cache=calibration_cache_file)
        if version.parse(trt.__version__) < version.parse('8'):
            builder.int8_mode = int8_mode
            builder.int8_calibrator = config.int8_calibrator
    # create engine
    engine = builder.build_engine(network, config)

    assert engine is not None, 'Failed to create TensorRT engine'

    save(engine, output_file_prefix + '.engine')
    return engine


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

    total_samples = len(data_loader)
    for i, data in enumerate(data_loader):
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
        if i != total_samples-1:
            continue
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
    # convert to tensorrt
    num_points_shape = ranks_bev.shape
    num_intervals_shape = interval_starts.shape
    imgs_shape = imgs.shape
    mlp_input_shape = mlp_input.shape
    feat_prev_shape = feat_prev.shape
    metas = [ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths, mlp_input, feat_prev]
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
    deploy_cfg = dict(
        backend_config=dict(
            type='tensorrt',
            common_config=dict(
                fp16_mode=args.fp16,
                max_workspace_size=1073741824,
                int8_mode=args.int8),
            model_inputs=[dict(input_shapes=input_shapes)]),
        codebase_config=dict(
            type='mmdet3d', task='VoxelDetection', model_type='end2end'))

    # if args.int8:
    #     calib_filename = 'calib_data.h5'
    #     calib_path = os.path.join(args.work_dir, calib_filename)
    #     create_calib_input_data(
    #         calib_path,
    #         deploy_cfg,
    #         args.config,
    #         args.checkpoint,
    #         dataset_cfg=None,
    #         dataset_type='val',
    #         device='cuda:0',
    #         metas=metas)

    # from_onnx(
    #     args.work_dir + model_prefix + '.onnx',
    #     args.work_dir + model_prefix,
    #     fp16_mode=args.fp16,
    #     int8_mode=args.int8,
    #     int8_param=dict(
    #         calib_file=os.path.join(args.work_dir, 'calib_data.h5'),
    #         model_type='end2end'),
    #     max_workspace_size=1 << 30,
    #     input_shapes=input_shapes)
    calibration_cache_file = args.work_dir + model_prefix + '.cache'
    from_onnxv2(
        args.work_dir + model_prefix + '.onnx',
        args.work_dir + model_prefix,
        calibration_cache_file=calibration_cache_file,
        fp16_mode=args.fp16,
        int8_mode=args.int8,
        input_shapes=input_shapes, 
        deploy_cfg=deploy_cfg,
        model_cfg=args.config,
        dataset_cfg=None,
        dataset_type='val',
        device='cuda:0',
        model=model,)

    # if args.int8:
    #     os.remove(calib_path)


if __name__ == '__main__':

    main()
