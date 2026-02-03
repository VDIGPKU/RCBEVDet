# Copyright (c) OpenMMLab. All rights reserved.
import os
import mmcv
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
import math

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from ...core.bbox import LiDARInstance3DBoxes,CameraInstance3DBoxes
from ..builder import PIPELINES
import cv2
from numpy.linalg import inv
from mmcv.runner import get_dist_info

from typing import Any, Dict, Tuple
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.map_api import locations as LOCATIONS
from nuscenes.utils.data_classes import RadarPointCloud
from ...core.points.radar_points import RadarPoints
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from functools import reduce
from mmdet3d.core.bbox import limit_period

import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms


@PIPELINES.register_module()
class RenamePoint2Radar(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int, optional): The max possible cat_id in input
            segmentation mask. Defaults to 40.
    """

    def __init__(self):
        print('rename points to radar!!!!!!!!')

    def __call__(self, results):

        results['radar'] = results['points']

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'rename points to radar!!!!!!!!'
        return repr_str

@PIPELINES.register_module()
class RenameRadar2Point(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int, optional): The max possible cat_id in input
            segmentation mask. Defaults to 40.
    """

    def __init__(self):
        print('rename radar to points!!!!!!!!')

    def __call__(self, results):

        results['points'] = results['radar']

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'rename points to radar!!!!!!!!'
        return repr_str


@PIPELINES.register_module()
class LoadOccGTFromFile(object):
    '''
    Must before LoadAnnotationsBEVDepth because of BEV augmentation
    '''
    def __call__(self, results):
        occ_gt_path = results['occ_gt_path']
        occ_gt_path = os.path.join(occ_gt_path, "labels.npz")

        occ_labels = np.load(occ_gt_path)
        semantics = occ_labels['semantics']
        mask_lidar = occ_labels['mask_lidar']
        mask_camera = occ_labels['mask_camera']

        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera

        return results





@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFileMono3D(LoadImageFromFile):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`LoadImageFromFile`.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        results['cam2img'] = results['img_info']['cam_intrinsic']
        return results


@PIPELINES.register_module()
class LoadImageFromFileVODLSS(LoadImageFromFile): #NEW VOD
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`LoadImageFromFile`.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        # results['lidar2img'] = results['lidar2img']
        # results['P2'] = results['P2']
        # results['Trv2c'] = results['Trv2c']
        # results['rect'] = results['rect']
        # img = results['img']
        # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5,interpolation=cv2.INTER_LINEAR)
        # # import matplotlib.pyplot as plt
        # # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # # plt.imshow(img)
        # # plt.savefig('img1.png')

        # #print(img.shape)
        # results['img_shape'] = img.shape
        # results['img'] = img
        return results

@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int, optional): Number of sweeps. Defaults to 10.
        load_dim (int, optional): Dimension number of the loaded points.
            Defaults to 5.
        use_dim (list[int], optional): Which dimension to use.
            Defaults to [0, 1, 2, 4].
        time_dim (int, optional): Which dimension to represent the timestamps
            of each points. Defaults to 4.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool, optional): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool, optional): Whether to remove close points.
            Defaults to False.
        test_mode (bool, optional): If `test_mode=True`, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 time_dim=4,
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.time_dim = time_dim
        assert time_dim < load_dim, \
            f'Expect the timestamp dimension < {load_dim}, got {time_dim}'
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float, optional): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data.
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, self.time_dim] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, self.time_dim] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class PointSegClassMapping(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int, optional): The max possible cat_id in input
            segmentation mask. Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=np.int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(valid_cat_ids={self.valid_cat_ids}, '
        repr_str += f'max_cat_id={self.max_cat_id})'
        return repr_str


@PIPELINES.register_module()
class NormalizePointsColor(object):
    """Normalize color of points.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
    """

    def __init__(self, color_mean):
        self.color_mean = color_mean

    def __call__(self, results):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points.
                Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = results['points']
        assert points.attribute_dims is not None and \
            'color' in points.attribute_dims.keys(), \
            'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                points.color.new_tensor(self.color_mean)
        points.color = points.color / 255.0
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(color_mean={self.color_mean})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk'),
                 keys_name='points'):
        self.keys_name = keys_name
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results[self.keys_name] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str



@PIPELINES.register_module()
class LoadPointsFromDict(LoadPointsFromFile):
    """Load Points From Dict."""

    def __call__(self, results):
        assert 'points' in results
        return results


@PIPELINES.register_module()
class LoadRadarPointsMultiSweeps(object):
    """Load radar points from multiple sweeps.
    This is usually used for nuScenes dataset to utilize previous sweeps.
    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 load_dim=18,
                 use_dim=[0, 1, 2, 3, 4],
                 sweeps_num=3, 
                 file_client_args=dict(backend='disk'),
                 max_num=300,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 
                 test_mode=False,
                 rote90=True,
                 ignore=[]):
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.sweeps_num = sweeps_num
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.max_num = max_num
        self.test_mode = test_mode
        self.pc_range = pc_range
        self.ignore = ignore
        self.rote90 = rote90
        if len(ignore)>0:
            print(self.ignore)

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.
        Args:
            pts_filename (str): Filename of point clouds data.
        Returns:
            np.ndarray: An array containing point clouds data.
            [N, 18]
        """
        if pts_filename.endswith(".bin"):
            # V2X-Radar format [x, y, z, range, velocity]
            points = np.fromfile(pts_filename, dtype=np.float32).reshape(-1, 5)
            # Pad to 18 dims to match NuScenes expectation
            padded_points = np.zeros((points.shape[0], 18), dtype=np.float32)
            padded_points[:, :3] = points[:, :3] # x, y, z
            padded_points[:, 5] = points[:, 3] # range -> rcs
            padded_points[:, 8] = points[:, 4] # velocity -> vx_comp
            return padded_points
        radar_obj = RadarPointCloud.from_file(pts_filename)

        #[18, N]
        points = radar_obj.points

        return points.transpose().astype(np.float32)
        

    def _pad_or_drop(self, points):
        '''
        points: [N, 18]
        ''' 

        num_points = points.shape[0]

        if num_points == self.max_num:
            masks = np.ones((num_points, 1), 
                        dtype=points.dtype)

            return points, masks
        
        if num_points > self.max_num:
            points = np.random.permutation(points)[:self.max_num, :]
            masks = np.ones((self.max_num, 1), 
                        dtype=points.dtype)
            
            return points, masks

        if num_points < self.max_num:
            zeros = np.zeros((self.max_num - num_points, points.shape[1]), 
                        dtype=points.dtype)
            masks = np.ones((num_points, 1), 
                        dtype=points.dtype)
            
            points = np.concatenate((points, zeros), axis=0)
            masks = np.concatenate((masks, zeros.copy()[:, [0]]), axis=0)

            return points, masks

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.
        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.
        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.
                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        radars_dict = results['radar']
        # print(radars_dict.keys())

        points_sweep_list = []
        for key, sweeps in radars_dict.items():
            if key in self.ignore:
                continue
            if len(sweeps) < self.sweeps_num:
                idxes = list(range(len(sweeps)))
            else:
                idxes = list(range(self.sweeps_num))
            
            ts = sweeps[0]['timestamp'] * 1e-6
            for idx in idxes:
                sweep = sweeps[idx]

                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                timestamp = sweep['timestamp'] * 1e-6
                time_diff = ts - timestamp
                # print(time_diff)
                time_diff = np.ones((points_sweep.shape[0], 1)) * time_diff

                # velocity compensated by the ego motion in sensor frame
                velo_comp = points_sweep[:, 8:10]
                velo_comp = np.concatenate(
                    (velo_comp, np.zeros((velo_comp.shape[0], 1))), 1)
                velo_comp = velo_comp @ sweep['sensor2lidar_rotation'].T
                velo_comp = velo_comp[:, :2]

                # velocity in sensor frame
                velo = points_sweep[:, 6:8]
                velo = np.concatenate(
                    (velo, np.zeros((velo.shape[0], 1))), 1)
                velo = velo @ sweep['sensor2lidar_rotation'].T
                velo = velo[:, :2]

                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                # print()
                points_sweep_ = np.concatenate(
                    [points_sweep[:, :6], velo,
                     velo_comp, points_sweep[:, 10:],
                     time_diff], axis=1)
                points_sweep_list.append(points_sweep_)
        
        points = np.concatenate(points_sweep_list, axis=0)
        # print(points.shape)
        
        points = points[:, self.use_dim]
        
        #print(points.shape[-1])

        points = RadarPoints(
            points, points_dim=points.shape[-1], attribute_dims=None
        )
        if(self.rote90):
            points.rotate(-math.pi/2) #
        results['radar'] = points
        return results
    
    
    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'




@PIPELINES.register_module()
class LoadRadarPointsMultiSweep2image(object):
    """Load radar points from multiple sweeps.
    This is usually used for nuScenes dataset to utilize previous sweeps.
    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 load_dim=18,
                 use_dim=[0, 1, 2, 3, 4],
                 sweeps_num=3, 
                 file_client_args=dict(backend='disk'),
                 max_num=300,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 
                 test_mode=False,
                 data_root='data/nuscenes/',
                 version='v1.0-trainval',
                 ):
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.sweeps_num = sweeps_num
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.max_num = max_num
        self.test_mode = test_mode
        self.pc_range = pc_range
        self.data_root = data_root
        self.version = version
        self.cam2radar_mappings = {'CAM_FRONT_LEFT': ['RADAR_FRONT_LEFT', 'RADAR_FRONT'],
                                   'CAM_FRONT': ['RADAR_FRONT'],
                                   'CAM_FRONT_RIGHT': ['RADAR_FRONT_RIGHT', 'RADAR_FRONT'],
                                   'CAM_BACK_LEFT': ['RADAR_FRONT_LEFT', 'RADAR_BACK_LEFT'],
                                   'CAM_BACK': ['RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'],
                                   'CAM_BACK_RIGHT': ['RADAR_FRONT_RIGHT', 'RADAR_BACK_RIGHT'] }
        self.nusc = NuScenes(self.version, dataroot = self.data_root, verbose=False)   

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.
        Args:
            pts_filename (str): Filename of point clouds data.
        Returns:
            np.ndarray: An array containing point clouds data.
            [N, 18]
        """
        radar_obj = RadarPointCloud.from_file(pts_filename)

        #[18, N]
        points = radar_obj.points

        return points.transpose().astype(np.float32)
        

    def _pad_or_drop(self, points):
        '''
        points: [N, 18]
        ''' 

        num_points = points.shape[0]

        if num_points == self.max_num:
            masks = np.ones((num_points, 1), 
                        dtype=points.dtype)

            return points, masks
        
        if num_points > self.max_num:
            points = np.random.permutation(points)[:self.max_num, :]
            masks = np.ones((self.max_num, 1), 
                        dtype=points.dtype)
            
            return points, masks

        if num_points < self.max_num:
            zeros = np.zeros((self.max_num - num_points, points.shape[1]), 
                        dtype=points.dtype)
            masks = np.ones((num_points, 1), 
                        dtype=points.dtype)
            
            points = np.concatenate((points, zeros), axis=0)
            masks = np.concatenate((masks, zeros.copy()[:, [0]]), axis=0)

            return points, masks

    def cal_matrix_refSensor_from_global(self, nusc, sensor_token):    
        sensor_data = nusc.get('sample_data', sensor_token)    
        ref_pose_rec = nusc.get('ego_pose', sensor_data['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', sensor_data['calibrated_sensor_token'])    
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)    
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True)        
        M_ref_from_global = reduce(np.dot, [ref_from_car, car_from_global])    
        return M_ref_from_global


    def cal_matrix_refSensor_to_global(self, nusc, sensor_token):    
        sensor_data = nusc.get('sample_data', sensor_token)       
        current_pose_rec = nusc.get('ego_pose', sensor_data['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                        Quaternion(current_pose_rec['rotation']), inverse=False)
        current_cs_rec = nusc.get('calibrated_sensor', sensor_data['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']), inverse=False)    
        M_ref_to_global = reduce(np.dot, [global_from_car, car_from_current])    
        return M_ref_to_global



    def cal_trans_matrix(self, nusc, sensor1_token, sensor2_token):          
        M_ref_to_global = self.cal_matrix_refSensor_to_global(nusc, sensor1_token)    
        M_ref_from_global = self.cal_matrix_refSensor_from_global(nusc, sensor2_token)
        trans_matrix = reduce(np.dot, [M_ref_from_global, M_ref_to_global])   
        return trans_matrix

    def proj2im(self, nusc, pc_cam, cam_token, min_z = 2):            
        cam_data = nusc.get('sample_data', cam_token) 
        cs_rec = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])         
        depth = pc_cam.points[2]    
        msk = pc_cam.points[2] >= min_z       
        points = view_points(pc_cam.points[:3, :], np.array(cs_rec['camera_intrinsic']), normalize=True)        
        x, y = points[0], points[1]
        msk =  reduce(np.logical_and, [x>0, x<1600, y>0, y<900, msk])        
        return x, y, depth, msk 

    def load_multi_radar_to_cam(self, nusc, cam2radar_mappings, results):


        sample_token = results['img_info']['token']
        cam_token =  results['img_info']['id']   
        cam_channel = nusc.get('sample_data', cam_token)['channel']          
        radar_channels = cam2radar_mappings[cam_channel]   
        sample = nusc.get('sample', sample_token)         
        RadarPointCloud.disable_filters()
        n_dims = RadarPointCloud.nbr_dims()
        all_pc = RadarPointCloud(np.zeros((n_dims, 0)))
        
        for radar_channel in radar_channels:
            radar_token = sample['data'][radar_channel]
            radar_path = nusc.get_sample_data_path(radar_token)            
            pc = RadarPointCloud.from_file(radar_path)
            
            T_r2c = self.cal_trans_matrix(nusc, radar_token, cam_token)
            pc.transform(T_r2c)      
            R_r2c = T_r2c[:3,:3] 
            v0 = np.vstack(( pc.points[[6,7],:], np.zeros(pc.nbr_points()) ))  
            v0_comp = np.vstack(( pc.points[[8,9],:], np.zeros(pc.nbr_points()) )) 
            v1 = R_r2c.dot(v0)
            v1_comp = R_r2c.dot(v0_comp)
            
            pc.points[[6,7],:] = v1[[0,2],:]         
            pc.points[[8,9],:] = v1_comp[[0,2],:]                  
            all_pc.points = np.hstack((all_pc.points, pc.points))
                
        xz_cam = all_pc.points[[0,2],:]   
        v_raw = all_pc.points[[6,7],:]    
        v_comp = all_pc.points[[8,9],:]   
        rcs = all_pc.points[5,:]
                
        x_i, y_i, depth, msk = self.proj2im(nusc, all_pc, cam_token)
        x_i, y_i, depth = x_i[msk], y_i[msk], depth[msk]
        
        xz_cam, v_raw, v_comp = xz_cam[:,msk], v_raw[:,msk], v_comp[:,msk]  
        rcs = rcs[msk]
        xy_im = np.stack([x_i, y_i])     
        radar_pts = np.concatenate([xz_cam, xy_im, v_comp, v_raw], axis=0)  
        
        radar_pts = np.concatenate([radar_pts, rcs[None,:]], axis=0)  
            
        h_im, w_im = 900, 1600
        radar_map = np.zeros( (h_im, w_im, 10) , dtype=float) 
        
        x_i = np.clip(x_i, 0, w_im - 1)
        y_i = np.clip(y_i, 0, h_im - 1)
        
        x = xz_cam[0,:]
        assert np.array_equal(xz_cam[1,:], depth)
        
        vx, vz = v_raw[0,:], v_raw[1,:]
        v_amplitude = (vx**2 + vz**2)**0.5
        vx_comp, vz_comp = v_comp[0,:], v_comp[1,:]
        v_comp_amplitude = (vx_comp**2 + vz_comp**2)**0.5

        for i in range(len(x_i)):
            x_one, y_one = int(round( x_i[i] )), int(round( y_i[i] )) 
                
            if radar_map[y_one,x_one,0] == 0 or radar_map[y_one,x_one,0] > depth[i]:
                radar_map[y_one,x_one,:] = [x[i], depth[i], 1, vx[i], vz[i], v_amplitude[i], vx_comp[i], vz_comp[i], v_comp_amplitude[i], rcs[i]]  
                
        results['radar_map'] = radar_map.astype('float32')
        results['radar_pts'] = radar_pts.astype('float32')  
        
        return results    

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.
        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.
        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.
                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """

        ''' 先不用
        radars_dict = results['radar']

        points_sweep_list = []
        for key, sweeps in radars_dict.items():
            if len(sweeps) < self.sweeps_num:
                idxes = list(range(len(sweeps)))
            else:
                idxes = list(range(self.sweeps_num))
            
            ts = sweeps[0]['timestamp'] * 1e-6
            for idx in idxes:
                sweep = sweeps[idx]

                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                timestamp = sweep['timestamp'] * 1e-6
                time_diff = ts - timestamp
                time_diff = np.ones((points_sweep.shape[0], 1)) * time_diff

                # velocity compensated by the ego motion in sensor frame
                velo_comp = points_sweep[:, 8:10]
                velo_comp = np.concatenate(
                    (velo_comp, np.zeros((velo_comp.shape[0], 1))), 1)
                velo_comp = velo_comp @ sweep['sensor2lidar_rotation'].T
                velo_comp = velo_comp[:, :2]

                # velocity in sensor frame
                velo = points_sweep[:, 6:8]
                velo = np.concatenate(
                    (velo, np.zeros((velo.shape[0], 1))), 1)
                velo = velo @ sweep['sensor2lidar_rotation'].T
                velo = velo[:, :2]

                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']

                points_sweep_ = np.concatenate(
                    [points_sweep[:, :6], velo,
                     velo_comp, points_sweep[:, 10:],
                     time_diff], axis=1)
                points_sweep_list.append(points_sweep_)
        
        points = np.concatenate(points_sweep_list, axis=0)
        
        points = points[:, self.use_dim]
        
        #print(points.shape[-1])

        points = RadarPoints(
            points, points_dim=points.shape[-1], attribute_dims=None
        )
        
        results['radar'] = points
        '''
        data1 = open("newfile.txt",'w',encoding="utf-8")
        print(results,file=data1)
        results=self.load_multi_radar_to_cam(self.nusc, self.cam2radar_mappings, results)

        return results

@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 poly2mask=True,
                 seg_3d_dtype=np.int64,
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        #print(type(results['ann_infos'])) #test
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']  #change infos
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results['centers2d'] = results['ann_info']['centers2d']
        results['depths'] = results['ann_info']['depths']
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int64)
        except ConnectionError:
            mmcv.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.int64)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.int64)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str


@PIPELINES.register_module()
class PointToMultiViewDepth(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        if 'ori_points' in results:
            points_lidar = results['ori_points']
        else:
            points_lidar = results['points']
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        depth_map_list = []
        lidar2img_list = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
                results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            lidar2lidarego = torch.from_numpy(lidar2lidarego)

            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(
                results['curr']['ego2global_rotation']).rotation_matrix
            lidarego2global[:3, 3] = results['curr']['ego2global_translation']
            lidarego2global = torch.from_numpy(lidarego2global)

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['sensor2ego_rotation']).rotation_matrix
            cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                'sensor2ego_translation']
            cam2camego = torch.from_numpy(cam2camego)

            camego2global = np.eye(4, dtype=np.float32)
            camego2global[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['ego2global_rotation']).rotation_matrix
            camego2global[:3, 3] = results['curr']['cams'][cam_name][
                'ego2global_translation']
            camego2global = torch.from_numpy(camego2global)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global.matmul(lidar2lidarego))
            lidar2img = cam2img.matmul(lidar2cam)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map = self.points2depthmap(points_img, imgs.shape[2],
                                             imgs.shape[3])
            depth_map_list.append(depth_map)
            lidar2img_list.append(lidar2img)
        depth_map = torch.stack(depth_map_list)
        lidar2img_list = torch.stack(lidar2img_list)
        results['lidar2img'] = lidar2img_list
        results['gt_depth'] = depth_map
        return results


@PIPELINES.register_module()
class PointToMultiViewDepthVoD(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        # print(depth.max(), depth.min())
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
                coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                        depth < self.grid_config['depth'][1]) & (
                        depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        # import pdb;pdb.set_trace()
        # imgs, rots, trans, intrins = results['img_inputs'][:4]
        imgs, sensor2keyegos, ego2globals, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        depth_map_list = []
        lidar2img_list = []
        cid = 0

        # lidar2lidarego = np.eye(4, dtype=np.float32)
        # lidar2lidarego[:3, :3] = Quaternion(
        #     results['curr']['lidar2ego_rotation']).rotation_matrix
        # lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
        # lidar2lidarego = torch.from_numpy(lidar2lidarego)

        # lidarego2global = np.eye(4, dtype=np.float32)
        # lidarego2global[:3, :3] = Quaternion(
        #     results['curr']['ego2global_rotation']).rotation_matrix
        # lidarego2global[:3, 3] = results['curr']['ego2global_translation']
        # lidarego2global = torch.from_numpy(lidarego2global)

        # cam2camego = np.eye(4, dtype=np.float32)
        # cam2camego[:3, :3] = Quaternion(
        #     results['curr']['cams'][cam_name]
        #     ['sensor2ego_rotation']).rotation_matrix
        # cam2camego[:3, 3] = results['curr']['cams'][cam_name][
        #     'sensor2ego_translation']
        # cam2camego = torch.from_numpy(cam2camego)

        # camego2global = np.eye(4, dtype=np.float32)
        # camego2global[:3, :3] = Quaternion(
        #     results['curr']['cams'][cam_name]
        #     ['ego2global_rotation']).rotation_matrix
        # camego2global[:3, 3] = results['curr']['cams'][cam_name][
        #     'ego2global_translation']
        # camego2global = torch.from_numpy(camego2global)

        cam2img = np.eye(4, dtype=np.float32)
        cam2img = torch.from_numpy(cam2img)
        # print(intrins.shape)
        cam2img[:3, :3] = intrins[cid]

        # lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
        #     lidarego2global.matmul(lidar2lidarego))
        lidar2cam = torch.inverse(sensor2keyegos[0])
        lidar2img = cam2img.matmul(lidar2cam)

        points_img = points_lidar.tensor[:, :3].matmul(
            lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
        # print(points_img.shape)
        points_img = torch.cat(
            [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
            1)
        # print(points_img.shape)
        points_img = points_img.matmul(
            post_rots[cid].T) + post_trans[cid:cid + 1, :]
        depth_map = self.points2depthmap(points_img, imgs.shape[2],
                                         imgs.shape[3])
        depth_map_list.append(depth_map)
        lidar2img_list.append(lidar2img)

        depth_map = torch.stack(depth_map_list)
        lidar2img_list = torch.stack(lidar2img_list)
        results['lidar2img'] = lidar2img_list
        results['gt_depth'] = depth_map
        return results


@PIPELINES.register_module()
class PointToMultiViewDepthLongterm(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs, rots, trans, intrins = results['img_inputs_lt'][:4]
        post_rots, post_trans, bda = results['img_inputs_lt'][4:]
        depth_map_list = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
                results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            lidar2lidarego = torch.from_numpy(lidar2lidarego)

            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(
                results['curr']['ego2global_rotation']).rotation_matrix
            lidarego2global[:3, 3] = results['curr']['ego2global_translation']
            lidarego2global = torch.from_numpy(lidarego2global)

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['sensor2ego_rotation']).rotation_matrix
            cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                'sensor2ego_translation']
            cam2camego = torch.from_numpy(cam2camego)

            camego2global = np.eye(4, dtype=np.float32)
            camego2global[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['ego2global_rotation']).rotation_matrix
            camego2global[:3, 3] = results['curr']['cams'][cam_name][
                'ego2global_translation']
            camego2global = torch.from_numpy(camego2global)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global.matmul(lidar2lidarego))
            lidar2img = cam2img.matmul(lidar2cam)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map = self.points2depthmap(points_img, imgs.shape[2],
                                             imgs.shape[3])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        results['gt_depth_lt'] = depth_map
        return results


def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


@PIPELINES.register_module()
class PrepareImageInputs(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        data_config,
        is_train=False,
        sequential=False,
        ego_cam='CAM_FRONT',
        add_adj_bbox=False,
        with_stereo=False,
        with_future_pred=False,
        img_norm_cfg=None,
        ignore=[],
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.ego_cam = ego_cam
        self.with_future_pred = with_future_pred
        self.add_adj_bbox = add_adj_bbox
        self.img_norm_cfg = img_norm_cfg
        self.with_stereo = with_stereo
        self.ignore = ignore
        if len(ignore) > 0:
            print(self.ignore)

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sweep2key_transformation(self,
                                    cam_info,
                                    key_info,
                                    cam_name,
                                    ego_cam=None):
        if ego_cam is None:
            ego_cam = cam_name
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        sweepego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepego2global_tran = torch.Tensor(
            cam_info['cams'][cam_name]['ego2global_translation'])
        sweepego2global = sweepego2global_rot.new_zeros((4, 4))
        sweepego2global[3, 3] = 1
        sweepego2global[:3, :3] = sweepego2global_rot
        sweepego2global[:3, -1] = sweepego2global_tran

        # global sensor to cur ego
        w, x, y, z = key_info['cams'][ego_cam]['ego2global_rotation']
        keyego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keyego2global_tran = torch.Tensor(
            key_info['cams'][ego_cam]['ego2global_translation'])
        keyego2global = keyego2global_rot.new_zeros((4, 4))
        keyego2global[3, 3] = 1
        keyego2global[:3, :3] = keyego2global_rot
        keyego2global[:3, -1] = keyego2global_tran
        global2keyego = keyego2global.inverse()

        sweepego2keyego = global2keyego @ sweepego2global

        return sweepego2keyego
    
    def get_sensor_transforms(self, cam_info, cam_name):
        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
        # sweep sensor to sweep ego
        sensor2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sensor2ego_tran = torch.Tensor(
            cam_info['cams'][cam_name]['sensor2ego_translation'])
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        ego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        ego2global_tran = torch.Tensor(
            cam_info['cams'][cam_name]['ego2global_translation'])
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names

        canvas = []
        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            results['img_file_paths'][cam_name] = filename  # for vis
            img = Image.open(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            sensor2ego, ego2global = \
                self.get_sensor_transforms(results['curr'], cam_name)
            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2
            # print(cam_name, self.ignore)
            if cam_name in self.ignore:
                canvas.append(np.zeros_like(np.array(img)))
                imgs.append(torch.zeros_like(self.normalize_img(img)))
            else:
                canvas.append(np.array(img))
                imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    if cam_name in self.ignore:
                        imgs.append(torch.zeros_like(self.normalize_img(img_adjacent)))
                    else:
                        imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            for adj_info in results['adjacent']:
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # align
                for cam_name in cam_names:
                    sensor2ego, ego2global = \
                        self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)
            if self.add_adj_bbox:
                results['adjacent_bboxes'] = self.align_adj_bbox2keyego(results)

        imgs = torch.stack(imgs)

        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        results['canvas'] = canvas
        results['img_shape'] = [(self.data_config['input_size'][0], self.data_config['input_size'][1]) for _ in range(6)]
        return (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans)

    def __call__(self, results):
        results['img_file_paths'] = {}
        if self.add_adj_bbox:
            results['adjacent_bboxes'] = self.get_adjacent_bboxes(results)
        results['img_inputs'] = self.get_inputs(results)
        results['input_shape'] = self.data_config['input_size']  # new
        return results

    def get_adjacent_bboxes(self, results):
        adjacent_bboxes = list()
        for idx, adj_info in enumerate(results['adjacent']):
            if self.with_stereo and idx > 0:  # reference frame不用读gt
                break    
            adjacent_bboxes.append(adj_info['ann_infos'])
        return adjacent_bboxes

    def align_adj_bbox2keyego(self, results):
        cam_name = self.choose_cams()[0]
        ret_list = []
        for idx, adj_info in enumerate(results['adjacent']):
            if self.with_stereo and idx > 0:  # reference frame不用读gt 也不用align
                break   
            sweepego2keyego = self.get_sweep2key_transformation(adj_info,
                                results['curr'],
                                cam_name,
                                self.ego_cam)
            adj_bbox, adj_labels = results['adjacent_bboxes'][idx]
            adj_bbox = torch.Tensor(adj_bbox)
            adj_labels = torch.tensor(adj_labels)
            gt_bbox = adj_bbox
            if len(adj_bbox) == 0:
                adj_bbox = torch.zeros(0, 9)
                ret_list.append((adj_bbox, adj_labels))
                continue
            # center
            homo_sweep_center = torch.cat([gt_bbox[:,:3], torch.ones_like(gt_bbox[:,0:1])], dim=-1)
            homo_key_center = (sweepego2keyego @ homo_sweep_center.t()).t() # [4, N]
            # velo
            rot = sweepego2keyego[:3, :3]
            homo_sweep_velo =  torch.cat([gt_bbox[:, 7:], torch.zeros_like(gt_bbox[:,0:1])], dim=-1)
            homo_key_velo = (rot @ homo_sweep_velo.t()).t()
            # yaw
            def get_new_yaw(box_cam, extrinsic):
                corners = box_cam.corners
                cam2lidar_rt = torch.tensor(extrinsic)
                N = corners.shape[0]
                corners = corners.reshape(N*8, 3)
                extended_xyz = torch.cat(
                    [corners, corners.new_ones(corners.size(0), 1)], dim=-1)
                corners = extended_xyz @ cam2lidar_rt.T
                corners = corners.reshape(N, 8, 4)[:, :, :3]
                yaw = np.arctan2(corners[:,1,1]-corners[:,2,1], corners[:,1,0]-corners[:,2,0])
                def limit_period(val, offset=0.5, period=np.pi):
                    """Limit the value into a period for periodic function.

                    Args:
                        val (np.ndarray): The value to be converted.
                        offset (float, optional): Offset to set the value range. \
                            Defaults to 0.5.
                        period (float, optional): Period of the value. Defaults to np.pi.

                    Returns:
                        torch.Tensor: Value in the range of \
                            [-offset * period, (1-offset) * period]
                    """
                    return val - np.floor(val / period + offset) * period
                return limit_period(yaw + (np.pi/2), period=np.pi * 2)

            new_yaw_sweep = get_new_yaw(LiDARInstance3DBoxes(adj_bbox, box_dim=adj_bbox.shape[-1],
                                        origin=(0.5, 0.5, 0.5)), sweepego2keyego).reshape(-1,1)
            adj_bbox = torch.cat([homo_key_center[:, :3], gt_bbox[:,3:6], new_yaw_sweep, homo_key_velo[:, :2]], dim=-1)
            ret_list.append((adj_bbox, adj_labels))

        return ret_list


@PIPELINES.register_module()
class PrepareImageInputsLongterm(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        data_config,
        is_train=False,
        sequential=False,
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sensor_transforms(self, cam_info, cam_name):
        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
        # sweep sensor to sweep ego
        sensor2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sensor2ego_tran = torch.Tensor(
            cam_info['cams'][cam_name]['sensor2ego_translation'])
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        ego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        ego2global_tran = torch.Tensor(
            cam_info['cams'][cam_name]['ego2global_translation'])
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []
        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            sensor2ego, ego2global = \
                self.get_sensor_transforms(results['curr'], cam_name)
            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            canvas.append(np.array(img))
            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent_lt' in results
                for adj_info in results['adjacent_lt']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            for adj_info in results['adjacent_lt']:
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # align
                for cam_name in cam_names:
                    sensor2ego, ego2global = \
                        self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)

        imgs = torch.stack(imgs)

        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        results['canvas_lt'] = canvas
        return (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans)

    def __call__(self, results):
        results['img_inputs_lt'] = self.get_inputs(results)
        return results


@PIPELINES.register_module()
class LoadAnnotationsBEVDepth(object):

    def __init__(self, bda_aug_conf, classes, is_train=True, sequential=False, align_adj_bbox=False, with_hop=False, is_val=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes
        self.sequential = sequential
        self.align_adj_bbox = align_adj_bbox
        self.with_hop = with_hop
        self.is_val = is_val  # if is_val then load bbox gt for seg gt

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        fsr_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                fsr_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:,
                         6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                                           6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                fsr_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, fsr_mat, rot_mat, flip_mat, scale_mat

    def __call__(self, results):
        if self.is_train or self.is_val:
            gt_boxes, gt_labels = results['ann_infos']
            gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
        else:
            gt_boxes = torch.zeros(0, 9)
            gt_labels = torch.zeros(0, 1)

        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(
        )
        results['rotate_bda']=rotate_bda
        results['scale_bda']=scale_bda
        results['flip_dx']=flip_dx
        results['flip_dy']=flip_dy #save


        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot, rm, fm, sm = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_mat[:3, :3] = rm
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels

        if 'points' in results:
            points = results['points']
            lidar2ego = results['lidar2ego']
            # points.rotate(lidar2ego[:3, :3].T)
            # points.tensor[:, :3] = points.tensor[:, :3] + lidar2ego[:3, 3]
            points.tensor[:, :3] = (bda_rot @ points.tensor[:, :3].unsqueeze(-1)).squeeze(-1)
            results['points'] = points

            if False:  # for debug vis
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                import random
                print('\n******** BEGIN PRINT GT and LiDAR POINTS **********\n')
                corner = results['gt_bboxes_3d'].corners
                fig = plt.figure(figsize=(16, 16))
                plt.plot([50, 50, -50, -50, 50], [50, -50, -50, 50, 50], lw=0.5)
                plt.plot([65, 65, -65, -65, 65], [65, -65, -65, 65, 65], lw=0.5)
                for i in range(corner.shape[0]):
                    x1 = corner[i][0][0]
                    y1 = corner[i][0][1]
                    x2 = corner[i][2][0]
                    y2 = corner[i][2][1]
                    x3 = corner[i][6][0]
                    y3 = corner[i][6][1]
                    x4 = corner[i][4][0]
                    y4 = corner[i][4][1]
                    plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], lw=1)
                plt.scatter(points.tensor[:, 0].view(-1).numpy(), points.tensor[:, 1].view(-1).numpy(), s=0.5, c='black')
                plt.savefig("/home/xiazhongyu/vis/gt" + str(random.randint(1, 9999)) + ".png")
                print('\n******** END PRINT GT and LiDAR POINTS **********\n')
                exit(0)

        if 'img_inputs' in results:
            imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
            post_rots, post_trans = results['img_inputs'][4:]
            results['img_inputs'] = (imgs, sensor2egos, ego2globals, intrins, post_rots,
                                     post_trans, bda_rot)
            ego2img_rts = []
            if not self.sequential and self.with_hop:
                sensor2keyegos = self.get_sensor2keyego_transformation(sensor2egos, ego2globals)
                for sensor2keyego, intrin, post_rot, post_tran in zip(
                        sensor2keyegos, intrins, post_rots, post_trans):
                    rot = sensor2keyego[:3, :3]
                    tran = sensor2keyego[:3, 3]
                    viewpad = torch.eye(3).to(imgs.device)
                    viewpad[:post_rot.shape[0], :post_rot.shape[1]] = \
                        post_rot @ intrin[:post_rot.shape[0], :post_rot.shape[1]]
                    viewpad[:post_tran.shape[0], 2] += post_tran
                    intrinsic = viewpad
                    
                    # need type float
                    ego2img_r = intrinsic.float() @ torch.linalg.inv(rot.float()) @ torch.linalg.inv(bda_rot.float())
                    ego2img_t = -intrinsic.float() @ torch.linalg.inv(rot.float()) @ tran.float()
                    ego2img_rt = torch.eye(4).to(imgs.device)
                    ego2img_rt[:3, :3] = ego2img_r
                    ego2img_rt[:3, 3] = ego2img_t
                    '''
                    X_{3d} = bda * (rots * (intrinsic)^(-1) * X_{img} + trans)
                    bda^(-1) * X_{3d} = rots * (intrinsic)^(-1) * X_{img} + trans
                    bda^(-1) * X_{3d} - trans = rots * (intrinsic)^(-1) * X_{img}
                    intrinsic * rots^(-1) * (bda^(-1) * X_{3d} - trans) = X_{img}
                    intrinsic * rots^(-1) * bda^(-1) * X_{3d} - intrinsic * rots^(-1) * trans = X_{img}
                    rotate = intrinsic * rots^(-1) * bda^(-1)
                    translation = - intrinsic * rots^(-1) * trans
                    '''
                    ego2img_rts.append(ego2img_rt)
                ego2img_rts = torch.stack(ego2img_rts, dim=0)
            if self.align_adj_bbox:
                results = self.align_adj_bbox_bda(results, rotate_bda, scale_bda,
                                                flip_dx, flip_dy)
            results['lidar2img'] = np.asarray(ego2img_rts)

            if 'img_inputs_lt' in results.keys():
                imgs_lt, rots_lt, trans_lt, intrins_lt = results['img_inputs_lt'][:4]
                post_rots_lt, post_trans_lt = results['img_inputs_lt'][4:]
                results['img_inputs_lt'] = (imgs_lt, rots_lt, trans_lt, intrins_lt,
                                            post_rots_lt, post_trans_lt, bda_rot)

        results["bda_r"] = bda_mat
        results["bda_f"] = (flip_dx, flip_dy)
        results["bda_s"] = scale_bda
        return results
    
    def get_sensor2keyego_transformation(self, sensor2egos, ego2globals):
        # sensor2ego -> sweep sensor to sweep ego
        # ego2globals -> sweep ego to global 
        sensor2keyegos = []
        keyego2global = ego2globals[0] # assert key ego is frame 0 with CAM_FRONT
        global2keyego = torch.inverse(keyego2global.double())
        for sensor2ego, ego2global in zip(sensor2egos, ego2globals):
            # calculate the transformation from sweep sensor to key ego
            sensor2keyego = global2keyego @ ego2global.double() @ sensor2ego.double()
            sensor2keyegos.append(sensor2keyego)
        return sensor2keyegos

    def align_adj_bbox_bda(self, results, rotate_bda, scale_bda, flip_dx, flip_dy):
        for adjacent_bboxes in results['adjacent_bboxes']:
            adj_bbox, adj_label = adjacent_bboxes
            gt_boxes = adj_bbox
            if len(gt_boxes) == 0:
                gt_boxes = torch.zeros(0, 9)
            gt_boxes, _, _, _, _ = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                        flip_dx, flip_dy)
            if not 'adj_gt_3d' in results.keys():
                adj_bboxes_3d = \
                    LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                        origin=(0.5, 0.5, 0.5))
                adj_labels_3d = adj_label
                results['adj_gt_3d'] = [[adj_bboxes_3d, adj_labels_3d]]
            else:
                adj_bboxes_3d = \
                    LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                        origin=(0.5, 0.5, 0.5))
                results['adj_gt_3d'].append([
                    adj_bboxes_3d, adj_label
                ])
        return results 


@PIPELINES.register_module()
class LoadAnnotationsBEVDepthLidar(object):

    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']

            translation_std = np.array(self.bda_aug_conf['trans_xyz'], dtype=np.float32)
            trans_bda = np.random.normal(scale=translation_std, size=3).reshape(-1)

            # input_dict['points'].translate(trans_factor)
            # input_dict['pcd_trans'] = trans_factor
            # for key in input_dict['bbox3d_fields']:
            #     input_dict[key].translate(trans_factor)
            # trans_x = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
            # trans_x = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
            # trans_x = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
            trans_bda = np.array([0., 0., 0.]).reshape(-1)
        return rotate_bda, scale_bda, flip_dx, flip_dy, trans_bda

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy, trans_bda):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        trans_mat = torch.Tensor(trans_bda)

        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        fsr_mat = flip_mat @ (scale_mat @ rot_mat)

        sr_mat = scale_mat @ rot_mat

        if gt_boxes.shape[0] > 0:
            # gt_boxes[:, :3] = (
            #     fsr_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)

            gt_boxes[:, :3] = (
                sr_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)

            print('bevfore box', trans_mat, gt_boxes[0, :3])
            gt_boxes[:, :3] += trans_mat.view(1, -1)
            print('after box', gt_boxes[0, :3])
            
            gt_boxes[:, :3] = (
                flip_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)


            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:,
                         6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                                           6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                fsr_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, fsr_mat, rot_mat, flip_mat, scale_mat, trans_mat

    def __call__(self, results):
        gt_boxes, gt_labels = results['ann_infos']
        gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
        rotate_bda, scale_bda, flip_dx, flip_dy, trans_bda = self.sample_bda_augmentation(
        )
        results['rotate_bda']=rotate_bda
        results['scale_bda']=scale_bda
        results['flip_dx']=flip_dx
        results['flip_dy']=flip_dy #save
        results['trans_bda']=trans_bda


        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot, rm, fm, sm, tm = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy, trans_bda)
        bda_mat[:3, :3] = rm
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels

        if 'points' in results:
            points = results['points']
            lidar2ego = results['lidar2ego']
            points.rotate(lidar2ego[:3, :3].T)
            points.tensor[:, :3] = points.tensor[:, :3] + lidar2ego[:3, 3]

            points.tensor[:, :3] = (sm @ (rm @ points.tensor[:, :3].unsqueeze(-1))).squeeze(-1)
            # print('bevfore', tm, points.tensor[0, :3])
            points.tensor[:, :3] += tm.view(1, -1)
            # print('after', points.tensor[0, :3])
            points.tensor[:, :3] = (fm @ points.tensor[:, :3].unsqueeze(-1)).squeeze(-1)
            results['points'] = points

            if False:  # for debug vis
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                import random
                print('\n******** BEGIN PRINT GT and LiDAR POINTS **********\n')
                corner = results['gt_bboxes_3d'].corners
                fig = plt.figure(figsize=(16, 16))
                plt.plot([50, 50, -50, -50, 50], [50, -50, -50, 50, 50], lw=0.5)
                plt.plot([65, 65, -65, -65, 65], [65, -65, -65, 65, 65], lw=0.5)
                for i in range(corner.shape[0]):
                    x1 = corner[i][0][0]
                    y1 = corner[i][0][1]
                    x2 = corner[i][2][0]
                    y2 = corner[i][2][1]
                    x3 = corner[i][6][0]
                    y3 = corner[i][6][1]
                    x4 = corner[i][4][0]
                    y4 = corner[i][4][1]
                    plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], lw=1)
                plt.scatter(points.tensor[:, 0].view(-1).numpy(), points.tensor[:, 1].view(-1).numpy(), s=0.5, c='black')
                plt.savefig("/home/xiazhongyu/vis/gt" + str(random.randint(1, 9999)) + ".png")
                print('\n******** END PRINT GT and LiDAR POINTS **********\n')
                exit(0)

        if 'img_inputs' in results:
            imgs, rots, trans, intrins = results['img_inputs'][:4]
            post_rots, post_trans = results['img_inputs'][4:]
            results['img_inputs'] = (imgs, rots, trans, intrins, post_rots,
                                     post_trans, bda_rot)
            if 'img_inputs_lt' in results.keys():
                imgs_lt, rots_lt, trans_lt, intrins_lt = results['img_inputs_lt'][:4]
                post_rots_lt, post_trans_lt = results['img_inputs_lt'][4:]
                results['img_inputs_lt'] = (imgs_lt, rots_lt, trans_lt, intrins_lt,
                                            post_rots_lt, post_trans_lt, bda_rot)

        results["bda_r"] = bda_mat
        results["bda_f"] = (flip_dx, flip_dy)
        results["bda_s"] = scale_bda
        return results


@PIPELINES.register_module()
class LoadBEVSegmentation:
    def __init__(
        self,
        dataset_root: str,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        classes: Tuple[str, ...],
        bbox_classes,
    ) -> None:
        super().__init__()
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.classes = list(classes)
        self.need_vehicle = False
        if 'vehicle' in self.classes:
            if self.classes[0] != 'vehicle':
                raise ValueError("Please set vehicle to the first position in the map_classes list.")
            self.classes.remove('vehicle')
            self.need_vehicle = True
            self.bbox_classes = bbox_classes

        self.maps = {}
        for location in LOCATIONS:
            self.maps[location] = NuScenesMap(dataset_root, location)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:

        num_classes = len(self.classes)
        if num_classes > 0:

            if "bda_r" in data.keys():  # if use BEV aug
                bda_r = data["bda_r"]
                point2global = np.linalg.inv(bda_r)
                flip_dx, flip_dy = data["bda_f"]
                scale = data["bda_s"]
                # lidar2ego = data["lidar2ego"]
                ego2global = data["ego2global"]
                # perception coor is lidar ego(IMU) after mmdet3d1.0, instead of lidar sensor(TOP_LIDAR)
                # lidar2global = ego2global @ lidar2ego @ point2lidar
                lidar2global = ego2global @ point2global
            else:
                # lidar2ego = data["lidar2ego"]
                flip_dx = flip_dy = False
                scale = 1
                ego2global = data["ego2global"]
                lidar2global = ego2global  # @ lidar2ego

            map_pose = lidar2global[:2, 3]
            patch_box = (map_pose[0], map_pose[1], self.patch_size[0] / scale, self.patch_size[1] / scale)

            rotation = lidar2global[:3, :3]
            v = np.dot(rotation, np.array([1, 0, 0]))
            yaw = np.arctan2(v[1], v[0])
            patch_angle = yaw / np.pi * 180

            mappings = {}
            for name in self.classes:
                if name == "drivable_area*":
                    mappings[name] = ["road_segment", "lane"]
                elif name == "divider":
                    mappings[name] = ["road_divider", "lane_divider"]
                else:
                    mappings[name] = [name]

            layer_names = []
            for name in mappings:
                layer_names.extend(mappings[name])
            layer_names = list(set(layer_names))

            location = data["location"]
            masks = self.maps[location].get_map_mask(
                patch_box=patch_box,
                patch_angle=patch_angle,
                layer_names=layer_names,
                canvas_size=self.canvas_size,
            )
            # masks = masks[:, ::-1, :].copy()
            masks = masks.transpose(0, 2, 1)
            masks = masks.astype(np.bool)

            labelsmap = np.zeros((num_classes, *self.canvas_size), dtype=np.long)
            for k, name in enumerate(self.classes):
                for layer_name in mappings[name]:
                    index = layer_names.index(layer_name)
                    labelsmap[k, masks[index]] = 1

            if flip_dx:
                labelsmap = np.flip(labelsmap, axis=1)
            if flip_dy:
                labelsmap = np.flip(labelsmap, axis=2)

        if self.need_vehicle:
            labelsv = np.zeros((self.canvas_size[0], self.canvas_size[1]), dtype=np.uint8)
            corner = data["gt_bboxes_3d"].corners
            bbox_labels = data["gt_labels_3d"]

            needed_classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'motorcycle', 'bicycle',]

            v_bboxs = []
            for i in range(bbox_labels.shape[0]):
                if self.bbox_classes[bbox_labels[i]] in needed_classes:
                    x1 = (corner[i][0][0] + 50) / 0.5
                    y1 = (corner[i][0][1] + 50) / 0.5
                    x2 = (corner[i][2][0] + 50) / 0.5
                    y2 = (corner[i][2][1] + 50) / 0.5
                    x3 = (corner[i][6][0] + 50) / 0.5
                    y3 = (corner[i][6][1] + 50) / 0.5
                    x4 = (corner[i][4][0] + 50) / 0.5
                    y4 = (corner[i][4][1] + 50) / 0.5
                    v_bboxs.append([[y1, x1], [y2, x2], [y3, x3], [y4, x4]])
            cv2.fillPoly(labelsv, np.array(v_bboxs, dtype=np.int32), [1])

            labelsv = labelsv.reshape([1, self.canvas_size[0], self.canvas_size[1]])

        if self.need_vehicle and num_classes > 0:
            labels = np.concatenate([labelsv, labelsmap], axis=0)
        elif self.need_vehicle:
            labels = labelsv
        else:
            labels = labelsmap.copy()

        # print(type(labels), labels.shape)
        # exit(0)

        data["gt_masks_bev"] = labels

        # DEBUG for visibility
        if False:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import random
            print('\n******** BEGIN PRINT GT**********\n')
            print("LoadBEVSegmentation:", labels.shape)
            corner = data["gt_bboxes_3d"].corners
            fig = plt.figure(figsize=(16, 16))
            plt.plot([50, 50, -50, -50, 50], [50, -50, -50, 50, 50], lw=0.5)
            plt.plot([65, 65, -65, -65, 65], [65, -65, -65, 65, 65], lw=0.5)
            for xx in range(200):
                for yy in range(200):
                    xc = -50 + xx * 0.5
                    yc = -50 + yy * 0.5
                    if labels[0, xx, yy] == 1:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="blue"))
                    if labels[1, xx, yy] == 1:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="green"))
                    if labels[2, xx, yy] == 1:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="red"))
            for i in range(corner.shape[0]):
                x1 = corner[i][0][0]
                y1 = corner[i][0][1]
                x2 = corner[i][2][0]
                y2 = corner[i][2][1]
                x3 = corner[i][6][0]
                y3 = corner[i][6][1]
                x4 = corner[i][4][0]
                y4 = corner[i][4][1]
                plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], lw=0.5)
            plt.savefig("/home/xiazhongyu/vis/gt"+str(random.randint(1, 9999))+".png")
            print('\n******** END PRINT GT**********\n')
            # exit(0)

        return data


@PIPELINES.register_module()
class PrepareImageInputsVOD(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
            self,
            data_config,
            is_train=False,
            sequential=False,
            ego_cam='CAM_FRONT',
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.ego_cam = ego_cam

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sensor2ego_transformation(self,
                                      cam_info,
                                      key_info,
                                      cam_name,
                                      ego_cam=None):
        if ego_cam is None:
            ego_cam = cam_name
        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
        # sweep sensor to sweep ego
        sweepsensor2sweepego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepsensor2sweepego_tran = torch.Tensor(
            cam_info['cams'][cam_name]['sensor2ego_translation'])
        sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros((4, 4))
        sweepsensor2sweepego[3, 3] = 1
        sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
        sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        sweepego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepego2global_tran = torch.Tensor(
            cam_info['cams'][cam_name]['ego2global_translation'])
        sweepego2global = sweepego2global_rot.new_zeros((4, 4))
        sweepego2global[3, 3] = 1
        sweepego2global[:3, :3] = sweepego2global_rot
        sweepego2global[:3, -1] = sweepego2global_tran

        # global sensor to cur ego
        w, x, y, z = key_info['cams'][ego_cam]['ego2global_rotation']
        keyego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keyego2global_tran = torch.Tensor(
            key_info['cams'][ego_cam]['ego2global_translation'])
        keyego2global = keyego2global_rot.new_zeros((4, 4))
        keyego2global[3, 3] = 1
        keyego2global[:3, :3] = keyego2global_rot
        keyego2global[:3, -1] = keyego2global_tran
        global2keyego = keyego2global.inverse()

        sweepsensor2keyego = \
            global2keyego @ sweepego2global @ sweepsensor2sweepego

        # global sensor to cur ego
        w, x, y, z = key_info['cams'][cam_name]['ego2global_rotation']
        keyego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keyego2global_tran = torch.Tensor(
            key_info['cams'][cam_name]['ego2global_translation'])
        keyego2global = keyego2global_rot.new_zeros((4, 4))
        keyego2global[3, 3] = 1
        keyego2global[:3, :3] = keyego2global_rot
        keyego2global[:3, -1] = keyego2global_tran
        global2keyego = keyego2global.inverse()

        # cur ego to sensor
        w, x, y, z = key_info['cams'][cam_name]['sensor2ego_rotation']
        keysensor2keyego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keysensor2keyego_tran = torch.Tensor(
            key_info['cams'][cam_name]['sensor2ego_translation'])
        keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
        keysensor2keyego[3, 3] = 1
        keysensor2keyego[:3, :3] = keysensor2keyego_rot
        keysensor2keyego[:3, -1] = keysensor2keyego_tran
        keyego2keysensor = keysensor2keyego.inverse()
        keysensor2sweepsensor = (
                keyego2keysensor @ global2keyego @ sweepego2global
                @ sweepsensor2sweepego).inverse()
        return sweepsensor2keyego, keysensor2sweepsensor

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []
        sensor2sensors = []
        for cam_name in cam_names:
            # import ipdb; ipdb.set_trace()
            # cam_data = results['curr']['calib']['P2']
            # filename = cam_data['data_path']
            filename = results['curr']['image']['image_path']
            img = Image.open('data/view_of_delft_PUBLIC/radar_5frames/' + filename)

            # self.data_config['input_size'] = img.size
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            # import ipdb; ipdb.set_trace()
            # TODO check the camera parameters
            # intrin = torch.Tensor(results['curr']['calib']['P2']*
            #                       results['curr']['calib']['R0_rect'])[:3, :3]
            intrin = torch.Tensor(results['curr']['calib']['P2'])[:3, :3]
            # intrin = torch.Tensor(results['curr']['calib']['P2'])[:3, :4]

            # sensor2keyego, sensor2sensor = \
            #     self.get_sensor2ego_transformation(results['curr'],
            #                                        results['curr'],
            #                                        cam_name,
            #                                        self.ego_cam)

            # rot = torch.eye(3)
            # tran = torch.zeros(3)
            rect = results['ann_info']['calib']['R0_rect'].astype(np.float32)
            Trv2c = results['ann_info']['calib']['Tr_velo_to_cam'].astype(np.float32)
            # TODO check coordinate
            cam2Trv = torch.tensor(rect @ Trv2c).inverse()
            # breakpoint()
            # cam2Trv = torch.tensor(rect @ Trv2c)
            rot = cam2Trv[:3, :3]
            tran = cam2Trv[:3, 3]

            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            canvas.append(np.array(img))
            transform = transforms.ToTensor()
            imgs.append(transform(img))  # self.normalize_img()

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            # sensor2sensors.append(sensor2sensor)

        if self.sequential:
            for adj_info in results['adjacent']:
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # align
                trans_adj = []
                rots_adj = []
                sensor2sensors_adj = []
                for cam_name in cam_names:
                    adjsensor2keyego, sensor2sensor = \
                        self.get_sensor2ego_transformation(adj_info,
                                                           results['curr'],
                                                           cam_name,
                                                           self.ego_cam)
                    rot = adjsensor2keyego[:3, :3]
                    tran = adjsensor2keyego[:3, 3]
                    rots_adj.append(rot)
                    trans_adj.append(tran)
                    sensor2sensors_adj.append(sensor2sensor)
                rots.extend(rots_adj)
                trans.extend(trans_adj)
                sensor2sensors.extend(sensor2sensors_adj)

        imgs = torch.stack(imgs)
        rots = torch.stack(rots)
        trans = torch.stack(trans)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)

        # array_to_draw = tensor_to_draw.cpu().numpy()

        # import matplotlib.pyplot as plt

        # plt.imshow(array_to_draw.transpose(1, 2, 0))
        # plt.savefig(f'img_tensor_.png')
        # bda = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(8, 1, 1, 1).to(imgs).float()
        # sensor2sensors = torch.stack(sensor2sensors)

        results['canvas'] = canvas
        # results['sensor2sensors'] = sensor2sensors
        return (imgs, rots, trans, intrins, post_rots, post_trans)

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results


@PIPELINES.register_module()
class PrepareImageInputsVODDebug(PrepareImageInputs):
    def get_inputs(self, results, flip=None, scale=None):
        assert not self.sequential
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        # cam_names = self.choose_cams()
        # results['cam_names'] = cam_names
        filename = results['img_info']['filename']

        canvas = []
        # for cam_name in cam_names:
        # cam_data = results['curr']['cams'][cam_name]
        # filename = cam_data['data_path']
        img = Image.open(filename)
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)

        intrin = torch.Tensor(results['P2'])[..., :3, :3]

        # ego2global = torch.inverse(torch.Tensor(results['Trv2c']))
        # sensor2ego = torch.eye(4)
        sensor2ego = torch.inverse(torch.Tensor(results['Trv2c']))
        ego2global = torch.eye(4)
        # image view augmentation (resize, crop, horizontal flip, rotate)
        img_augs = self.sample_augmentation(
            H=img.height, W=img.width, flip=flip, scale=scale)
        resize, resize_dims, crop, flip, rotate = img_augs
        img, post_rot2, post_tran2 = \
            self.img_transform(img, post_rot,
                               post_tran,
                               resize=resize,
                               resize_dims=resize_dims,
                               crop=crop,
                               flip=flip,
                               rotate=rotate)

        # for convenience, make augmentation matrices 3x3
        post_tran = torch.zeros(3)
        post_rot = torch.eye(3)
        post_tran[:2] = post_tran2
        post_rot[:2, :2] = post_rot2
        # print(cam_name, self.ignore)

        canvas.append(np.array(img))
        imgs.append(self.normalize_img(img))

        intrins.append(intrin)
        sensor2egos.append(sensor2ego)
        ego2globals.append(ego2global)
        post_rots.append(post_rot)
        post_trans.append(post_tran)

        imgs = torch.stack(imgs)

        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        results['canvas'] = canvas
        return (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans)


@PIPELINES.register_module()
class LoadAnnotationsBEVDepthVOD(object):

    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                    rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:,
                6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                         6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            # velocity
            # gt_boxes[:, 7:] = (
            #     rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat

    def __call__(self, results):
        # import ipdb; ipdb.set_trace()
        ann_info = results['ann_info']
        # results['ann_info']['gt_bboxes_3d']
        gt_boxes, gt_labels = ann_info['gt_bboxes_3d'], ann_info['gt_labels_3d']
        gt_boxes, gt_labels = gt_boxes.tensor.clone(), torch.tensor(gt_labels)
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(
        )
        results['rotate_bda'] = rotate_bda
        results['scale_bda'] = scale_bda
        results['flip_dx'] = flip_dx
        results['flip_dy'] = flip_dy  # save
        # print(rotate_bda)
        # print(scale_bda)
        # print(flip_dx, flip_dy)

        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 7)
        # results['gt_bboxes_3d'] = CameraInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1])
        results['gt_bboxes_3d'] = LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1])
        #  origin=(0.5, 0.5, 0.))
        results['gt_labels_3d'] = gt_labels
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans = results['img_inputs'][4:]
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots,
                                 post_trans, bda_rot)
        return results


@PIPELINES.register_module()
class LoadAnnotationsBEVDepthLidarPre(object):

    def __init__(self, classes=None, img_info_prototype='bevdet',):

        self.classes = classes # unused
        self.img_info_prototype = img_info_prototype
        assert img_info_prototype in ['bevdet', 'mmcv']

    def __call__(self, results):
        if self.img_info_prototype == 'bevdet':
            gt_boxes, gt_labels = results['ann_infos']
            gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)

            if len(gt_boxes) == 0:
                gt_boxes = torch.zeros(0, 9)

            results['gt_bboxes_3d'] = \
                LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1])
            results['gt_labels_3d'] = gt_labels
        else:
            results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']

            box = results['ann_info']['gt_bboxes_3d']  #change infos

            if len(box.tensor) != 0:

                lidar2ego = results['lidar2ego']
                ego2global = results['ego2global']
                cam_ego2global = results['cam_ego2global']
                # print(box[0])

                box.tensor[:, 2] += box.tensor[:, 5]*0.5

                box.rotate(lidar2ego[:3, :3].T)
                box.translate(lidar2ego[:3, 3].reshape(1, 3))    
                
                box.rotate(ego2global[:3, :3].T)
                box.translate(ego2global[:3, 3].reshape(1, 3))

                box.translate(-cam_ego2global[:3, 3].reshape(1, 3))
                box.rotate(cam_ego2global[:3, :3])

                yaw = box.tensor[:, 6]
                yaw = limit_period(yaw, period=np.pi * 2)
                box.tensor[:, 6] = yaw

                # box.tensor[:, 2] -= box.tensor[:, 5]*0.5

            results['gt_bboxes_3d'] = box
            # results['bbox3d_fields'].append('gt_bboxes_3d')

        # gt_boxes_origin = results['ann_info']['gt_bboxes_3d']

        if 'points' in results:
            points = results['points']

            # old
            # lidar2ego = results['lidar2ego']
            # points.rotate(lidar2ego[:3, :3].T)
            # points.tensor[:, :3] = points.tensor[:, :3] + lidar2ego[:3, 3]

            lidar2ego = results['lidar2ego']
            ego2global = results['ego2global']
            cam_ego2global = results['cam_ego2global']

            points.rotate(lidar2ego[:3, :3].T)
            points.tensor[:, :3] = points.tensor[:, :3] + lidar2ego[:3, 3]
            
            points.rotate(ego2global[:3, :3].T)
            points.tensor[:, :3] = points.tensor[:, :3] + ego2global[:3, 3]

            points.tensor[:, :3] = points.tensor[:, :3] - cam_ego2global[:3, 3]
            points.rotate(cam_ego2global[:3, :3])

            results['points'] = points

        return results


@PIPELINES.register_module()
class LoadAnnotationsBEVDepthLidarPost(object):

    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']

            translation_std = np.array(self.bda_aug_conf['trans_xyz'], dtype=np.float32)
            trans_bda = np.random.normal(scale=translation_std, size=3).reshape(-1)

            # input_dict['points'].translate(trans_factor)
            # input_dict['pcd_trans'] = trans_factor
            # for key in input_dict['bbox3d_fields']:
            #     input_dict[key].translate(trans_factor)
            # trans_x = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
            # trans_x = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
            # trans_x = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
            trans_bda = np.array([0., 0., 0.]).reshape(-1)
        return rotate_bda, scale_bda, flip_dx, flip_dy, trans_bda

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy, trans_bda):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        trans_mat = torch.Tensor(trans_bda)

        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        fsr_mat = flip_mat @ (scale_mat @ rot_mat)

        sr_mat = scale_mat @ rot_mat

        if gt_boxes.shape[0] > 0:
            # gt_boxes[:, :3] = (
            #     fsr_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)

            gt_boxes[:, :3] = (
                sr_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)

            gt_boxes[:, :3] += trans_mat.view(1, -1)
            
            gt_boxes[:, :3] = (
                flip_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)


            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:,
                         6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                                           6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                fsr_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, fsr_mat, rot_mat, flip_mat, scale_mat, trans_mat

    def __call__(self, results):
        # gt_boxes, gt_labels = results['ann_infos']
        # gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
        gt_boxes = results['gt_bboxes_3d'].tensor
        gt_labels = results['gt_labels_3d']

        rotate_bda, scale_bda, flip_dx, flip_dy, trans_bda = self.sample_bda_augmentation(
        )
        results['rotate_bda']=rotate_bda
        results['scale_bda']=scale_bda
        results['flip_dx']=flip_dx
        results['flip_dy']=flip_dy #save
        results['trans_bda']=trans_bda


        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot, rm, fm, sm, tm = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy, trans_bda)
        bda_mat[:3, :3] = rm
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels

        if 'points' in results:
            points = results['points']

            points.tensor[:, :3] = (sm @ (rm @ points.tensor[:, :3].unsqueeze(-1))).squeeze(-1)
            points.tensor[:, :3] += tm.view(1, -1)
            points.tensor[:, :3] = (fm @ points.tensor[:, :3].unsqueeze(-1)).squeeze(-1)
            
            results['points'] = points

            if False:  # for debug vis
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                import random
                print('\n******** BEGIN PRINT GT and LiDAR POINTS **********\n')
                corner = results['gt_bboxes_3d'].corners
                fig = plt.figure(figsize=(16, 16))
                plt.plot([50, 50, -50, -50, 50], [50, -50, -50, 50, 50], lw=0.5)
                plt.plot([65, 65, -65, -65, 65], [65, -65, -65, 65, 65], lw=0.5)
                for i in range(corner.shape[0]):
                    x1 = corner[i][0][0]
                    y1 = corner[i][0][1]
                    x2 = corner[i][2][0]
                    y2 = corner[i][2][1]
                    x3 = corner[i][6][0]
                    y3 = corner[i][6][1]
                    x4 = corner[i][4][0]
                    y4 = corner[i][4][1]
                    plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], lw=1)
                plt.scatter(points.tensor[:, 0].view(-1).numpy(), points.tensor[:, 1].view(-1).numpy(), s=0.5, c='black')
                plt.savefig("/home/xiazhongyu/vis/gt" + str(random.randint(1, 9999)) + ".png")
                print('\n******** END PRINT GT and LiDAR POINTS **********\n')
                exit(0)

        if 'img_inputs' in results:
            imgs, rots, trans, intrins = results['img_inputs'][:4]
            post_rots, post_trans = results['img_inputs'][4:]
            results['img_inputs'] = (imgs, rots, trans, intrins, post_rots,
                                     post_trans, bda_rot)
            if 'img_inputs_lt' in results.keys():
                imgs_lt, rots_lt, trans_lt, intrins_lt = results['img_inputs_lt'][:4]
                post_rots_lt, post_trans_lt = results['img_inputs_lt'][4:]
                results['img_inputs_lt'] = (imgs_lt, rots_lt, trans_lt, intrins_lt,
                                            post_rots_lt, post_trans_lt, bda_rot)

        results["bda_r"] = bda_mat
        results["bda_f"] = (flip_dx, flip_dy)
        results["bda_s"] = scale_bda
        return results



@PIPELINES.register_module()
class LoadAnnotationsBEVDepthReverse(object):

    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        fsr_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                fsr_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:,
                         6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                                           6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                fsr_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, fsr_mat, rot_mat, flip_mat, scale_mat

    def __call__(self, results):
        gt_boxes, gt_labels = results['ann_infos']
        gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)

        lidar2ego = results['lidar2ego']
        ego2global = results['ego2global']
        cam_ego2global = results['cam_ego2global']
        # print(box[0])

        # box.rotate(lidar2ego[:3, :3].T)
        # box.translate(lidar2ego[:3, 3].reshape(1, 3))    
        
        # box.rotate(ego2global[:3, :3].T)
        # box.translate(ego2global[:3, 3].reshape(1, 3))

        # box.translate(-cam_ego2global[:3, 3].reshape(1, 3))
        # box.rotate(cam_ego2global[:3, :3])
        if self.is_train and len(gt_boxes) != 0:
            box = LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1])
            box.rotate(cam_ego2global[:3, :3].T)
            box.translate(cam_ego2global[:3, 3].reshape(1, 3))

            box.translate(-ego2global[:3, 3].reshape(1, 3))
            box.rotate(ego2global[:3, :3])

            box.translate(-lidar2ego[:3, 3].reshape(1, 3))    
            box.rotate(lidar2ego[:3, :3])

            yaw = box.tensor[:, 6]
            yaw = limit_period(-yaw - np.pi / 2, period=np.pi * 2)
            box.tensor[:, 6] = yaw
            whl = box.tensor[:, 3:6]
            box.tensor[:, 3:6] = whl[:,[1,0,2]]

            gt_boxes = box.tensor

        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(
        )
        results['rotate_bda']=rotate_bda
        results['scale_bda']=scale_bda
        results['flip_dx']=flip_dx
        results['flip_dy']=flip_dy #save


        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot, rm, fm, sm = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_mat[:3, :3] = rm
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels

        # if 'img_inputs' in results:
        #     imgs, rots, trans, intrins = results['img_inputs'][:4]
        #     post_rots, post_trans = results['img_inputs'][4:]
        #     results['img_inputs'] = (imgs, rots, trans, intrins, post_rots,
        #                              post_trans, bda_rot)
        #     if 'img_inputs_lt' in results.keys():
        #         imgs_lt, rots_lt, trans_lt, intrins_lt = results['img_inputs_lt'][:4]
        #         post_rots_lt, post_trans_lt = results['img_inputs_lt'][4:]
        #         results['img_inputs_lt'] = (imgs_lt, rots_lt, trans_lt, intrins_lt,
        #                                     post_rots_lt, post_trans_lt, bda_rot)

        results["bda_r"] = bda_mat
        results["bda_f"] = (flip_dx, flip_dy)
        results["bda_s"] = scale_bda
        return results


@PIPELINES.register_module()
class LoadAnnotations3DDebug(LoadAnnotations3D):
    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        # print(type(results['ann_infos'])) #test
        box = results['ann_info']['gt_bboxes_3d']  # change infos

        lidar2ego = results['lidar2ego']
        ego2global = results['ego2global']
        cam_ego2global = results['cam_ego2global']
        # print(box[0])

        box.tensor[:, 2] += box.tensor[:, 5] * 0.5

        box.rotate(lidar2ego[:3, :3].T)
        box.translate(lidar2ego[:3, 3].reshape(1, 3))

        box.rotate(ego2global[:3, :3].T)
        box.translate(ego2global[:3, 3].reshape(1, 3))

        box.translate(-cam_ego2global[:3, 3].reshape(1, 3))
        box.rotate(cam_ego2global[:3, :3])

        yaw = box.tensor[:, 6]
        yaw = limit_period(yaw, period=np.pi * 2)
        box.tensor[:, 6] = yaw

        box.tensor[:, 2] -= box.tensor[:, 5] * 0.5

        results['gt_bboxes_3d'] = box
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        if 'ann_info' in results:
            results = super().__call__(results)

        assert 'points' in results
        points = results['points']

        lidar2ego = results['lidar2ego']
        ego2global = results['ego2global']
        cam_ego2global = results['cam_ego2global']

        points.rotate(lidar2ego[:3, :3].T)
        points.tensor[:, :3] = points.tensor[:, :3] + lidar2ego[:3, 3]

        points.rotate(ego2global[:3, :3].T)
        points.tensor[:, :3] = points.tensor[:, :3] + ego2global[:3, 3]

        points.tensor[:, :3] = points.tensor[:, :3] - cam_ego2global[:3, 3]
        points.rotate(cam_ego2global[:3, :3])

        results['points'] = points

        return results


def compose_lidar2img(ego2global_translation_curr,
                      ego2global_rotation_curr,
                      lidar2ego_translation_curr,
                      lidar2ego_rotation_curr,
                      sensor2global_translation_past,
                      sensor2global_rotation_past,
                      cam_intrinsic_past):
    R = sensor2global_rotation_past @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T)
    T = sensor2global_translation_past @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T)
    T -= ego2global_translation_curr @ (
                inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T) + lidar2ego_translation_curr @ inv(
        lidar2ego_rotation_curr).T

    lidar2cam_r = inv(R.T)
    lidar2cam_t = T @ lidar2cam_r.T

    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4)
    viewpad[:cam_intrinsic_past.shape[0], :cam_intrinsic_past.shape[1]] = cam_intrinsic_past
    lidar2img = (viewpad @ lidar2cam_rt.T).astype(np.float32)

    return lidar2img


@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweeps(object):
    def __init__(self,
                 sweeps_num=5,
                 color_type='color',
                 test_mode=False):
        self.sweeps_num = sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode

        self.train_interval = [4, 8]
        self.test_interval = 6

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def load_offline(self, results):
        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
        else:
            if self.test_mode:
                interval = self.test_interval
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]
            elif len(results['sweeps']['prev']) <= self.sweeps_num:
                pad_len = self.sweeps_num - len(results['sweeps']['prev'])
                choices = list(range(len(results['sweeps']['prev']))) + [len(results['sweeps']['prev']) - 1] * pad_len
            else:
                max_interval = len(results['sweeps']['prev']) // self.sweeps_num
                max_interval = min(max_interval, self.train_interval[1])
                min_interval = min(max_interval, self.train_interval[0])
                interval = np.random.randint(min_interval, max_interval + 1)
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        return results

    def load_online(self, results):
        # only used when measuring FPS
        assert self.test_mode
        assert self.test_interval == 6

        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
        else:
            interval = self.test_interval
            choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    # skip loading history frames
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        return results

    def __call__(self, results):
        if self.sweeps_num == 0:
            return results

        world_size = get_dist_info()[1]
        if world_size == 1 and self.test_mode:
            return self.load_online(results)
        else:
            return self.load_offline(results)


@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweepsFuture(object):
    def __init__(self,
                 prev_sweeps_num=5,
                 next_sweeps_num=5,
                 color_type='color',
                 test_mode=False):
        self.prev_sweeps_num = prev_sweeps_num
        self.next_sweeps_num = next_sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode

        assert prev_sweeps_num == next_sweeps_num

        self.train_interval = [4, 8]
        self.test_interval = 6

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def __call__(self, results):
        if self.prev_sweeps_num == 0 and self.next_sweeps_num == 0:
            return results

        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if self.test_mode:
            interval = self.test_interval
        else:
            interval = np.random.randint(self.train_interval[0], self.train_interval[1] + 1)

        # previous sweeps
        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.prev_sweeps_num):
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.prev_sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(sweep[sensor]['data_path'])
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        # future sweeps
        if len(results['sweeps']['next']) == 0:
            for _ in range(self.next_sweeps_num):
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.next_sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['next']) - 1)
                sweep = results['sweeps']['next'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['next'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(sweep[sensor]['data_path'])
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        return results


@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweepsFutureInterleave(object):
    def __init__(self,
                 prev_sweeps_num=5,
                 next_sweeps_num=5,
                 color_type='color',
                 test_mode=False):
        self.prev_sweeps_num = prev_sweeps_num
        self.next_sweeps_num = next_sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode

        assert prev_sweeps_num == next_sweeps_num

        self.train_interval = [4, 8]
        self.test_interval = 6

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def __call__(self, results):
        if self.prev_sweeps_num == 0 and self.next_sweeps_num == 0:
            return results

        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if self.test_mode:
            interval = self.test_interval
        else:
            interval = np.random.randint(self.train_interval[0], self.train_interval[1] + 1)

        results_prev = dict(
            img=[],
            img_timestamp=[],
            filename=[],
            lidar2img=[]
        )
        results_next = dict(
            img=[],
            img_timestamp=[],
            filename=[],
            lidar2img=[]
        )

        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.prev_sweeps_num):
                for j in range(len(cam_types)):
                    results_prev['img'].append(results['img'][j])
                    results_prev['img_timestamp'].append(results['img_timestamp'][j])
                    results_prev['filename'].append(results['filename'][j])
                    results_prev['lidar2img'].append(np.copy(results['lidar2img'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.prev_sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results_prev['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results_prev['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results_prev['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results_prev['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        if len(results['sweeps']['next']) == 0:
            print(1, len(results_next['img']) )
            for _ in range(self.next_sweeps_num):
                for j in range(len(cam_types)):
                    results_next['img'].append(results['img'][j])
                    results_next['img_timestamp'].append(results['img_timestamp'][j])
                    results_next['filename'].append(results['filename'][j])
                    results_next['lidar2img'].append(np.copy(results['lidar2img'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.next_sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['next']) - 1)
                sweep = results['sweeps']['next'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['next'][sweep_idx - 1]

                for sensor in cam_types:
                    results_next['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results_next['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results_next['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results_next['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        assert len(results_prev['img']) % 6 == 0
        assert len(results_next['img']) % 6 == 0

        for i in range(len(results_prev['img']) // 6):
            for j in range(6):
                results['img'].append(results_prev['img'][i * 6 + j])
                results['img_timestamp'].append(results_prev['img_timestamp'][i * 6 + j])
                results['filename'].append(results_prev['filename'][i * 6 + j])
                results['lidar2img'].append(results_prev['lidar2img'][i * 6 + j])

            for j in range(6):
                results['img'].append(results_next['img'][i * 6 + j])
                results['img_timestamp'].append(results_next['img_timestamp'][i * 6 + j])
                results['filename'].append(results_next['filename'][i * 6 + j])
                results['lidar2img'].append(results_next['lidar2img'][i * 6 + j])

        return results


@PIPELINES.register_module()
class LoadRadarPointsMultiSweepsFuture(object):
    """Load radar points from multiple sweeps.
    This is usually used for nuScenes dataset to utilize previous sweeps.
    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 load_dim=18,
                 use_dim=[0, 1, 2, 3, 4],
                 sweeps_num=3, 
                 future_sweeps_num=7, 
                 file_client_args=dict(backend='disk'),
                 max_num=300,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 
                 test_mode=False,
                 rote90=True,
                 ignore=[]):
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.sweeps_num = sweeps_num
        self.future_sweeps_num = future_sweeps_num
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.max_num = max_num
        self.test_mode = test_mode
        self.pc_range = pc_range
        self.ignore = ignore
        self.rote90 = rote90
        if len(ignore)>0:
            print(self.ignore)

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.
        Args:
            pts_filename (str): Filename of point clouds data.
        Returns:
            np.ndarray: An array containing point clouds data.
            [N, 18]
        """
        radar_obj = RadarPointCloud.from_file(pts_filename)

        #[18, N]
        points = radar_obj.points

        return points.transpose().astype(np.float32)
        

    def _pad_or_drop(self, points):
        '''
        points: [N, 18]
        ''' 

        num_points = points.shape[0]

        if num_points == self.max_num:
            masks = np.ones((num_points, 1), 
                        dtype=points.dtype)

            return points, masks
        
        if num_points > self.max_num:
            points = np.random.permutation(points)[:self.max_num, :]
            masks = np.ones((self.max_num, 1), 
                        dtype=points.dtype)
            
            return points, masks

        if num_points < self.max_num:
            zeros = np.zeros((self.max_num - num_points, points.shape[1]), 
                        dtype=points.dtype)
            masks = np.ones((num_points, 1), 
                        dtype=points.dtype)
            
            points = np.concatenate((points, zeros), axis=0)
            masks = np.concatenate((masks, zeros.copy()[:, [0]]), axis=0)

            return points, masks

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.
        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.
        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.
                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        radars_dict = results['radar']
        # print(radars_dict.keys())

        points_sweep_list = []
        for key, sweeps in radars_dict.items():
            if key in self.ignore:
                continue
            if len(sweeps) < self.sweeps_num:
                idxes = list(range(len(sweeps)))
                print(f'pkl sweeps({sweeps})<needed sweeps num({self.sweeps_num})')
            else:
                idxes = list(range(self.sweeps_num))
            
            ts = sweeps[0]['timestamp'] * 1e-6
            for idx in idxes:
                sweep = sweeps[idx]

                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                timestamp = sweep['timestamp'] * 1e-6
                time_diff = ts - timestamp
                # print(time_diff)
                time_diff = np.ones((points_sweep.shape[0], 1)) * time_diff

                # velocity compensated by the ego motion in sensor frame
                velo_comp = points_sweep[:, 8:10]
                velo_comp = np.concatenate(
                    (velo_comp, np.zeros((velo_comp.shape[0], 1))), 1)
                velo_comp = velo_comp @ sweep['sensor2lidar_rotation'].T
                velo_comp = velo_comp[:, :2]

                # velocity in sensor frame
                velo = points_sweep[:, 6:8]
                velo = np.concatenate(
                    (velo, np.zeros((velo.shape[0], 1))), 1)
                velo = velo @ sweep['sensor2lidar_rotation'].T
                velo = velo[:, :2]

                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                # print()
                points_sweep_ = np.concatenate(
                    [points_sweep[:, :6], velo,
                     velo_comp, points_sweep[:, 10:],
                     time_diff], axis=1)
                points_sweep_list.append(points_sweep_)
        
        # future 
        radars_dict_future = results['radars_future']
        for key, sweeps in radars_dict_future.items():
            if key in self.ignore:
                continue
            if len(sweeps) < self.future_sweeps_num:
                idxes = list(range(len(sweeps)))
                print(f'pkl sweeps({sweeps})<needed future sweeps num({self.future_sweeps_num})')
            else:
                idxes = list(range(self.future_sweeps_num))
            
            # ts = sweeps[0]['timestamp'] * 1e-6
            for idx in idxes:
                sweep = sweeps[idx]

                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                timestamp = sweep['timestamp'] * 1e-6
                time_diff = ts - timestamp
                # print(time_diff)
                time_diff = np.ones((points_sweep.shape[0], 1)) * time_diff

                # velocity compensated by the ego motion in sensor frame
                velo_comp = points_sweep[:, 8:10]
                velo_comp = np.concatenate(
                    (velo_comp, np.zeros((velo_comp.shape[0], 1))), 1)
                velo_comp = velo_comp @ sweep['sensor2lidar_rotation'].T
                velo_comp = velo_comp[:, :2]

                # velocity in sensor frame
                velo = points_sweep[:, 6:8]
                velo = np.concatenate(
                    (velo, np.zeros((velo.shape[0], 1))), 1)
                velo = velo @ sweep['sensor2lidar_rotation'].T
                velo = velo[:, :2]

                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                # print()
                points_sweep_ = np.concatenate(
                    [points_sweep[:, :6], velo,
                     velo_comp, points_sweep[:, 10:],
                     time_diff], axis=1)
                points_sweep_list.append(points_sweep_)
        
        points = np.concatenate(points_sweep_list, axis=0)
        # print(points.shape)
        
        points = points[:, self.use_dim]
        
        #print(points.shape[-1])

        points = RadarPoints(
            points, points_dim=points.shape[-1], attribute_dims=None
        )
        if(self.rote90):
            points.rotate(-math.pi/2) #
        results['radar'] = points
        return results
    
    
    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'