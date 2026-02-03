from av2.datasets.sensor.sensor_dataloader import LIDAR_PATTERN
from av2.datasets.sensor.utils import convert_path_to_named_record
from pathlib import Path
import pandas as pd
import av2.utils.io as io_utils
from av2.datasets.sensor.constants import RingCameras
from av2.utils.synchronization_database import SynchronizationDB
from av2.datasets.sensor.constants import AnnotationCategories
from pathlib import Path
from typing import Dict, Final
import torch
import mmcv
import numpy as np
from av2.utils.io import read_feather
from av2.structures.cuboid import CuboidList
from av2.geometry.geometry import quat_to_mat, mat_to_xyz
from os import path as osp
from tqdm import tqdm
from shapely.geometry import MultiPoint, box
from av2.utils.io import  read_city_SE3_ego, read_feather
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.evaluation.detection.constants import CompetitionCategories

dataset_dir = Path("data/av2") # replace with absolute path
CAMERAS = tuple(x.value for x in RingCameras)
LABEL_ATTR = (
    "tx_m","ty_m","tz_m","length_m","width_m","height_m","qw","qx","qy","qz",
)
AV2_ANNO_NAMES_TO_INDEX: Final[Dict[str, int]] = {
    x.value: i for i, x in enumerate(AnnotationCategories)
}

DATASET_TO_TAXONOMY: Final[Dict[str, Dict[str, int]]] = {
    "av2": AV2_ANNO_NAMES_TO_INDEX,
}
CompetitionCLASSES = tuple(x.value for x in CompetitionCategories)

def create_av2_infos(dataset_dir, split, out_dir):
    src_dir = dataset_dir / split
    paths = sorted(src_dir.glob(LIDAR_PATTERN), key=lambda x: int(x.stem))   #每个split包含的所有lidar帧路径
    records = [convert_path_to_named_record(p) for p in paths]      #每帧信息：split，log_id, sensor_name, timestamp_ns
    sensor_caches = pd.DataFrame(records)  #pandas形式
    sensor_caches.set_index(["log_id", "sensor_name", "timestamp_ns"], inplace=True) #将相同场景id的对应帧放在一起
    sensor_caches.sort_index(inplace=True) #根据场景id排序
    loader = SynchronizationDB(dataset_dir=src_dir)
    loader.MAX_LIDAR_RING_CAM_TIMESTAMP_DIFF *= 4  # from 25ms to 100ms, ensure cam!=None for every val frame.

    av2_split_infos = []
    for i in tqdm(range(len(sensor_caches))): 
        # if i % 5000 != 0: # to create mini pkl for debug
        #     continue
        infos = {}
        record = sensor_caches.iloc[i].name
        log_id, _, lidar_timestamp_ns = record
        log_dir = src_dir / log_id
        timestamp_city_SE3_ego_dict = read_city_SE3_ego(log_dir=log_dir)
        city_SE3_ego_lidar_t = timestamp_city_SE3_ego_dict[lidar_timestamp_ns]
        infos['scene_id'] = log_id
        infos['lidar_timestamp_ns'] = lidar_timestamp_ns
        infos['city_SE3_ego_lidar_t'] = city_SE3_ego_lidar_t
        cam_infos = {}
        camera_models = {}
        for cam_name in CAMERAS:
            # 根据lidar timestamp取对应相机的timestamp
            cam_timestamp_ns = loader.get_closest_cam_channel_timestamp(
                lidar_timestamp=lidar_timestamp_ns, cam_name=cam_name, log_id=log_id)
            if cam_timestamp_ns is None:
                # print("No corresponding ring image")
                cam_infos[cam_name] = None
                camera_models[cam_name] = None
                continue
            
            fpath = Path(split) / log_id / "sensors" / "cameras" / cam_name / f"{cam_timestamp_ns}.jpg" #img path
            camera_model = PinholeCamera.from_feather(log_dir=log_dir, cam_name=cam_name)
            intrinsics_path = log_dir / "calibration" / "intrinsics.feather"
            intrinsics_df = io_utils.read_feather(intrinsics_path).set_index("sensor_name")
            params = intrinsics_df.loc[cam_name]
            intrinsics = intrinsics_matrix(
                fx=params["fx_px"],
                fy=params["fy_px"],
                cx=params["cx_px"],
                cy=params["cy_px"],
            ) #内参矩阵
            distortion = intrinsics_df.loc[cam_name, ["k1", "k2", "k3"]] #畸变系数
            sensor_name_to_pose = io_utils.read_ego_SE3_sensor(log_dir)
            ego_SE3_cam = sensor_name_to_pose[cam_name] #cam2ego
            city_SE3_ego_cam_t = timestamp_city_SE3_ego_dict[cam_timestamp_ns] #ego2glo at cam timestamp
            
            cam_infos[cam_name] = dict(
                fpath=fpath,
                cam_timestamp_ns=cam_timestamp_ns,
                ego_SE3_cam=ego_SE3_cam,
                intrinsics=intrinsics,
                cam_distortion=distortion,
                city_SE3_ego_cam_t=city_SE3_ego_cam_t,
            )
            camera_models[cam_name] = camera_model
            
        infos['cam_infos'] = cam_infos #用于制作2d标签

        if split != 'test':
            annotations = _load_annotations(split, log_id, lidar_timestamp_ns)
            gt3d_infos = get_gt3d_data(dataset_dir=dataset_dir, split=split, log_id=log_id, 
                                       timestamp_ns=lidar_timestamp_ns)
            gt2d_infos = get_gt2d_data(camera_models, cam_infos, timestamp_city_SE3_ego_dict, lidar_timestamp_ns, 
                                       annotations)
            infos['gt3d_infos'] = gt3d_infos
            infos['gt2d_infos'] = gt2d_infos
        
        #存储所有samples的info
        av2_split_infos.append(infos)
    print('{}_sample:{}'.format(split, len(av2_split_infos)))
    data = dict(infos=av2_split_infos, split=split)
    info_path = osp.join(out_dir, 'av2_{}_infos.pkl'.format(split))
    mmcv.dump(data, info_path)
        

def _load_annotations(split, log_id, timestamp_ns):
    annotations_feather_path = dataset_dir / split / log_id / "annotations.feather"

    # Load annotations from disk.
    # NOTE: This contains annotations for the ENTIRE sequence.
    # The sweep annotations are selected below.
    cuboid_list = CuboidList.from_feather(annotations_feather_path)
    cuboids = list(filter(lambda x: x.timestamp_ns == timestamp_ns, cuboid_list.cuboids))
    return CuboidList(cuboids=cuboids)

def intrinsics_matrix(fx, fy, cx, cy):
    K = np.eye(3, dtype=float)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K

def get_gt3d_data(dataset_dir, split, log_id, timestamp_ns):

    gt3d_infos = {}
    log_dir = dataset_dir / split / log_id

    annotations_path = log_dir / "annotations.feather"
    annotations = read_feather(annotations_path)
    
    annotations = annotations[annotations["timestamp_ns"] == timestamp_ns]

    #xyzlwh+yaw
    cuboid_params = torch.as_tensor(
        annotations.loc[:, list(LABEL_ATTR)].to_numpy(),
        dtype=torch.float,
    )
    rot_mat = quat_to_mat(cuboid_params[:, -4:])
    rot = mat_to_xyz(rot_mat)[..., -1]
    gt_boxes = torch.cat((cuboid_params[:, :-4], torch.as_tensor(rot).unsqueeze(1)), dim=-1)

    names =[label_class
            for label_class in annotations["category"].to_numpy()]
    
    num_interior_pts =torch.as_tensor(
        [num_interior_pt
            for num_interior_pt in annotations["num_interior_pts"].to_numpy()],
        dtype=torch.long,
    )
    
    gt3d_infos['gt_boxes'] = gt_boxes
    gt3d_infos['gt_names'] = names
    gt3d_infos['num_interior_pts'] = num_interior_pts

    return gt3d_infos 

def get_gt2d_data(synchronized_imagery, cam_infos, timestamp_city_SE3_ego_dict, lidar_timestamp_ns, annotations):
    cam_name_to_img = {}
    if synchronized_imagery is not None:    
        gt_2dbboxes_cams = []
        gt_2dlabels_cams = []
        centers2d_cams = []
        depths_cams = []
        for _, cam in synchronized_imagery.items():
            if cam is None:
                continue
            gt_2dbboxes = []
            gt_2dlabels = []
            centers2d = []
            depths = []
            
            city_SE3_ego_cam_t = timestamp_city_SE3_ego_dict[cam_infos[cam.cam_name]['cam_timestamp_ns']] #ego2glo at cam timestamp
            city_SE3_ego_lidar_t = timestamp_city_SE3_ego_dict[lidar_timestamp_ns] #ego2glo at lidar timestamp
            centers3d = annotations.xyz_center_m # N, 3
            categories = annotations.categories
            categories_idx =np.array([label_class
                             for label_class in categories])
            cuboids_vertices_ego = annotations.vertices_m # 8 corners of annotations
            _, V, D = cuboids_vertices_ego.shape #N, 8, 3
            
            if city_SE3_ego_cam_t is not None and city_SE3_ego_lidar_t is not None: 
                uv, centers3d_cam, _ = cam.project_ego_to_img_motion_compensated(
                    centers3d,
                    city_SE3_ego_cam_t=city_SE3_ego_cam_t,
                    city_SE3_ego_lidar_t=city_SE3_ego_lidar_t,
                    ) #centers3d -> centers2d
                
                # 3d points with valid projections have a positive z-coordinate (lying in
                # front of the camera frustum)
                is_valid = np.where(centers3d_cam[:, 2] > 0)
                uv = uv[is_valid]
                centers3d_cam = centers3d_cam[is_valid]
                categories_idx = [categories_idx[i]for i in is_valid][0]
                cuboids_vertices_ego = cuboids_vertices_ego[is_valid, ...]
                
                cuboids_vertices_ego = cuboids_vertices_ego.reshape(-1, D)                                               
                uv_corners, cuboids_vertices_cam, _ = cam.project_ego_to_img_motion_compensated(
                    cuboids_vertices_ego, 
                    city_SE3_ego_cam_t=city_SE3_ego_cam_t, 
                    city_SE3_ego_lidar_t=city_SE3_ego_lidar_t
                    ) #corners3d -> corners2d
            else:
                uv, centers3d_cam, _ = cam.project_ego_to_img(centers3d)
                is_valid = np.where(centers3d_cam[:, 2] > 0)
                uv = uv[is_valid]
                centers3d_cam = centers3d_cam[is_valid]
                categories_idx = [categories_idx[i]for i in is_valid][0]
                cuboids_vertices_ego = cuboids_vertices_ego[is_valid, ...]
                
                cuboids_vertices_ego = cuboids_vertices_ego.reshape(-1, D)       
                uv_corners, cuboids_vertices_cam, _ = cam.project_ego_to_img(cuboids_vertices_ego)
            
            uv_corners = uv_corners.reshape(-1, V, D - 1)
            cuboids_vertices_cam = cuboids_vertices_cam[..., :-1].reshape(-1, V, D)
            for i in range(cuboids_vertices_cam.shape[0]):
                if not (categories_idx[i] in CompetitionCLASSES):
                    continue
                uv_corner = uv_corners[i]
                corner3d = cuboids_vertices_cam[i]
                in_front = np.argwhere(corner3d[..., 2] > 0).flatten()
                uv_corner = uv_corner[in_front, :].tolist()
                
                final_coords = post_process_coords(uv_corner, imsize=(cam.width_px, cam.height_px))
                if final_coords is None:
                    continue
                else:
                    min_x, min_y, max_x, max_y = final_coords
                    
                gt_2dbboxes.append([min_x, min_y, max_x, max_y])
                gt_2dlabels.append(CompetitionCLASSES.index(categories_idx[i]))
                centers2d.append(uv[i])
                depths.append(centers3d_cam[i, 2])
            gt_2dbboxes = np.array(gt_2dbboxes, dtype=np.float32)
            gt_2dlabels = np.array(gt_2dlabels, dtype=np.int64)
            centers2d = np.array(centers2d, dtype=np.float32)
            depths = np.array(depths, dtype=np.float32)
            gt_2dbboxes_cams.append(gt_2dbboxes)
            gt_2dlabels_cams.append(gt_2dlabels)
            centers2d_cams.append(centers2d)
            depths_cams.append(depths)
            
            cam_name_to_img['gt_2dbboxes'] = gt_2dbboxes_cams
            cam_name_to_img['gt_2dlabels'] = gt_2dlabels_cams
            cam_name_to_img['centers2d'] = centers2d_cams
            cam_name_to_img['depths'] = depths_cams
                      
    return cam_name_to_img

def post_process_coords(corner_coords, imsize = (2048, 1550)):
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords]
        )
        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None
    
if __name__ == '__main__':
    splits = ["val", "train"]
    for split in splits:
        create_av2_infos(
            dataset_dir=dataset_dir,
            split=split,
            out_dir=dataset_dir,
        )

