import os
import numpy as np
import pickle
import mmcv
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def get_calib(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    
    calib = {}
    for line in lines:
        if ':' not in line:
            continue
        key, value = line.split(':')
        calib[key] = np.array([float(x) for x in value.split()])
    
    return calib

def convert_v2x_radar_to_nuscenes(root_path, out_path):
    splits = ['train', 'val']
    
    # NuScenes categories
    categories = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier']
    
    cat_map = {
        'Car': 'car',
        'Truck': 'truck',
        'Bus': 'bus',
        'Pedestrian': 'pedestrian',
        'Cyclist': 'bicycle',
        'Van': 'car', # Map Van to Car
    }

    for split in splits:
        print(f"Processing {split} split...")
        image_set_path = os.path.join(root_path, 'ImageSets', f'{split}.txt')
        with open(image_set_path, 'r') as f:
            frame_ids = [line.strip() for line in f.readlines()]
            
        infos = []
        for frame_id in tqdm(frame_ids):
            calib_path = os.path.join(root_path, 'training', 'calib', f'{frame_id}.txt')
            calib = get_calib(calib_path)
            
            # Intrinsics
            P1 = calib['P1'].reshape(3, 4)
            P2 = calib['P2'].reshape(3, 4)
            P3 = calib['P3'].reshape(3, 4)
            
            # Rectification
            R0_rect = calib['R0_rect'].reshape(3, 3)
            
            # Extrinsics (Velo to Cam)
            Tr_v2c1 = calib['Tr_velo_to_cam1'].reshape(3, 4)
            Tr_v2c2 = calib['Tr_velo_to_cam'].reshape(3, 4)
            Tr_v2c3 = calib['Tr_velo_to_cam3'].reshape(3, 4)
            
            # Radar to Velo
            Tr_r2v = calib['Tr_radar_to_velo'].reshape(3, 4)
            
            radar_bins_root = '/mnt/datasets/V2X-Radar-I/V2X-Radar-I/'
            
            # NuScenes info structure
            info = {
                'lidar_path': os.path.join(root_path, f'training/velodyne/{frame_id}.bin'),
                'token': frame_id,
                'sweeps': [],
                'cams': {},
                'radars': {},
                'lidar2ego_translation': [0, 0, 0],
                'lidar2ego_rotation': [1, 0, 0, 0],
                'ego2global_translation': [0, 0, 0],
                'ego2global_rotation': [1, 0, 0, 0],
                'timestamp': int(frame_id), # Dummy timestamp
                'scene_token': 'scene_0', # Dummy scene
                'location': 'v2x-radar-i',
            }
            
            # Process Cameras
            # Note: NuScenes expects NCams=6. We will provide 3 and maybe 3 dummy or just use 3 in config.
            # Names: CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT
            cam_configs = [
                ('CAM_FRONT_LEFT', 'image_1', P1, Tr_v2c1),
                ('CAM_FRONT', 'image_2', P2, Tr_v2c2),
                ('CAM_FRONT_RIGHT', 'image_3', P3, Tr_v2c3),
            ]
            
            for cam_name, cam_dir, P, Tr in cam_configs:
                # Extrinsic: Cam to Lidar
                # In KITTI: P_cam = R_rect @ Tr_v2c @ P_lidar
                # So Extrinsic_Lidar_to_Cam = R_rect @ Tr_v2c
                extrinsic_l2c = np.eye(4)
                extrinsic_l2c[:3, :3] = R0_rect @ Tr[:3, :3]
                extrinsic_l2c[:3, 3] = R0_rect @ Tr[:3, 3]
                
                extrinsic_c2l = np.linalg.inv(extrinsic_l2c)
                
                # NuScenes expects sensor2lidar or sensor2ego
                # since ego=lidar here, we use extrinsic_c2l
                # Use scipy for robust rotation to quaternion conversion
                # scipy returns [x, y, z, w], NuScenes expects [w, x, y, z]
                q = R.from_matrix(extrinsic_c2l[:3, :3]).as_quat()
                q_nusc = [q[3], q[0], q[1], q[2]]
                
                cam_info = {
                    'data_path': os.path.join(root_path, f'training/{cam_dir}/{frame_id}.jpg'),
                    'sensor2lidar_rotation': extrinsic_c2l[:3, :3],
                    'sensor2lidar_translation': extrinsic_c2l[:3, 3],
                    'cam_intrinsic': P[:3, :3],
                    # NuScenes format also needs ego2global etc. in cam_info sometimes
                    'ego2global_translation': [0,0,0],
                    'ego2global_rotation': [1,0,0,0],
                    'sensor2ego_translation': extrinsic_c2l[:3, 3],
                    'sensor2ego_rotation': q_nusc
                }
                # sensor2lidar used in pipeline
                cam_info['sensor2lidar_rotation'] = cam_info['sensor2lidar_rotation'] # as matrix
                
                info['cams'][cam_name] = cam_info
            
            # Process Radar
            radar_info = {
                'data_path': os.path.join(root_path, f'training/radar/{frame_id}.bin'),
                'sensor2lidar_rotation': Tr_r2v[:3, :3],
                'sensor2lidar_translation': Tr_r2v[:3, 3],
                'timestamp': int(frame_id),
            }
            # Only use one radar for V2X-Radar-I
            info['radars'] = {
                'RADAR_FRONT': [radar_info],
            }
            
            # Process Labels
            label_path = os.path.join(root_path, 'training', 'label_2', f'{frame_id}.txt')
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    label_lines = f.readlines()
                
                gt_boxes = []
                gt_names = []
                gt_velocity = [] # (N, 2)
                
                for line in label_lines:
                    parts = line.split()
                    cat = parts[0]
                    if cat not in cat_map:
                        continue
                        
                    # KITTI: [h, w, l, x, y, z, rot_y]
                    # locations are in camera frame usually, but V2X-Radar-I labels 
                    # might be different. Let's refer to BEVHeight's load_annotations.
                    # BEVHeight maps loc_cam to loc_lidar using Tr_cam2velo.
                    
                    # Wait, looking at BEVHeight:
                    # 491: loc_lidar = np.matmul(Tr_cam2velo, loc_cam).squeeze(-1)[:3]
                    # 492: loc_lidar[2] += 0.5 * float(row['dh'])
                    
                    # I should do the same.
                    # Extrinsic Lidar to Cam (Front cam)
                    extrinsic_l2c_front = np.eye(4)
                    extrinsic_l2c_front[:3, :3] = R0_rect @ Tr_v2c2[:3, :3]
                    extrinsic_l2c_front[:3, 3] = R0_rect @ Tr_v2c2[:3, 3]
                    tr_c2l = np.linalg.inv(extrinsic_l2c_front)
                    
                    h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
                    x_cam, y_cam, z_cam = float(parts[11]), float(parts[12]), float(parts[13])
                    rot_y = float(parts[14])
                    
                    loc_cam = np.array([x_cam, y_cam, z_cam, 1.0])
                    loc_lidar = (tr_c2l @ loc_cam)[:3]
                    loc_lidar[2] += 0.5 * h # BEVHeight center adjustment
                    
                    # rot_y in KITTI is around camera y-axis.
                    # We need yaw in Lidar frame.
                    # BEVHeight: rot_y_lidar = -0.5 * np.pi - rot_y (approx)
                    # Actually, a better way:
                    # Yaw in lidar is rot_y + offset? 
                    # Let's use the BEVHeight formula which matches their visualization.
                    yaw = -0.5 * np.pi - rot_y
                    
                    # NuScenes box format for this model: [x, y, z, dx, dy, dz, yaw, vx, vy]
                    # We have [x, y, z, l, w, h, yaw]
                    # and dummy [vx, vy]
                    gt_box = [loc_lidar[0], loc_lidar[1], loc_lidar[2], l, w, h, yaw, 0.0, 0.0]
                    gt_boxes.append(gt_box)
                    gt_names.append(cat_map[cat])
                
                gt_labels = []
                for name in gt_names:
                    if name in categories:
                        gt_labels.append(categories.index(name))
                    else:
                        gt_labels.append(-1)
                
                info['gt_boxes'] = np.array(gt_boxes, dtype=np.float32)
                info['gt_names'] = np.array(gt_names)
                info['gt_velocity'] = np.zeros((len(gt_boxes), 2), dtype=np.float32)
                info['num_lidar_pts'] = np.ones(len(gt_boxes), dtype=int)
                # KITTI-style annos for evaluation
                kitti_annos = {
                    'name': np.array(gt_names),
                    'truncated': np.zeros(len(gt_names)),
                    'occluded': np.zeros(len(gt_names), dtype=int),
                    'alpha': np.zeros(len(gt_names)), # Dummy alpha
                    'bbox': np.zeros((len(gt_names), 4)), # Dummy 2D bbox
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                }
                
                for i in range(len(gt_boxes)):
                    box = gt_boxes[i]
                    # KITTI: dim is [h, w, l], loc is [x, y, z] in camera or [x, y, z] in lidar
                    # The Evaluation class expects camera coords or just uses them for iou.
                    # If we evaluate in lidar, we need to be careful.
                    # But often it's h, w, l.
                    # Let's use the format expected by VOD evaluation.
                    kitti_annos['dimensions'].append([box[5], box[4], box[3]]) # h, w, l
                    kitti_annos['location'].append([box[0], box[1], box[2]])
                    kitti_annos['rotation_y'].append(box[6]) # yaw
                
                kitti_annos['dimensions'] = np.array(kitti_annos['dimensions'])
                kitti_annos['location'] = np.array(kitti_annos['location'])
                kitti_annos['rotation_y'] = np.array(kitti_annos['rotation_y'])
                
                info['annos'] = kitti_annos
                
                # Pre-format ann_infos [N, 9] boxes, [N] labels
                info['ann_infos'] = (info['gt_boxes'], np.array(gt_labels, dtype=np.int64))

            infos.append(info)
        
        # Save as pkl
        data = {'infos': infos, 'metadata': {'version': 'v1.0-trainval'}}
        out_file = os.path.join(out_path, f'v2x-radar_infos_{split}.pkl')
        with open(out_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {out_file}")

if __name__ == "__main__":
    convert_v2x_radar_to_nuscenes('/mnt/datasets/V2X-Radar-I/V2X-Radar-I/', '/home/tamoghno/RCBEVDet/rcbevdet-master/data/v2x-radar/')
