import json
import os
import pickle
import refile
import ipdb
import numpy as np

def ref():
    # Define the directory paths for the input and output data
    input_dir = 'input_data'
    output_dir = 'output_data/coco'

    # Define the category IDs and names for your dataset
    category_ids = {
        'cat': 1,
        'dog': 2,
        'bird': 3
    }

    # Create the categories list
    categories = [{'id': v, 'name': k} for k, v in category_ids.items()]

    # Initialize dictionaries for storing information about images and annotations
    images = []
    annotations = []
    image_id = 1
    annotation_id = 1

    # Loop over all the images in the input directory
    for filename in os.listdir(input_dir):
        # Get the image ID
        image_id_str = filename.split('.')[0]
        image_id = int(image_id_str)

        # Add information about the image to the images list
        image = {
            'id': image_id,
            'file_name': filename,
            'width': 640,
            'height': 480
        }
        images.append(image)

        # Open the annotation file for the image
        annotation_file = os.path.join(input_dir, f'{image_id_str}.txt')
        with open(annotation_file, 'r') as f:
            lines = f.readlines()

        # Loop over all the annotations for the image
        for line in lines:
            # Parse the annotation information
            line = line.strip().split()
            x, y, w, h = map(float, line[1:])
            category_id = category_ids[line[0]]

            # Add information about the annotation to the annotations list
            annotation = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': category_id,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0
            }
            annotations.append(annotation)
            annotation_id += 1

    # Create the COCO dictionary
    coco_dict = {
        'info': {
            'description': 'My dataset',
            'url': '',
            'version': '1.0',
            'year': 2022,
            'contributor': '',
            'date_created': '2022-01-01'
        },
        'licenses': [],
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    # Save the COCO dictionary as a JSON file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'instances_train.json')
    with open(output_file, 'w') as f:
        json.dump(coco_dict, f)

def convert_info_to_coco(data_infos, name2nori, images, annotations):
    image_id = 0
    annotation_id = 0

    # Loop over all the samples
    for info in data_infos:

        # annotations  data['infos'][0]['gt2d_infos']['gt_2dlabels']  ['gt2d_infos']['gt_2dbboxes']
        gt_2dbboxes = info['gt2d_infos']['gt_2dbboxes']  # 7 * (M, 4), format w0 h0 w1 h1
        gt_2dlabels = info['gt2d_infos']['gt_2dlabels']  # 7 * (M)
        gt_centers2d = info['gt2d_infos']['centers2d']  # 7 * (M, 2)

        # "id": 0, "token": "0008f55832e94c089ec2feba68bd4250", "file_name": "samples/CAM_FRONT/n003-2018-01-05-16-53-19+0800__CAM_FRONT__1515142478908480.jpg", "width": 1600, "height": 900, "nori_id":
        # data['infos'][0]['cam_infos']['ring_rear_left']
        for jth, subimg_dir in enumerate(['ring_rear_left', 'ring_side_left', 'ring_front_left', 'ring_front_center',
                                          'ring_front_right', 'ring_side_right', 'ring_rear_right']):
            # image: fpath
            fpath = str(info['cam_infos']['ring_rear_left']['fpath'])    # eg. 'val/**/**/**/fid.jpg'
            if subimg_dir != 'ring_front_center':
                width, height = 2048, 1550
            else:
                width, height = 1550, 2048

            # image: nori_id
            nori_id = name2nori.get(fpath, -1)
            if nori_id == -1:
                print('Not Found its nori: ', fpath)
                raise NotImplementedError

            # image: collect
            image = {
                'id': image_id,
                'file_name': fpath,
                'width': width,
                'height': height,
                'nori_id': nori_id,
            }
            images.append(image)

            # anno: box and label
            gt_2dbox = gt_2dbboxes[jth].tolist()
            gt_2dlabel = gt_2dlabels[jth].tolist()
            gt_2dcenter = gt_centers2d[jth].tolist()
            len_anno = len(gt_2dlabel)
            for kth in range(len_anno):
                ci, cj = gt_2dcenter[kth]   # ci, cj
                cw, ch = gt_2dbox[kth][2]-gt_2dbox[kth][0], gt_2dbox[kth][3]-gt_2dbox[kth][1] # w, h
                category_id = gt_2dlabel[kth]
                annotation = {
                    'id': annotation_id,
                    'image_id': int(image_id),
                    'category_id': category_id,
                    'bbox': [int(ci), int(cj), int(cw), int(ch)],
                    'area': int(cw) * int(ch),
                    'iscrowd': 0
                }
                annotations.append(annotation)
                annotation_id += 1

            # update image_id
            image_id += 1

if __name__ == '__main__':
    splits = ['train', 'val']
    split = splits[1]
    info_path = f'data/av2/av2_{split}_infos.pkl'
    name2nori_path = f's3://argoverse/nori/0410_camera/argoverse2_{split}_camera.pkl'
    save_path = f's3://argoverse/argo2d/json/argo2d_instances_{split}.json'

    images = []
    annotations = []
    with open(info_path, 'rb') as f:
        data = pickle.load(f)
    with refile.smart_open(name2nori_path, "rb") as f:
        name2nori = dict(pickle.load(f))
    convert_info_to_coco(data['infos'], name2nori, images, annotations)

    # class categories list, fg start from 1
    class_names = ['ARTICULATED_BUS', 'BICYCLE', 'BICYCLIST', 'BOLLARD', 'BOX_TRUCK', 'BUS',
                   'CONSTRUCTION_BARREL', 'CONSTRUCTION_CONE', 'DOG', 'LARGE_VEHICLE',
                   'MESSAGE_BOARD_TRAILER', 'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'MOTORCYCLE',
                   'MOTORCYCLIST', 'PEDESTRIAN', 'REGULAR_VEHICLE', 'SCHOOL_BUS', 'SIGN',
                   'STOP_SIGN', 'STROLLER', 'TRUCK', 'TRUCK_CAB', 'VEHICULAR_TRAILER',
                   'WHEELCHAIR', 'WHEELED_DEVICE', 'WHEELED_RIDER']
    categories = [{'id': v, 'name': k} for k, v in enumerate(class_names)]

    # Create the COCO dictionary
    coco_dict = {
        'info': {
            'description': 'Argoverse2 2D',
            'url': '',
            'version': '1.0',
            'year': 2023,
            'contributor': '',
            'date_created': '2023-04-24'
        },
        'licenses': [],
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }

    # Save the COCO dictionary as a JSON file
    with refile.smart_open(save_path, 'w') as f:
        json.dump(coco_dict, f) # , cls=NpEncoder