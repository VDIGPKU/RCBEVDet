import mmcv
from mmdet3d.datasets import build_dataset
from mmcv import Config
from mmcv.parallel import DataContainer

cfg = Config.fromfile('configs/rcbevdet/rcbevdet-v2x-radar.py')
dataset = build_dataset(cfg.data.train)

print(f"Dataset length: {len(dataset)}")
print("Loading first item...")
data = dataset[0]
print("Keys in data:", data.keys())

if 'img_inputs' in data:
    img_inputs = data['img_inputs']
    if isinstance(img_inputs, DataContainer):
        img_inputs = img_inputs.data
    print("Image inputs type:", type(img_inputs))
    if isinstance(img_inputs, (list, tuple)):
        print("Number of multi-view items:", len(img_inputs))
        print("First item shape:", img_inputs[0].shape)

if 'radar' in data:
    radar_data = data['radar']
    if isinstance(radar_data, DataContainer):
        radar_data = radar_data.data
    
    print("Radar data type:", type(radar_data))
    if hasattr(radar_data, 'tensor'):
        print("Radar points shape:", radar_data.tensor.shape)
        print("Radar points features (first point):", radar_data.tensor[0])
    else:
        print("Radar points shape (fallback):", getattr(radar_data, 'shape', 'N/A'))

if 'gt_bboxes_3d' in data:
    gt_boxes = data['gt_bboxes_3d']
    if isinstance(gt_boxes, DataContainer):
        gt_boxes = gt_boxes.data
    print("GT Boxes type:", type(gt_boxes))
    print("GT Boxes shape:", gt_boxes.tensor.shape if hasattr(gt_boxes, 'tensor') else "N/A")
    if hasattr(gt_boxes, 'tensor'):
        print("First GT box:", gt_boxes.tensor[0])
