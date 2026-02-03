# RCBEVDet Execution Guide

This document provides stepwise instructions for running the RCBEVDet model on the **nuScenes** and **V2X-Radar-I** datasets.

---

## 1. Running on V2X-Radar-I (KITTI Evaluation)

The V2X-Radar-I dataset is originally in KITTI format. We convert it to a nuScenes-compatible structure and use a customized KITTI evaluation protocol.

### Step 1: Data Preparation
Convert the V2X-Radar dataset into the format expected by the model.
```bash
python tools/data_converter/v2x_radar_converter.py
```
*Outputs: `data/v2x-radar/v2x-radar_infos_train.pkl` and `v2x-radar_infos_val.pkl`.*

### Step 2: Training
Train the model using the custom configuration (Front Camera + Radar fusion).
```bash
python tools/train.py configs/rcbevdet/rcbevdet-v2x-radar.py
```

### Step 3: Evaluation
Run evaluation using the KITTI protocol (measures 3D AP/BEV AP).
```bash
python tools/test.py configs/rcbevdet/rcbevdet-v2x-radar.py \
    work_dirs/rcbevdet-v2x-radar/latest.pth \
    --eval kitti
```

### Step 4: Visualization
Generate top-down BEV visualizations of the detections.
```bash
python tools/test.py configs/rcbevdet/rcbevdet-v2x-radar.py \
    work_dirs/rcbevdet-v2x-radar/latest.pth \
    --show --show-dir results/v2x_radar_vis
```

---

## 2. Running on nuScenes (Official Metrics)

The standard nuScenes pipeline uses the official nuScenes devkit for evaluation (mAP, NDS).

### Step 1: Data Preparation
Generate the pickle files for nuScenes. Ensure your dataset is at `./data/nuscenes`.
```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes --extra-tag nuscenes
```
*(Note: If using RCBEVDet's specific preprocessing, refer to scripts like `tools/create_data_nuscenes_RC.py`)*

### Step 2: Training
Train using the standard RCBEVDet configuration (6 Camera + Radar fusion).
```bash
python tools/train.py configs/rcbevdet/rcbevdet-256x704-r50-BEV128-9kf-depth-cbgs12e-circlelarger.py
```

### Step 3: Evaluation
Run the official nuScenes evaluation.
```bash
python tools/test.py configs/rcbevdet/rcbevdet-256x704-r50-BEV128-9kf-depth-cbgs12e-circlelarger.py \
    checkpoints/rcbevdet_nuscenes.pth \
    --eval bbox
```

---

## Technical Notes & Fixes
- **Evaluation**: The `kitti` metric for V2X-Radar was manually integrated into `NuScenesDatasetRC` to bypass the requirement for official nuScenes JSON database tables.
- **V2X Configuration**: The `rcbevdet-v2x-radar.py` config is modified to use only **1 Camera** (`CAM_FRONT`) and **5 Radar features**, matching the V2X-Radar-I specification.
- **Compatibility**: If you encounter a `TypeError` regarding `FormatCode` or an `AttributeError` for `type` in PyTorch 2.0+, the current versions of `tools/train.py` and `tools/test.py` in this repository have been fixed with monkey-patches.
