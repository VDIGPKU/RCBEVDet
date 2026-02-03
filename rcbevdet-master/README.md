# BEV Perception

PKU VDIG BEV-Perception Team

## Getting Started

### environment

The code is tested in the following two environment:

```
python                       3.8.13
cuda                         12.1
pytorch                      2.0.1+cu118
torchvision                  0.15.2+cu118
numpy                        1.23.4
mmcv-full                    1.6.0
mmcls                        0.25.0
mmdet                        2.28.2
nuscenes-devkit              1.1.11
av2                          0.2.1
detectron2                   0.6
(for A800 or A40 + cuda 12.1)
```

```
python                       3.8.13
cuda                         11.6
pytorch                      1.12.1+cu116
torchvision                  0.13.0+cu116
numpy                        1.19.5
mmcv-full                    1.6.2
mmcls                        1.0.0rc1
mmdet                        2.24.0
nuscenes-devkit              1.1.9
detectron2                   0.6
(for other GPUs + cuda 11.6)
```

If you encounter slow download speed or timeout when downloading dependency packages, 
you need to consider installing the dependency packages from the mirror website first, 
and then execute the installation:

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install {Find the dependencies in setup.py:setup(install_requires=[...]) and write them down here} -i https://pypi.tuna.tsinghua.edu.cn/simple
python setup.py develop
```

The most recommended installation steps are:

1. Create a Python environment. Install [PyTorch](https://pytorch.org/get-started/previous-versions/)
corresponding to your machine's CUDA version;

2. Install [mmcv](https://github.com/open-mmlab/mmcv) corresponding to your PyTorch and CUDA version;

3. Install other dependencies of mmdet and install [mmdet](https://github.com/open-mmlab/mmdetection);

4. Install other dependencies of this project (Please change the spconv version
in the requirements.txt to the CUDA version you are using) and setup this project;

5. Compile some operators manually.
```bash
cd mmdet3d/ops/csrc
python setup.py build_ext --inplace
cd ../deformattn
python setup.py build install
```

6. Install other dependencies of detectron2 and install [detectron2](https://github.com/facebookresearch/detectron2);


### data preparetion
If your folder structure is different from the following, you may need to change the 
corresponding paths in config files.

```
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   │   ├── basemap
│   │   │   ├── expansion
│   │   │   ├── prediction
│   │   │   ├── *.png
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
```

For RCBEVDet, prepare nuscenes data by running
```bash
python tools/create_data_nuscenes_RC.py
```


### training

```bash
./tools/dist_train.sh $config_path $gpus
```

### testing

testing on validation set

```bash
./tools/dist_test.sh $config_path $checkpoint_path $gpus --eval bbox
```

testing on test set

```bash
./tools/dist_test.sh $config_path $checkpoint_path $gpus --format-only --eval-options 'jsonfile_prefix=work_dirs'
mv work_dirs/pts_bbox/results_nusc.json work_dirs/pts_bbox/{$name}.json
```

benchmarking test latency

```bash
python tools/analysis_tools/benchmark_sequential.py $config 1 --fuse-conv-bn
```

TTA or Ensemble

```bash
python tools/merge_result_json.py --paths a.json,b.json,c.json
mv work_dirs/ens/results.json work_dirs/ens/{$name}.json
```

If you have any other questions, please refer to 
<a href='https://mmdetection3d.readthedocs.io/en/v1.0.0rc1/'>mmdet3d docs</a>.

