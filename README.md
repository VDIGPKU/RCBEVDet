# RCBEVDet

This is the official implementation of CVPR2024 paper: [**RCBEVDet: Radar-camera Fusion in Bird’s Eye View for 3D Object Detection**](https://arxiv.org/abs/2403.16440) and its extended version RCBEVDet++.



## Update

* 2024/06/28 - RCBEVDet++ achieves SOTA 3D object detection, BEV semantic segmentation, and 3D multi-object tracking results on nuScenes benchmark. **The paper and code for RCBEVDet++ is coming soon~**
* 2024/06/01 - Code for RCBEVDet is released.



## Weight & Code

* Model weights for RCBEVDet are released: [google drive](https://drive.google.com/drive/folders/1VhOBcJ7wT71R8Dqyr5MlQUKv7lVcjfrz?usp=sharing)
* Code: please sign the [application](https://github.com/VDIGPKU/RCBEVDet/blob/main/RCBEVDet%20Application.docx) to obtain the code



## Results

##### 3D Object Detection (nuScenes Validation)

| Method     | Input | Backbone  | NDS  | mAP  |
| :--------- | ----- | --------- | ---- | :--- |
| BEVDepth4D | C     | ResNet-50 | 51.9 | 40.5 |
| RCBEVDet   | C+R   | ResNet-50 | 56.8 | 45.3 |
| SparseBEV  | C     | ResNet-50 | 54.5 | 43.2 |
| RCBEVDet++ | C+R   | ResNet-50 | 60.4 | 51.9 |

##### 3D Object Detection (nuScenes Test)

| Method     | Input | Backbone | Future frame | NDS  | mAP  |
| :--------- | ----- | -------- | ------------ | ---- | :--- |
| BEVDepth4D | C     | V2-99    | No           | 60.5 | 51.5 |
| RCBEVDet   | C+R   | V2-99    | No           | 63.9 | 55.0 |
| SparseBEV  | C     | V2-99    | No           | 63.6 | 55.6 |
| RCBEVDet++ | C+R   | V2-99    | No           | 68.7 | 62.6 |
| SparseBEV  | C     | ViT-L    | Yes          | 70.2 | ——   |
| RCBEVDet++ | C+R   | ViT-L    | Yes          | 72.7 | 67.3 |

##### BEV Semantic Segmentation (nuScenes Validation)

| Method     | Input | Backbone   | mIoU |
| :--------- | ----- | ---------- | ---- |
| RCBEVDet++ | C+R   | ResNet-101 | 62.8 |

##### 3D Multi-object Tracking (nuScenes Test)

| Method     | Input | Backbone | AMOTA | AMOTP |
| :--------- | ----- | -------- | ----- | ----- |
| RCBEVDet++ | C+R   | ViT-L    | 59.6  | 0.713 |



## Acknowledgements

The overall code are based on [mmdetection3D](https://github.com/open-mmlab/mmdetection3d), [BEVDet](https://github.com/HuangJunJie2017/BEVDet) and [SparseBEV](https://github.com/MCG-NJU/SparseBEV/tree/main). We sincerely thank the authors for their great work.



## License

The project is only free for academic research purposes, but needs authorization for commerce. For commerce permission, please contact wyt@pku.edu.cn.
