# Copyright (c) OpenMMLab. All rights reserved.
from .dgcnn_head import DGCNNHead
from .paconv_head import PAConvHead
from .pointnet2_head import PointNet2Head
from .futr3d_head import FUTR3DHead

__all__ = ['PointNet2Head', 'DGCNNHead', 'PAConvHead', 'FUTR3DHead']
