from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

import mmcv

from mmdet3d.models.builder import HEADS

__all__ = ["BEVSegHead"]


def sigmoid_xent_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    return F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


class BEVGridTransform(nn.Module):
    def __init__(
        self,
        *,
        input_scope: List[Tuple[float, float, float]],
        output_scope: List[Tuple[float, float, float]],
        prescale_factor: float = 1,
    ) -> None:
        super().__init__()
        self.input_scope = input_scope
        self.output_scope = output_scope
        self.prescale_factor = prescale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prescale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.prescale_factor,
                mode="bilinear",
                align_corners=False,
            )

        coords = []
        for (imin, imax, _), (omin, omax, ostep) in zip(
            self.input_scope, self.output_scope
        ):
            v = torch.arange(omin + ostep / 2, omax, ostep)
            v = (v - imin) / (imax - imin) * 2 - 1
            coords.append(v.to(x.device))

        u, v = torch.meshgrid(coords, indexing="ij")
        grid = torch.stack([v, u], dim=-1)
        grid = torch.stack([grid] * x.shape[0], dim=0)

        x = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            align_corners=False,
        )
        return x


@HEADS.register_module()
class BEVSegHead(nn.Module):
    def __init__(
            self,
            conv_config=[[256, 256, 3], [256, 256, 3]],
            classes=['vehicle'],
            loss='focal',
            loss_weight=1.,
            grid_transform=None,
            multi_head=False,
            norm_type='BatchNorm',
    ) -> None:
        super().__init__()
        self.conv_config = conv_config
        self.classes = classes
        self.loss = loss
        self.loss_weight = loss_weight
        self.grid_transform = grid_transform
        self.multi_head = multi_head

        assert type(loss_weight) == int or type(loss_weight) == float \
               or type(loss_weight) == mmcv.utils.config.ConfigDict

        if self.grid_transform is not None:
            self.transform = BEVGridTransform(**grid_transform)

        if self.multi_head == False:
            classseq = nn.Sequential()
            for i, (in_c, out_c, k_size) in enumerate(self.conv_config):
                classseq.add_module(
                    'Conv' + str(i),
                    nn.Conv2d(in_c, out_c, k_size, padding=(k_size-1)//2, bias=False),
                )
                if norm_type == 'BatchNorm':
                    classseq.add_module(
                        'BN' + str(i),
                        nn.BatchNorm2d(out_c),
                    )
                elif norm_type == 'InstanceNorm':
                    classseq.add_module(
                        'IN' + str(i),
                        nn.InstanceNorm2d(out_c),
                    )
                else:
                    raise TypeError("not support norm_type:", norm_type,
                                    ", if you want to use syncBatchNorm, add 'syncbn=True' in config.")
                classseq.add_module(
                    "ReLU" + str(i),
                    nn.ReLU(True),
                )
            classseq.add_module(
                'Convlast',
                nn.Conv2d(self.conv_config[-1][1], len(classes), 1),
            )
            self.classifier = classseq
        else:
            self.classifiers = nn.ModuleList()
            for _ in classes:
                classseq = nn.Sequential()
                for i, (in_c, out_c, k_size) in enumerate(self.conv_config):
                    classseq.add_module(
                        'Conv' + str(i),
                        nn.Conv2d(in_c, out_c, k_size, padding=(k_size-1)//2, bias=False),
                    )
                    if norm_type == 'BatchNorm':
                        classseq.add_module(
                            'BN' + str(i),
                            nn.BatchNorm2d(out_c),
                        )
                    elif norm_type == 'InstanceNorm':
                        classseq.add_module(
                            'IN' + str(i),
                            nn.InstanceNorm2d(out_c),
                        )
                    else:
                        raise TypeError("not support norm_type:", norm_type,
                                        ", if you want to use syncBatchNorm, add 'syncbn=True' in config.")
                    classseq.add_module(
                        "ReLU" + str(i),
                        nn.ReLU(True),
                    )
                classseq.add_module(
                    'Convlast',
                    nn.Conv2d(self.conv_config[-1][1], 1, 1),
                )
                self.classifiers.append(classseq)

    def forward(
            self,
            x: torch.Tensor,
            target: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:

        if isinstance(x, (list, tuple)):
            x = x[0]

        if isinstance(target, (list, tuple)):
            target = torch.stack(target, dim=0)

        if self.grid_transform is not None:
            x = self.transform(x)

        # print("before", x.shape)
        if self.multi_head == False:
            x = self.classifier(x)
        else:
            out_list = []
            for classifier in self.classifiers:
                out = classifier(x)
                out_list.append(out)
            x = torch.cat(out_list, dim=1)
        # print("after", x.shape)

        # DEBUG for visibility
        # target = target[0]
        # if not self.training:
        #     for i in range(x.shape[0]):
        #         self.vis(x[i], target[i], 1)

        if self.training:
            losses = {}
            for index, name in enumerate(self.classes):
                if self.loss == "xent":
                    if type(self.loss_weight) == mmcv.utils.config.ConfigDict:
                        loss = sigmoid_xent_loss(x[:, index], target[:, index]) * self.loss_weight[name]
                    else:
                        loss = sigmoid_xent_loss(x[:, index], target[:, index]) * self.loss_weight
                elif self.loss == "focal":
                    if type(self.loss_weight) == mmcv.utils.config.ConfigDict:
                        loss = sigmoid_focal_loss(x[:, index], target[:, index]) * self.loss_weight[name]
                    else:
                        loss = sigmoid_focal_loss(x[:, index], target[:, index]) * self.loss_weight
                else:
                    raise ValueError(f"unsupported loss: {self.loss}")
                losses[f"{name}_{self.loss}_loss"] = loss
            return losses
        else:
            return torch.sigmoid(x)

    def vis(self, x, t, type):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import random
        from datetime import datetime

        randstr = datetime.now().strftime('%H:%M:%S')

        print('\n******** BEGIN PRINT GT**********\n')
        print("BEV:", t.shape)
        fig = plt.figure(figsize=(16, 16))
        plt.plot([50, 50, -50, -50, 50], [50, -50, -50, 50, 50], lw=0.5)
        plt.plot([65, 65, -65, -65, 65], [65, -65, -65, 65, 65], lw=0.5)
        for xx in range(200):
            for yy in range(200):
                xc = -50 + xx * 0.5
                yc = -50 + yy * 0.5
                # 0 vehicle, 1 可行驶区域, 2 车道线
                if type == 0:
                    if t[0, xx, yy] == 1:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="blue"))
                elif type == 1:
                    if t[0, xx, yy] == 1:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="green"))
                elif type == 2:
                    if t[0, xx, yy] == 1:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="red"))
                else:
                    raise ValueError
        if type == 0:
            plt.savefig("/home/wangxinhao/vis/seg_v/gt/gt" + randstr + ".png")
        elif type == 1:
            plt.savefig("/home/wangxinhao/vis/seg_a/gt/gt" + randstr + ".png")
        elif type == 2:
            plt.savefig("/home/wangxinhao/vis/seg_d/gt/gt" + randstr + ".png")
        else:
            raise TypeError
        print('\n******** END PRINT GT**********\n')

        print('\n******** BEGIN PRINT PRED**********\n')
        print("BEV:", x.shape)
        fig = plt.figure(figsize=(16, 16))
        plt.plot([50, 50, -50, -50, 50], [50, -50, -50, 50, 50], lw=0.5)
        plt.plot([65, 65, -65, -65, 65], [65, -65, -65, 65, 65], lw=0.5)
        for xx in range(200):
            for yy in range(200):
                xc = -50 + xx * 0.5
                yc = -50 + yy * 0.5
                # 0 vehicle, 1 可行驶区域, 2 车道线
                if type == 0:
                    if x[0, xx, yy] > 0.45:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="blue"))
                elif type == 1:
                    if x[0, xx, yy] > 0.45:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="green"))
                elif type == 2:
                    if x[0, xx, yy] > 0.40:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="red"))
                else:
                    raise TypeError
        if type == 0:
            plt.savefig("/home/wangxinhao/vis/seg_v/pred/pred" + randstr + ".png")
        elif type == 1:
            plt.savefig("/home/wangxinhao/vis/seg_a/pred/pred" + randstr + ".png")
        elif type == 2:
            plt.savefig("/home/wangxinhao/vis/seg_d/pred/pred" + randstr + ".png")
        else:
            raise TypeError

        print('\n******** END PRINT PRED**********\n')
