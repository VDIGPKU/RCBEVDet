# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook
from mmdet3d.core.hook.utils import is_parallel
from functools import partial

__all__ = ['LrDecayControlHook']

def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.

    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)

@HOOKS.register_module()
class LrDecayControlHook(Hook):
    """ """

    def __init__(self, control_model='ViT'):
        super().__init__()
        self.control_model=control_model

    def set_lr_func(self, runner):
        if self.control_model == 'ViT':
            runner.optimizer.lr_factor_func=partial(get_vit_lr_decay_rate, lr_decay_rate=0.8, num_layers=24)
            runner.optimizer.overrides = {}
            runner.optimizer.weight_decay_norm = None
        else:
            assert False, TypeError

    def before_run(self, runner):
        if self.control_model:
            self.set_lr_func(runner)

    # def before_train_epoch(self, runner):
    #     if runner.epoch > self.lr_start_epoch:
    #         self.set_lr_func(runner)