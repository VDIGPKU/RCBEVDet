import torch
import numpy as np
import math
import torch.nn as nn
from torch.nn import functional as F
from mmdet3d.models.fusion_layers import apply_3d_transformation
from mmdet3d.ops import localattention
from .ip_basic import depth_map_utils
import pdb

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, bias='auto',
                 inplace=True, affine=True):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.use_norm = norm_layer is not None
        self.use_activation = activation_layer is not None
        if bias == 'auto':
            bias = not self.use_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.use_norm:
            self.bn = norm_layer(out_channels, affine=affine)
        if self.use_activation:
            self.activation = activation_layer(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.bn(x)
        if self.use_activation:
            x = self.activation(x)
        return x
    
class similarFunction(torch.autograd.Function):
    """ credit: https://github.com/zzd1992/Image-Local-Attention """

    @staticmethod
    def forward(ctx, x_ori, x_loc, kH, kW):
        ctx.save_for_backward(x_ori, x_loc)
        ctx.kHW = (kH, kW)
        output = localattention.similar_forward(
            x_ori, x_loc, kH, kW)

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        x_ori, x_loc = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = localattention.similar_backward(
            x_loc, grad_outputs, kH, kW, True)
        grad_loc = localattention.similar_backward(
            x_ori, grad_outputs, kH, kW, False)

        return grad_ori, grad_loc, None, None


class weightingFunction(torch.autograd.Function):
    """ credit: https://github.com/zzd1992/Image-Local-Attention """

    @staticmethod
    def forward(ctx, x_ori, x_weight, kH, kW):
        ctx.save_for_backward(x_ori, x_weight)
        ctx.kHW = (kH, kW)
        output = localattention.weighting_forward(
            x_ori, x_weight, kH, kW)

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        x_ori, x_weight = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = localattention.weighting_backward_ori(
            x_weight, grad_outputs, kH, kW)
        grad_weight = localattention.weighting_backward_weight(
            x_ori, grad_outputs, kH, kW)

        return grad_ori, grad_weight, None, None
    

class LocalContextAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, last_affine=True):
        super().__init__()

        self.f_similar = similarFunction.apply
        self.f_weighting = weightingFunction.apply

        self.kernel_size = kernel_size
        self.query_project = nn.Sequential(ConvBNReLU(in_channels,
                                                                  out_channels,
                                                                  kernel_size=1,
                                                                  norm_layer=nn.BatchNorm2d,
                                                                  activation_layer=nn.ReLU),
                                           ConvBNReLU(out_channels,
                                                                  out_channels,
                                                                  kernel_size=1,
                                                                  norm_layer=nn.BatchNorm2d,
                                                                  activation_layer=nn.ReLU))
        self.key_project = nn.Sequential(ConvBNReLU(in_channels,
                                                                out_channels,
                                                                kernel_size=1,
                                                                norm_layer=nn.BatchNorm2d,
                                                                activation_layer=nn.ReLU),
                                         ConvBNReLU(out_channels,
                                                                out_channels,
                                                                kernel_size=1,
                                                                norm_layer=nn.BatchNorm2d,
                                                                activation_layer=nn.ReLU))
        self.value_project = ConvBNReLU(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    norm_layer=nn.BatchNorm2d,
                                                    activation_layer=nn.ReLU,
                                                    affine=last_affine)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, target_feats, source_feats, **kwargs):
        query = self.query_project(target_feats)
        key = self.key_project(source_feats)
        value = self.value_project(source_feats)

        weight = self.f_similar(query, key, self.kernel_size, self.kernel_size)
        weight = nn.functional.softmax(weight / math.sqrt(key.size(1)), -1)
        out = self.f_weighting(value, weight, self.kernel_size, self.kernel_size)
        return out

class BEVWarp(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    # 可以with no grad
    @torch.no_grad()
    def forward(self, lidar_feats, img_feats, img_metas, pts_metas, **kwargs):
        batch_size, num_views, I_C, I_H, I_W = img_feats.shape
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = img_feats.new_tensor(lidar2img)
        img2lidar = torch.inverse(lidar2img)
        pts = pts_metas['pts']
        decorated_img_feats = []
        for b in range(batch_size):
            img_feat = img_feats[b]
            ori_H, ori_W = img_metas[b]['input_shape']
            pts_3d = pts[b][...,:3]
            pts_3d = apply_3d_transformation(pts_3d, 'LIDAR', img_metas[b], reverse=True).detach()
            pts_4d = torch.cat((pts_3d,torch.ones_like(pts_3d[...,:1])),dim=-1).unsqueeze(0).unsqueeze(-1)
            proj_mat = lidar2img[b].unsqueeze(1)
            pts_2d = torch.matmul(proj_mat, pts_4d).squeeze(-1)
            eps = 1e-5
            depth = pts_2d[..., 2:3]
            mask = (pts_2d[..., 2:3] > eps)
            pts_2d = pts_2d[..., 0:2] / torch.maximum(
                pts_2d[..., 2:3], torch.ones_like(pts_2d[..., 2:3])*eps)
            proj_x = (pts_2d[...,0:1] / ori_W - 0.5) * 2
            proj_y = (pts_2d[...,1:2] / ori_H - 0.5) * 2
            mask = (mask & (proj_x > -1.0) 
                             & (proj_x < 1.0) 
                             & (proj_y > -1.0) 
                             & (proj_y < 1.0))
            mask = torch.nan_to_num(mask)
            depth_map = img_feat.new_zeros(num_views, I_H, I_W)
            for i in range(num_views):
                depth_map[i, (pts_2d[i,mask[i,:,0],1]/ori_H*I_H).long(), (pts_2d[i,mask[i,:,0],0]/ori_W*I_W).long()] = depth[i,mask[i,:,0],0]
            fill_type = 'multiscale'
            extrapolate = False
            blur_type = 'bilateral'
            for i in range(num_views):
                final_depths, _ = depth_map_utils.fill_in_multiscale(
                                depth_map[i].detach().cpu().numpy(), extrapolate=extrapolate, blur_type=blur_type,
                                show_process=False)
                depth_map[i] = depth_map.new_tensor(final_depths)
            xs = torch.linspace(0, ori_W - 1, I_W, dtype=torch.float32).to(depth_map.device).view(1, 1, I_W).expand(num_views, I_H, I_W)
            ys = torch.linspace(0, ori_H - 1, I_H, dtype=torch.float32).to(depth_map.device).view(1, I_H, 1).expand(num_views, I_H, I_W)
            xyd = torch.stack((xs, ys, depth_map, torch.ones_like(depth_map)), dim = -1)
            xyd [..., 0] *= xyd [..., 2]
            xyd [..., 1] *= xyd [..., 2]
            xyz = img2lidar[b].view(num_views,1,1,4,4).matmul(xyd.unsqueeze(-1)).squeeze(-1)[...,:3] #(6,112,200,3)
            xyz = apply_3d_transformation(xyz.view(num_views*I_H*I_W, 3), 'LIDAR', img_metas[b], reverse=False).view(num_views, I_H, I_W, 3).detach()
            pc_range = xyz.new_tensor([-54, -54, -5, 54, 54, 3])  #TODO: fix it to support other outdoor dataset!!!
            lift_mask = (xyz[...,0] > pc_range[0]) & (xyz[...,1] > pc_range[1]) & (xyz[...,2] > pc_range[2])\
                        & (xyz[...,0] < pc_range[3]) & (xyz[...,1] < pc_range[4]) & (xyz[...,2] < pc_range[5])
            xy_bev = (xyz[...,0:2] - pc_range[0:2]) / (pc_range[3:5] - pc_range[0:2])
            xy_bev = (xy_bev - 0.5) * 2
            decorated_img_feat = F.grid_sample(lidar_feats[b].unsqueeze(0).repeat(num_views,1,1,1), xy_bev, align_corners=False).permute(0,2,3,1) #N, H, W, C
            decorated_img_feat[~lift_mask]=0
            decorated_img_feats.append(decorated_img_feat.permute(0,3,1,2))
        decorated_img_feats = torch.stack(decorated_img_feats, dim=0)
        return decorated_img_feats


class MMRI_P2I(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, last_affine=True):
        super().__init__()
        self.Warp = BEVWarp()
        self.Local = LocalContextAttentionBlock(in_channels, out_channels, kernel_size, last_affine=True)
    
    def forward(self, lidar_feats, img_feats, img_metas, pts_metas, **kwargs):
        warped_img_feats = self.Warp(lidar_feats, img_feats, img_metas, pts_metas) #B, N, C, H, W
        B, N, C, H, W = warped_img_feats.shape
        decorated_img_feats = self.Local(img_feats.view(B*N,C,H,W), warped_img_feats.view(B*N,C,H,W)).view(B,N,C,H,W)
        return decorated_img_feats
    

class MMRI_I2P(nn.Module):
    
    def __init__(self, pts_channels, img_channels, dropout):
        super().__init__()
        self.pts_channels = pts_channels
        self.img_channels = img_channels
        self.dropout = dropout
        self.learnedAlign = nn.MultiheadAttention(pts_channels, 1, dropout=dropout, 
                                                  kdim=img_channels, vdim=img_channels,
                                                  batch_first=True)

    def forward(self, lidar_feat, img_feat, img_metas, pts_metas, **kwargs):
        batch_size = len(img_metas)
        decorated_lidar_feat = torch.zeros_like(lidar_feat)
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = lidar_feat.new_tensor(lidar2img) #(B,6,4,4)
        batch_cnt = lidar_feat.new_zeros(batch_size).int()
        for b in range(batch_size):
            batch_cnt[b] = (pts_metas['pillar_coors'][:,0] == b).sum()
        batch_bound = batch_cnt.cumsum(dim=0)
        cur_start = 0
        for b in range(batch_size):
            cur_end = batch_bound[b]
            voxel = pts_metas['pillars'][cur_start:cur_end]
            voxel_coor = pts_metas['pillar_coors'][cur_start:cur_end]
            pillars_num_points = pts_metas['pillars_num_points'][cur_start:cur_end]
            proj_mat = lidar2img[b]
            num_cam = proj_mat.shape[0]
            num_voxels, max_points, p_dim = voxel.shape
            num_pts = num_voxels * max_points
            pts = voxel.view(num_pts, p_dim)[...,:3]
            voxel_pts = apply_3d_transformation(pts, 'LIDAR', img_metas[b], reverse=True).detach()
            voxel_pts = torch.cat((voxel_pts,torch.ones_like(voxel_pts[...,:1])),dim=-1).unsqueeze(0).unsqueeze(-1)
            proj_mat = proj_mat.unsqueeze(1)
            xyz_cams = torch.matmul(proj_mat, voxel_pts).squeeze(-1)
            eps = 1e-5
            mask = (xyz_cams[..., 2:3] > eps)
            xy_cams = xyz_cams[..., 0:2] / torch.maximum(
                xyz_cams[..., 2:3], torch.ones_like(xyz_cams[..., 2:3])*eps)
            img_shape = img_metas[b]['input_shape']
            xy_cams[...,0] = xy_cams[...,0] / img_shape[1]
            xy_cams[...,1] = xy_cams[...,1] / img_shape[0]
            xy_cams = (xy_cams - 0.5) * 2
            mask = (mask & (xy_cams[..., 0:1] > -1.0) 
                 & (xy_cams[..., 0:1] < 1.0) 
                 & (xy_cams[..., 1:2] > -1.0) 
                 & (xy_cams[..., 1:2] < 1.0))
            mask = torch.nan_to_num(mask)
            sampled_feat = F.grid_sample(img_feat[b],xy_cams.unsqueeze(-2)).squeeze(-1).permute(2,0,1)
            sampled_feat = sampled_feat.view(num_voxels,max_points,num_cam,self.img_channels)
            mask = mask.permute(1,0,2).view(num_voxels,max_points,num_cam,1)

            # for i in range(num_voxels):
            #     mask[i,pillars_num_points[i]:] = False
            mask_points = mask.new_zeros((mask.shape[0],mask.shape[1]+1))
            mask_points[torch.arange(mask.shape[0],device=mask_points.device).long(),pillars_num_points.long()] = 1
            mask_points = mask_points.cumsum(dim=1).bool()
            mask_points = ~mask_points
            mask = mask_points[:,:-1].unsqueeze(-1).unsqueeze(-1) & mask

            mask = mask.reshape(num_voxels,max_points*num_cam,1)
            sampled_feat = sampled_feat.reshape(num_voxels,max_points*num_cam,self.img_channels)
            K = sampled_feat
            V = sampled_feat
            Q = lidar_feat[b,:,voxel_coor[:,2].long(),voxel_coor[:,3].long()].t().unsqueeze(1)
            valid = mask[...,0].sum(dim=1) > 0 
            attn_output = lidar_feat.new_zeros(num_voxels, 1, self.pts_channels)
            attn_output[valid] = self.learnedAlign(Q[valid],K[valid],V[valid],attn_mask=(~mask[valid]).permute(0,2,1))[0]
            decorated_lidar_feat[b,:,voxel_coor[:,2].long(),voxel_coor[:,3].long()] = attn_output.squeeze(1).t()
            cur_start = cur_end
        return decorated_lidar_feat
