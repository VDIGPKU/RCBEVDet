import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import ipdb
from math import cos, sin

def draw_feat(feat, name):
    plt.figure(figsize=(16, 16))

    # feat = [C,H,W]
    feat = feat.sum(dim=0).cpu().numpy()
    plt.imshow(feat)

    plt.savefig("./vis_results/" + name + ".png")

def print_gt_and_bev(pts_feats, gt, img_metas, name='vis'):
    print('\n******** BEGIN PRINT GT **********\n')
    assert(pts_feats.shape[0] == len(gt))
    for idx in range(len(gt)):
        # if img_metas[idx]['sample_idx'] != 'ca9a282c9e77460f8360f564131a8af5':
            # continue
        corner = gt[idx].corners

        fig = plt.figure(figsize=(16, 16))

        plt.plot([50,50,-50,-50,50], [50,-50,-50,50,50], lw=0.5)
        """
        Convert the boxes to corners in clockwise order, in form of
        ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``

        .. code-block:: none

                                            up z
                            front x           ^
                                    /            |
                                /             |
                    (x1, y0, z1) + -----------  + (x1, y1, z1)
                                /|            / |
                                / |           /  |
                (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                            |  /      .   |  /
                            | / oriign    | /
            left y<-------- + ----------- + (x0, y1, z0)
                (x0, y0, z0)
        """
        for i in range(corner.shape[0]):
            x1 = corner[i][0][0]
            y1 = corner[i][0][1]
            x2 = corner[i][2][0]
            y2 = corner[i][2][1]
            x3 = corner[i][6][0]
            y3 = corner[i][6][1]
            x4 = corner[i][4][0]
            y4 = corner[i][4][1]
            plt.plot([x1,x2,x3,x4,x1], [y1,y2,y3,y4,y1], lw=0.5)

        pts_feat = pts_feats[idx].cpu().detach().numpy() # 放到cpu上 去掉梯度 转换成numpy
        print('pts_feat.shape =', pts_feat.shape)
        feat_2d = np.zeros(pts_feat.shape[1:])
        print('feat_2d.shape =', feat_2d.shape)
        for h in range(feat_2d.shape[0]):
            for w in range(feat_2d.shape[1]):
                for c in range(pts_feat.shape[0]):
                    feat_2d[h][w] += abs(pts_feat[c][h][w])

        plt.imshow(feat_2d, cmap=plt.cm.gray, vmin=0, vmax=255, extent=[-50, 50, -50, 50], origin = 'lower') # extent=（left,right,bottom, top） # 注意图片坐标原点！
        #plt.savefig("./vis_results" + name + '-' + img_metas[idx]['sample_idx'] + ".png")
        plt.savefig("./vis_results" + name + '-' + str(idx) + ".png")
    print('\n******** END PRINT GT **********\n')
    exit(0)


def print_pcgt_on_bev(pc, gt, name='vis'):
    points = pc
    if gt!=None:
        if gt.shape[0] > 0:
            x = gt[:, 0].reshape(-1)
            y = gt[:, 1].reshape(-1)
            z = gt[:, 2].reshape(-1)
            dx = gt[:, 3].reshape(-1)
            dy = gt[:, 4].reshape(-1)
            dz = gt[:, 5].reshape(-1)
            yaw = gt[:, 6].reshape(-1)
            ll = x.shape[0]

    fig = plt.figure(figsize=(32, 32))
    # plt.plot([100, 100, -100, -100, 100], [100, -100, -100, 100, 100], lw=0.5)
    plt.plot([50, 50, -50, -50, 50], [50, -50, -50, 50, 50], lw=0.5)

    xxx = points[:, 0]
    yyy = points[:, 1]
    zzz = points[:, 2]
    plt.scatter(xxx, yyy, s=0.5)

    if gt!=None:
        if gt.shape[0] > 0:
            for i in range(ll):
                xcos = dx[i] * cos(yaw[i]) / 2
                xsin = dx[i] * sin(yaw[i]) / 2
                ycos = dy[i] * cos(yaw[i]) / 2
                ysin = dy[i] * sin(yaw[i]) / 2
                x1 = x[i] + xcos - ysin
                y1 = y[i] + xsin + ycos
                x2 = x[i] + xcos + ysin
                y2 = y[i] + xsin - ycos
                x3 = x[i] - xcos + ysin
                y3 = y[i] - xsin - ycos
                x4 = x[i] - xcos - ysin
                y4 = y[i] - xsin + ycos
                plt.plot([x1,x4,x3,x2,x1], [y3,y2,y1,y4,y3], lw=2)

    plt.savefig("./vis_results/" + name + ".png")


def print_pcgt_on_bev_radar(pc, gt, name='visradar'):
    
    for idx in range(len(gt)):
        points = pc[idx].cpu().numpy()
        corner = gt[idx].corners
        fig = plt.figure(figsize=(16, 16))
        # plt.plot([100, 100, -100, -100, 100], [100, -100, -100, 100, 100], lw=0.5)
        plt.plot([50, 50, -50, -50, 50], [50, -50, -50, 50, 50], lw=0.5)
        for i in range(corner.shape[0]):
            x1 = corner[i][0][0]
            y1 = corner[i][0][1]
            x2 = corner[i][2][0]
            y2 = corner[i][2][1]
            x3 = corner[i][6][0]
            y3 = corner[i][6][1]
            x4 = corner[i][4][0]
            y4 = corner[i][4][1]
            plt.plot([x1,x2,x3,x4,x1], [y1,y2,y3,y4,y1], lw=2)



        xxx = points[:, 0]
        yyy = points[:, 1]
        zzz = points[:, 2]
        plt.xlim(-50, 50)  
        plt.ylim(-50, 50)
        # indices = (xxx >= -50) & (xxx <= 50) & (yyy >= -50) & (yyy <= 50)


        # xxx_filtered = xxx[indices]
        # yyy_filtered = yyy[indices]
        plt.scatter(xxx, yyy, s=2)


 
        plt.savefig("./vis_results" + name + str(idx) + ".png")
    
    exit(0)


def print_sample_and_bev(pts_feats, sample, name='vis'):
    print('\n******** BEGIN PRINT GT **********\n')
    #assert(pts_feats.shape[0] == len(gt))
    sample = sample.cpu().detach().numpy()
    for idx in range(pts_feats.shape[0]):
        # if img_metas[idx]['sample_idx'] != 'ca9a282c9e77460f8360f564131a8af5':
            # continue
        #corner = gt[idx].corners

        fig = plt.figure(figsize=(16, 16))

        plt.plot([50,50,-50,-50,50], [50,-50,-50,50,50], lw=0.5)


        xxx = sample[idx, :, 0, ..., 0] # [B, Q, T, G, P, 3] torch.Size([8, 1290, 8, 4, 4, 3])无时序
        yyy = sample[idx, :, 0, ..., 1] 
        zzz = sample[idx, :, 0, ..., 2]
        plt.scatter(xxx, yyy, s=0.5)

        pts_feat = pts_feats[idx].cpu().detach().numpy() # 放到cpu上 去掉梯度 转换成numpy
        print('pts_feat.shape =', pts_feat.shape)
        feat_2d = np.zeros(pts_feat.shape[1:])
        print('feat_2d.shape =', feat_2d.shape)
        for h in range(feat_2d.shape[0]):
            for w in range(feat_2d.shape[1]):
                for c in range(pts_feat.shape[0]):
                    feat_2d[h][w] += abs(pts_feat[c][h][w])

        plt.imshow(feat_2d, cmap=plt.cm.gray, vmin=0, vmax=255, extent=[-50, 50, -50, 50], origin = 'lower') # extent=（left,right,bottom, top） # 注意图片坐标原点！
        #plt.savefig("./vis_results" + name + '-' + img_metas[idx]['sample_idx'] + ".png")
        plt.savefig("./vissample_results" + name + '-' + str(idx) + ".png")
    print('\n******** END PRINT GT **********\n')
    exit(0)