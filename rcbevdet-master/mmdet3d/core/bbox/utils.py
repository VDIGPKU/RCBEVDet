import torch 

def normalize_bbox(bboxes, pc_range=None):

    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8] 
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes

def denormalize_bbox(normalized_bboxes, pc_range=None):
    # rotation 
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp() 
    l = l.exp() 
    h = h.exp() 
    if normalized_bboxes.size(-1) > 8:
         # velocity 
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes


def encode_bbox(bboxes, pc_range=None):
    xyz = bboxes[..., 0:3].clone()
    wlh = bboxes[..., 3:6].log()
    rot = bboxes[..., 6:7]

    if pc_range is not None:
        xyz[..., 0] = (xyz[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
        xyz[..., 1] = (xyz[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
        xyz[..., 2] = (xyz[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])

    if bboxes.shape[-1] > 7:
        vel = bboxes[..., 7:9].clone()
        return torch.cat([xyz, wlh, rot.sin(), rot.cos(), vel], dim=-1)
    else:
        return torch.cat([xyz, wlh, rot.sin(), rot.cos()], dim=-1)


def decode_bbox(bboxes, pc_range=None):
    xyz = bboxes[..., 0:3].clone()
    wlh = bboxes[..., 3:6].exp()
    rot = torch.atan2(bboxes[..., 6:7], bboxes[..., 7:8])

    if pc_range is not None:
        xyz[..., 0] = xyz[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
        xyz[..., 1] = xyz[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
        xyz[..., 2] = xyz[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]

    if bboxes.shape[-1] > 8:
        vel = bboxes[..., 8:10].clone()
        return torch.cat([xyz, wlh, rot, vel], dim=-1)
    else:
        return torch.cat([xyz, wlh, rot], dim=-1)


def normalize_bbox_polar(bboxes, pc_range):
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]

    theta_center = torch.atan2(cx, cy)
    theta_center = theta_center
    radius_center = torch.sqrt(cx ** 2 + cy ** 2)
    cx = theta_center
    cy = radius_center

    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, cz, w, l, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, cz, w, l, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes


def denormalize_bbox_polar(normalized_bboxes, pc_range):
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    theta_center = normalized_bboxes[..., 0:1]
    radius_center = normalized_bboxes[..., 1:2]

    cx = theta_center.sin() * radius_center
    cy = theta_center.cos() * radius_center

    cz = normalized_bboxes[..., 2:3]

    # size
    w = normalized_bboxes[..., 3:4]
    l = normalized_bboxes[..., 4:5]
    h = normalized_bboxes[..., 5:6]

    w = w.exp()
    l = l.exp()
    h = h.exp()
    if normalized_bboxes.size(-1) > 8:
        # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes
