import torch

def index(feat, uv):
    '''
    extract image features at floating coordinates with bilinear interpolation
    args:
        feat: [B, C, H, W] image features
        uv: [B, 2, N] normalized image coordinates ranged in [-1, 1]
    return:
        [B, C, N] sampled pixel values
    '''
    uv = uv.transpose(1, 2)
    uv = uv.unsqueeze(2)
    samples = torch.nn.functional.grid_sample(feat, uv)
    return samples[:, :, :, 0]

def orthogonal(points, calib, transform=None):
    '''
    project points onto screen space using orthogonal projection
    args:
        points: [B, 3, N] 3d points in world coordinates
        calib: [B, 3, 4] projection matrix
        transform: [B, 2, 3] screen space transformation
    return:
        [B, 3, N] 3d coordinates in screen space
    '''
    rot = calib[:, :3, :3]
    trans = calib[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)
    if transform is not None:
        scale = transform[:2, :2]
        shift = transform[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts

def perspective(points, calib, transform=None):
    '''
    project points onto screen space using perspective projection
    args:
        points: [B, 3, N] 3d points in world coordinates
        calib: [B, 3, 4] projection matrix
        transform: [B, 2, 3] screen space trasnformation
    return:
        [B, 3, N] 3d coordinates in screen space
    '''
    rot = calib[:, :3, :3]
    trans = calib[:, :3, 3:4]
    homo = torch.baddbmm(trans, rot, points)
    xy = homo[:, :2, :] / homo[:, 2:3, :]
    if transform is not None:
        scale = transform[:2, :2]
        shift = transform[:2, 2:3]
        xy = torch.baddbmm(shift, scale, xy)
    
    xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz