'''
MIT License

Copyright (c) 2019 Shunsuke Saito, Zeng Huang, and Ryota Natsume

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
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
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)
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