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
import numpy as np


def create_grid(resX, resY, resZ, b_min=np.array([-1, -1, -1]), b_max=np.array([1, 1, 1]), transform=None):
    '''
    Create a dense grid of given resolution and bounding box
    :param resX: resolution along X axis
    :param resY: resolution along Y axis
    :param resZ: resolution along Z axis
    :param b_min: vec3 (x_min, y_min, z_min) bounding box corner
    :param b_max: vec3 (x_max, y_max, z_max) bounding box corner
    :return: [3, resX, resY, resZ] coordinates of the grid, and transform matrix from mesh index
    '''
    coords = np.mgrid[:resX, :resY, :resZ]
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4)
    length = b_max - b_min
    coords_matrix[0, 0] = length[0] / resX
    coords_matrix[1, 1] = length[1] / resY
    coords_matrix[2, 2] = length[2] / resZ
    coords_matrix[0:3, 3] = b_min
    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    if transform is not None:
        coords = np.matmul(transform[:3, :3], coords) + transform[:3, 3:4]
        coords_matrix = np.matmul(transform, coords_matrix)
    coords = coords.reshape(3, resX, resY, resZ)
    return coords, coords_matrix


def batch_eval(points, eval_func, num_samples=512 * 512 * 512):
    num_pts = points.shape[1]
    sdf = np.zeros(num_pts)

    num_batches = num_pts // num_samples
    for i in range(num_batches):
        sdf[i * num_samples:i * num_samples + num_samples] = eval_func(
            points[:, i * num_samples:i * num_samples + num_samples])
    if num_pts % num_samples:
        sdf[num_batches * num_samples:] = eval_func(points[:, num_batches * num_samples:])

    return sdf

def batch_eval_tensor(points, eval_func, num_samples=512 * 512 * 512):
    num_pts = points.size(1)

    num_batches = num_pts // num_samples
    vals = []
    for i in range(num_batches):
        vals.append(eval_func(points[:, i * num_samples:i * num_samples + num_samples]))
    if num_pts % num_samples:
        vals.append(eval_func(points[:, num_batches * num_samples:]))

    return np.concatenate(vals,0)

def eval_grid(coords, eval_func, num_samples=512 * 512 * 512):
    resolution = coords.shape[1:4]
    coords = coords.reshape([3, -1])
    sdf = batch_eval(coords, eval_func, num_samples=num_samples)
    return sdf.reshape(resolution)


import time
def eval_grid_octree(coords, eval_func,
                     init_resolution=64, threshold=0.05,
                     num_samples=512 * 512 * 512):
    resolution = coords.shape[1:4]

    sdf = np.zeros(resolution)

    notprocessed = np.zeros(resolution, dtype=np.bool)
    notprocessed[:-1,:-1,:-1] = True
    grid_mask = np.zeros(resolution, dtype=np.bool)

    reso = resolution[0] // init_resolution

    while reso > 0:
        # subdivide the grid
        grid_mask[0:resolution[0]:reso, 0:resolution[1]:reso, 0:resolution[2]:reso] = True
        # test samples in this iteration
        test_mask = np.logical_and(grid_mask, notprocessed)
        # print('step size:', reso, 'test sample size:', test_mask.sum())
        points = coords[:, test_mask]

        sdf[test_mask] = batch_eval(points, eval_func, num_samples=num_samples)
        notprocessed[test_mask] = False

        # do interpolation
        if reso <= 1:
            break
        x_grid = np.arange(0, resolution[0], reso)
        y_grid = np.arange(0, resolution[1], reso)
        z_grid = np.arange(0, resolution[2], reso)

        v = sdf[tuple(np.meshgrid(x_grid, y_grid, z_grid, indexing='ij'))]

        v0 = v[:-1,:-1,:-1]
        v1 = v[:-1,:-1,1:]
        v2 = v[:-1,1:,:-1]
        v3 = v[:-1,1:,1:]
        v4 = v[1:,:-1,:-1]
        v5 = v[1:,:-1,1:]
        v6 = v[1:,1:,:-1]
        v7 = v[1:,1:,1:]

        x_grid = x_grid[:-1] + reso//2
        y_grid = y_grid[:-1] + reso//2
        z_grid = z_grid[:-1] + reso//2

        nonprocessed_grid = notprocessed[tuple(np.meshgrid(x_grid, y_grid, z_grid, indexing='ij'))]

        v = np.stack([v0,v1,v2,v3,v4,v5,v6,v7], 0)
        v_min = v.min(0)
        v_max = v.max(0)
        v = 0.5*(v_min+v_max)

        skip_grid = np.logical_and(((v_max - v_min) < threshold), nonprocessed_grid)

        n_x = resolution[0] // reso
        n_y = resolution[1] // reso
        n_z = resolution[2] // reso

        xs, ys, zs = np.where(skip_grid)
        for x, y, z in zip(xs*reso, ys*reso, zs*reso):
            sdf[x:(x+reso+1), y:(y+reso+1), z:(z+reso+1)] = v[x//reso,y//reso,z//reso]
            notprocessed[x:(x+reso+1), y:(y+reso+1), z:(z+reso+1)] = False
        reso //= 2

    return sdf.reshape(resolution)
