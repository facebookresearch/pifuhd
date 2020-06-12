# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import trimesh
import trimesh.proximity
import trimesh.sample
import numpy as np
import math
import os
from PIL import Image

import argparse

def euler_to_rot_mat(r_x, r_y, r_z):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(r_x), -math.sin(r_x)],
                    [0, math.sin(r_x), math.cos(r_x)]
                    ])

    R_y = np.array([[math.cos(r_y), 0, math.sin(r_y)],
                    [0, 1, 0],
                    [-math.sin(r_y), 0, math.cos(r_y)]
                    ])

    R_z = np.array([[math.cos(r_z), -math.sin(r_z), 0],
                    [math.sin(r_z), math.cos(r_z), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


class MeshEvaluator:
    _normal_render = None

    @staticmethod
    def init_gl():
        from .render.gl.normal_render import NormalRender
        MeshEvaluator._normal_render = NormalRender(width=512, height=512)

    def __init__(self):
        pass

    def set_mesh(self, src_path, tgt_path, scale_factor=1.0, offset=0):
        self.src_mesh = trimesh.load(src_path)
        self.tgt_mesh = trimesh.load(tgt_path)

        self.scale_factor = scale_factor
        self.offset = offset


    def get_chamfer_dist(self, num_samples=10000):
        # Chamfer
        src_surf_pts, _ = trimesh.sample.sample_surface(self.src_mesh, num_samples)
        tgt_surf_pts, _ = trimesh.sample.sample_surface(self.tgt_mesh, num_samples)

        _, src_tgt_dist, _ = trimesh.proximity.closest_point(self.tgt_mesh, src_surf_pts)
        _, tgt_src_dist, _ = trimesh.proximity.closest_point(self.src_mesh, tgt_surf_pts)

        src_tgt_dist[np.isnan(src_tgt_dist)] = 0
        tgt_src_dist[np.isnan(tgt_src_dist)] = 0

        src_tgt_dist = src_tgt_dist.mean()
        tgt_src_dist = tgt_src_dist.mean()

        chamfer_dist = (src_tgt_dist + tgt_src_dist) / 2

        return chamfer_dist

    def get_surface_dist(self, num_samples=10000):
        # P2S
        src_surf_pts, _ = trimesh.sample.sample_surface(self.src_mesh, num_samples)

        _, src_tgt_dist, _ = trimesh.proximity.closest_point(self.tgt_mesh, src_surf_pts)

        src_tgt_dist[np.isnan(src_tgt_dist)] = 0

        src_tgt_dist = src_tgt_dist.mean()

        return src_tgt_dist

    def _render_normal(self, mesh, deg):
        view_mat = np.identity(4)
        view_mat[:3, :3] *= 2 / 256
        rz = deg / 180. * np.pi
        model_mat = np.identity(4)
        model_mat[:3, :3] = euler_to_rot_mat(0, rz, 0)
        model_mat[1, 3] = self.offset
        view_mat[2, 2] *= -1

        self._normal_render.set_matrices(view_mat, model_mat)
        self._normal_render.set_normal_mesh(self.scale_factor*mesh.vertices, mesh.faces, mesh.vertex_normals, mesh.faces)
        self._normal_render.draw()
        normal_img = self._normal_render.get_color()
        return normal_img

    def _get_reproj_normal_error(self, deg):
        tgt_normal = self._render_normal(self.tgt_mesh, deg)
        src_normal = self._render_normal(self.src_mesh, deg)

        error = ((src_normal[:, :, :3] - tgt_normal[:, :, :3]) ** 2).mean() * 3

        return error, src_normal, tgt_normal

    def get_reproj_normal_error(self, frontal=True, back=True, left=True, right=True, save_demo_img=None):
        # reproj error
        # if save_demo_img is not None, save a visualization at the given path (etc, "./test.png")
        if self._normal_render is None:
            print("In order to use normal render, "
                  "you have to call init_gl() before initialing any evaluator objects.")
            return -1

        side_cnt = 0
        total_error = 0
        demo_list = []
        if frontal:
            side_cnt += 1
            error, src_normal, tgt_normal = self._get_reproj_normal_error(0)
            total_error += error
            demo_list.append(np.concatenate([src_normal, tgt_normal], axis=0))
        if back:
            side_cnt += 1
            error, src_normal, tgt_normal = self._get_reproj_normal_error(180)
            total_error += error
            demo_list.append(np.concatenate([src_normal, tgt_normal], axis=0))
        if left:
            side_cnt += 1
            error, src_normal, tgt_normal = self._get_reproj_normal_error(90)
            total_error += error
            demo_list.append(np.concatenate([src_normal, tgt_normal], axis=0))
        if right:
            side_cnt += 1
            error, src_normal, tgt_normal = self._get_reproj_normal_error(270)
            total_error += error
            demo_list.append(np.concatenate([src_normal, tgt_normal], axis=0))
        if save_demo_img is not None:
            res_array = np.concatenate(demo_list, axis=1)
            res_img = Image.fromarray((res_array * 255).astype(np.uint8))
            res_img.save(save_demo_img)
        return total_error / side_cnt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, required=True)
    parser.add_argument('-t', '--tar_path', type=str, required=True)
    args = parser.parse_args()

    evaluator = MeshEvaluator()
    evaluator.init_gl()

    def run(root, exp_name, tar_path):
        src_path = os.path.join(root, exp_name, 'recon')
        rp_path = os.path.join(tar_path, 'RP', 'GEO', 'OBJ')
        bf_path = os.path.join(tar_path, 'BUFF', 'GEO', 'PLY')

        buff_files = [f for f in os.listdir(bf_path) if '.ply' in f]

        src_names = ['0_0_00.obj', '90_0_00.obj', '180_0_00.obj', '270_0_00.obj']

        total_vals = []
        items = []
        for file in buff_files:
            tar_name = os.path.join(bf_path, file)
            name = tar_name.split('/')[-1][:-4]

            for src in src_names:
                src_name = os.path.join(src_path, 'result_%s_%s' % (name, src))
                if not os.path.exists(src_name):
                    continue
                evaluator.set_mesh(src_name, tar_name, 0.13, -40)

                vals = []
                vals.append(0.1 * evaluator.get_chamfer_dist())
                vals.append(0.1 * evaluator.get_surface_dist())
                vals.append(4.0 * evaluator.get_reproj_normal_error(save_demo_img=os.path.join(src_path, '%s_%s.png' % (name, src[:-4]))))

                item = {
                    'name': '%s_%s' % (name, src),
                    'vals': vals
                }

                total_vals.append(vals)
                items.append(item)

        vals = np.array(total_vals).mean(0)
        buf_val = vals

        np.save(os.path.join(root, exp_name, 'buff-item.npy'), np.array(items))
        np.save(os.path.join(root, exp_name, 'buff-vals.npy'), total_vals)

        rp_files = [f for f in os.listdir(rp_path) if '.obj' in f]

        total_vals = []
        items = []
        for file in rp_files:
            tar_name = os.path.join(rp_path, file)
            name = tar_name.split('/')[-1][:-9]

            for src in src_names:
                src_name = os.path.join(src_path, 'result_%s_%s' % (name, src))
                if not os.path.exists(src_name):
                    continue

                evaluator.set_mesh(src_name, tar_name, 1.3, -120)

                vals = []
                vals.append(evaluator.get_chamfer_dist())
                vals.append(evaluator.get_surface_dist())
                vals.append(4.0 * evaluator.get_reproj_normal_error(save_demo_img=os.path.join(src_path, '%s_%s.png' % (name, src[:-4]))))

                item = {
                    'name': '%s_%s' % (name, src),
                    'vals': vals
                }

                total_vals.append(vals)
                items.append(item)

        np.save(os.path.join(root, exp_name, 'rp-item.npy'), np.array(items))
        np.save(os.path.join(root, exp_name, 'rp-vals.npy'), total_vals)

        vals = np.array(total_vals).mean(0)
        print('BUFF - chamfer: %.4f  p2s: %.4f  nml: %.4f' % (buf_val[0], buf_val[1], buf_val[2]))
        print('RP - chamfer: %.4f  p2s: %.4f  nml: %.4f' % (vals[0], vals[1], vals[2]))

    exp_list = ['pifuhd_final']

    root = args.root
    tar_path = args.tar_path

    for exp in exp_list:
        run(root, exp, tar_path)