import os
import math
import sys
import random
import copy

import numpy as np 
from numpy.linalg import inv
from PIL import Image, ImageOps
from PIL.ImageFilter import GaussianBlur
import cv2
import trimesh
from trimesh.proximity import *
import matplotlib
import matplotlib.cm as cm

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from tqdm import tqdm
from .RPDataset import RPDataset

import gc

# this is necessary to remove warning from trimesh
import logging
trimesh.util.attach_to_log(level=logging.CRITICAL, capture_warnings=False)

g_mesh_dics = None

def scalar2color(x, min=-1.0, max=1.0):

    norm = matplotlib.colors.Normalize(vmin=min, vmax=max, clip=True)
    # use jet colormap
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)

    colors = []
    for v in x:
        colors.append(mapper.to_rgba(v))

    return np.stack(colors, 0)[:,:3]

def save_points_color(filename, V, C=None, F=None):
    with open(filename, "w") as file:
        for i in range(V.shape[0]):
            if C is not None:
                file.write('v %f %f %f %f %f %f\n' % (V[i,0], V[i,1], V[i,2], C[i,0], C[i,1], C[i,2]))
            else:
                file.write('v %f %f %f\n' % (V[i,0], V[i,1], V[i,2]))
                
        if F is not None:
            for i in range(F.shape[0]):
                file.write('f %d %d %d\n' % (F[i,0]+1, F[i,1]+1, F[i,2]+1))
        file.close()    

def load_trimesh(root, n_verts='100k', interval=0):
    folders = os.listdir(os.path.join(root, 'GEO', 'OBJ'))
    
    meshes = {}
    cnt = 0
    mesh_dics = []
    failed_list = []
    for i, f in enumerate(folders):
        sub_name = f[:-8] 
        print(sub_name)
        obj_path = os.path.join(root, 'GEO', 'OBJ', f, '%s_%s.obj' % (sub_name, n_verts))
        if os.path.exists(obj_path):
            try:
                mesh = trimesh.load(obj_path)
                meshes[sub_name] = trimesh.Trimesh(mesh.vertices, mesh.faces)
            except:
                failed_list.append(sub_name)
                print('mesh load failed %s' % sub_name)
        if interval != 0 and i % interval == 0 and i != 0:
            mesh_dics.append(meshes)
            meshes = {}
    
    if len(meshes) != 0:
        mesh_dics.append(meshes)
    
    print('failed subject')
    print(failed_list)

    return mesh_dics

class RPOtfDataset(RPDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):
        RPDataset.__init__(self, opt, phase=phase)
        self.DICT = os.path.join(os.path.join(self.root, 'GEO', 'npy_files'))
        global g_mesh_dics
        if g_mesh_dics is None:
            g_mesh_dics = {}
            if not os.path.isdir(self.DICT):
                os.makedirs(self.DICT,exist_ok=True)
                mesh_dic = load_trimesh(self.root)
                for i, dic in enumerate(mesh_dic):
                    np.save(os.path.join(self.DICT, 'trimesh_dic%d.npy' % i), dic)
                    g_mesh_dics.update(dic)
            else:
                print('loading mesh_dic...')
                for i in tqdm(range(self.opt.num_pts_dic)):
                    g_mesh_dics = {**g_mesh_dics, **(np.load(os.path.join(self.DICT, 'trimesh_dic%d.npy' % i),allow_pickle=True).item())}
    
    def precompute_points(self, subject, num_files=1):
        SAMPLE_DIR = os.path.join(self.SAMPLE, self.opt.sampling_mode, subject)

        mesh = copy.deepcopy(g_mesh_dics[subject])
        ratio = 0.8
        for i in tqdm(range(num_files)):
            data_file = os.path.join(SAMPLE_DIR, '%05d.io' % i)
            if 'sigma' in self.opt.sampling_mode:
                surface_points, fid = trimesh.sample.sample_surface_even(mesh, int(ratio * self.num_sample_inout))
                theta = 2.0 * math.pi * np.random.rand(surface_points.shape[0])
                phi = np.arccos(1 - 2 * np.random.rand(surface_points.shape[0]))
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)
                dir = np.stack([x,y,z],1)
                radius = np.random.normal(scale=self.opt.sigma, size=[surface_points.shape[0],1])
                sample_points = surface_points + radius * dir
            if self.opt.sampling_mode == 'uniform':
                # add random points within image space
                random_points = np.concatenate(
                    [2.0 * np.random.rand(2.0*self.num_sample_inout, 3) - 1.0, np.ones((2.0*self.num_sample_inout, 1))],
                    1)  # [-1,1]
                length = self.B_MAX - self.B_MIN
                sample_points = np.random.rand(self.num_sample_inout, 3) * length + self.B_MIN
            elif 'uniform' in self.opt.sampling_mode:
                # add random points within image space
                length = self.B_MAX - self.B_MIN
                random_points = np.random.rand(int((1.0-ratio)*self.num_sample_inout), 3) * length + self.B_MIN
                sample_points = np.concatenate([sample_points, random_points], 0)
                np.random.shuffle(sample_points)

            inside = mesh.contains(sample_points)

            data = {'points': sample_points, 'labels': inside}
            
            os.makedirs(SAMPLE_DIR, exist_ok=True)
            np.save(data_file, data)

        del mesh
        gc.collect()


    def precompute_tsdf(self, subject, num_files=100, sigma=1.0):
        TSDF_DIR = os.path.join(self.TSDF, self.opt.sampling_mode, subject)

        mesh = g_mesh_dics[subject]

        for i in tqdm(range(num_files)):
            tsdf_file = os.path.join(TSDF_DIR, '%05d.xyzd' % i)

            if 'sigma' in self.opt.sampling_mode:
                surface_points, fid = trimesh.sample.sample_surface_even(mesh, self.num_sample_inout)
                sample_points = surface_points + np.random.normal(scale=self.opt.sigma, size=surface_points.shape)
                sample_points = sample_points[:(3*self.num_sample_inout//4)]
            if self.opt.sampling_mode == 'uniform':
                length = self.B_MAX - self.B_MIN
                sample_points = np.random.rand(self.num_sample_inout, 3) * length + self.B_MIN
            elif 'uniform' in self.opt.sampling_mode:
                length = self.B_MAX - self.B_MIN
                random_points = np.random.rand(self.num_sample_inout // 4, 3) * length + self.B_MIN
                sample_points = np.concatenate([sample_points, random_points], 0)
            dist = signed_distance(mesh, sample_points)
            dist /= sigma
            dist = dist.clip(-1, 1)
            sample_points = np.concatenate([sample_points, dist[:,None]], 1)

            os.makedirs(TSDF_DIR, exist_ok=True)
            np.save(tsdf_file, sample_points)
              

    def select_sampling_method(self, subject, calib):
        # test only
        if not self.is_train:
            random.seed(1991)
            np.random.seed(1991)
        mesh = copy.deepcopy(g_mesh_dics[subject])
        ratio = 0.8
        if 'sigma' in self.opt.sampling_mode:
            surface_points, fid = trimesh.sample.sample_surface_even(mesh, int(1.4 * ratio * self.num_sample_inout))
            theta = 2.0 * math.pi * np.random.rand(surface_points.shape[0])
            phi = np.arccos(1 - 2 * np.random.rand(surface_points.shape[0]))
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            dir = np.stack([x,y,z],1)
            radius = np.random.normal(scale=self.opt.sigma if self.is_train else 3.0, size=[surface_points.shape[0],1])
            sample_points = surface_points + radius * dir
        if self.opt.sampling_mode == 'uniform':
            # add random points within image space
            random_points = np.concatenate(
                [2.0 * np.random.rand(2.0*self.num_sample_inout, 3) - 1.0, np.ones((2.0*self.num_sample_inout, 1))],
                1)  # [-1,1]
            random_points = np.matmul(random_points, inv(calib).T)[:, :3]
            # length = self.B_MAX - self.B_MIN
            # sample_points = np.random.rand(self.num_sample_inout, 3) * length + self.B_MIN
        elif 'uniform' in self.opt.sampling_mode:
            # add random points within image space
            random_points = np.concatenate(
                [2.0 * np.random.rand(int((1.0-ratio)*self.num_sample_inout), 3) - 1.0, np.ones((int((1.0-ratio)*self.num_sample_inout), 1))],
                1)  # [-1,1]
            random_points = np.matmul(random_points, inv(calib).T)[:, :3]
            # length = self.B_MAX - self.B_MIN
            # random_points = np.random.rand(self.num_sample_inout // 4, 3) * length + self.B_MIN
            sample_points = np.concatenate([sample_points, random_points], 0)
            np.random.shuffle(sample_points)

        inbb = np.matmul(np.concatenate([sample_points, np.ones((sample_points.shape[0],1))], 1), calib.T)[:, :3]
        inbb = (inbb[:, 0] >= -1) & (inbb[:, 0] <= 1) & (inbb[:, 1] >= -1) & \
               (inbb[:, 1] <= 1) & (inbb[:, 2] >= -1) & (inbb[:, 2] <= 1)

        sample_points = sample_points[inbb]
        inside = mesh.contains(sample_points)
        inside_points = sample_points[inside]
        outside_points = sample_points[np.logical_not(inside)]

        nin = inside_points.shape[0]
        inside_points = inside_points[
                        :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else inside_points
        outside_points = outside_points[
                         :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else outside_points[
                             :(self.num_sample_inout - nin)]    

        samples = np.concatenate([inside_points, outside_points], 0).T # [3, N]
        samples = torch.Tensor(samples).float()
        labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1)
        labels = torch.Tensor(labels).float()
        
        del mesh
        gc.collect()
        return {
            'samples': samples,
            'labels': labels,
        }
