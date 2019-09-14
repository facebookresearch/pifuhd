import os
import random

import numpy as np 
from numpy.linalg import inv
from PIL import Image, ImageOps
from PIL.ImageFilter import GaussianBlur
import cv2
import trimesh

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from tqdm import tqdm
from .RPDataset import RPDataset

def load_trimesh(root, n_verts='30k', interval=0):
    folders = os.listdir(os.path.join(root, 'OBJ'))
    
    meshes = {}
    cnt = 0
    mesh_dics = []
    for i, f in enumerate(folders):
        sub_name = f 
        obj_path = os.path.join(root, 'OBJ', f, '%s_%s.obj' % (sub_name, n_verts))
        if os.path.exists(obj_path):
            meshes[sub_name] = trimesh.load(obj_path)
        if interval != 0 and i % interval == 0 and i != 0:
            mesh_dics.append(meshes)
            meshes = {}
    
    if len(meshes) != 0:
        mesh_dics.append(meshes)

    return mesh_dics

class RPOtfDataset(RPDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):
        RPDataset.__init__(self, opt, phase=phase)
        self.mesh_dic = {}
        self.DICT = os.path.join(os.path.join(self.root, 'GEO', 'npy_files'))
        if not os.path.isdir(self.DICT):
            mesh_dic = load_trimesh(self.root)
            for dic in mesh_dic:
                self.mesh_dic.update(dic)
        else:
            print('loading mesh_dic...')
            for i in tqdm(range(self.opt.num_pts_dic)):
                self.mesh_dic = {**self.mesh_dic, **(np.load(os.path.join(self.DICT, 'trimesh_dic%d.npy' % i)).item())}

    def select_sampling_method(self, subject, calib):
        # test only
        if not self.is_train:
             np.random.seed(1991)
        mesh = self.mesh_dic[subject]
        if 'sigma' in self.opt.sampling_mode:
            surface_points, fid = trimesh.sample.sample_surface(mesh, 4 * self.num_sample_inout)
            sample_points = surface_points + np.random.normal(scale=self.opt.sigma_max, size=surface_points.shape)
        if self.opt.sampling_mode == 'uniform':
            # add random points within image space
            random_points = np.concatenate(
                [2.0 * np.random.rand(self.num_sample_inout, 3) - 1.0, np.ones((self.num_sample_inout, 1))],
                1)  # [-1,1]
            random_points = np.matmul(random_points, inv(calib).T)[:, :3]
            # length = self.B_MAX - self.B_MIN
            # sample_points = np.random.rand(self.num_sample_inout, 3) * length + self.B_MIN
        elif 'uniform' in self.opt.sampling_mode:
            # add random points within image space
            random_points = np.concatenate(
                [2.0 * np.random.rand(self.num_sample_inout, 3) - 1.0, np.ones((self.num_sample_inout, 1))],
                1)  # [-1,1]
            random_points = np.matmul(random_points, inv(calib).T)[:, :3]
            # length = self.B_MAX - self.B_MIN
            # random_points = np.random.rand(self.num_sample_inout // 4, 3) * length + self.B_MIN
            sample_points = np.concatenate([sample_points, random_points], 0)
            np.random.shuffle(sample_points)

        inside = mesh.contains(sample_points)
        inside_points = sample_points[inside]
        outside_points = sample_points[np.logical_not(inside)]

        nin = inside_points.shape[0]
        inside_points = inside_points[
                        :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else inside_points
        outside_points = outside_points[
                         :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else outsize_points[
                             :(self.num_sample_inout - nin)]    

        samples = np.concatenate([inside_points, outside_points], 0).T # [3, N]
        samples = torch.Tensor(samples).float()
        labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1)
        labels = torch.Tensor(labels).float()
        return {
            'samples': samples,
            'labels': labels
        }
