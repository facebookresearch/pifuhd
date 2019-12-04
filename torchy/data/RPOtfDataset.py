import os
import math
import sys
import random
import copy
import gzip 
import time

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
from .RPDatasetParts import RPDatasetParts as RPDataset
from lib.sample_util import *

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
    
    def precompute_points(self, subject, num_files=1, start_id=0, sigma=None):
        if sigma is None:
            sigma = self.opt.sigma
        SAMPLE_DIR = os.path.join(self.SAMPLE, self.opt.sampling_mode, subject)

        mesh = copy.deepcopy(g_mesh_dics[subject])
        ratio = 0.8
        for i in range(start_id, start_id+num_files):
            data_file = os.path.join(SAMPLE_DIR, '%05d.io.npy' % i)
            if 'sigma' in self.opt.sampling_mode:
                surface_points, fid = trimesh.sample.sample_surface(mesh, int(ratio * self.num_sample_inout))
                theta = 2.0 * math.pi * np.random.rand(surface_points.shape[0])
                phi = np.arccos(1 - 2 * np.random.rand(surface_points.shape[0]))
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)
                dir = np.stack([x,y,z],1)
                radius = np.random.normal(scale=sigma, size=[surface_points.shape[0],1])
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
            inside_points = sample_points[inside]
            outside_points = sample_points[np.logical_not(inside)]

            data = {'in': inside_points, 'out': outside_points}

            os.makedirs(SAMPLE_DIR, exist_ok=True)
            np.save(data_file, data)

        del mesh
        gc.collect()