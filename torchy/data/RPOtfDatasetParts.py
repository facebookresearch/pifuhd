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
# from .RPDataset import RPDataset
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

class RPOtfDatasetParts(RPDataset):
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
    
    def get_sample(self, subject, calib, mask=None):
            return self.load_points_sample(subject, calib, mask, self.opt.sigma_surface)

    def get_otf_sampling(self, subject, calib, mask, sample_data):
        # test only
        if not self.is_train:
            random.seed(1991)
            np.random.seed(1991)
            sigma = 10.0
        else:
            sigma = self.opt.sigma
        
        mesh = copy.deepcopy(g_mesh_dics[subject])
        surface_points = None
        ratio = 1.0 - self.opt.uniform_ratio
        sample_size = int(1.5 * ratio * self.opt.num_sample_surface)
        for i in range(10):
            sample_points, fid = trimesh.sample.sample_surface(mesh, int(10 * sample_size))
            ptsh = np.matmul(np.concatenate([sample_points, np.ones((sample_points.shape[0],1))], 1), calib.T)[:, :3]
            inbb = (ptsh[:, 0] >= -1) & (ptsh[:, 0] <= 1) & (ptsh[:, 1] >= -1) & \
                    (ptsh[:, 1] <= 1) & (ptsh[:, 2] >= -1) & (ptsh[:, 2] <= 1)
            x = (self.load_size * (0.5 * ptsh[:,0] + 0.5)).astype(np.int32).clip(0, self.load_size-1)
            y = (self.load_size * (0.5 * ptsh[:,1] + 0.5)).astype(np.int32).clip(0, self.load_size-1)
            idx = y * self.load_size + x
            inmask = mask.reshape(-1)[idx] > 0
            inmask = inmask & inbb
            sample_points = sample_points[inmask]
            if surface_points is None:
                surface_points = sample_points
            else:
                surface_points = np.concatenate([surface_points, sample_points], 0)
            if surface_points.shape[0] >= sample_size:
                surface_points = surface_points[:sample_size]
                break
            if i == 9:
                raise IOError('failed surface point sampling %s' % subject)

        theta = 2.0 * math.pi * np.random.rand(surface_points.shape[0])
        phi = np.arccos(1 - 2 * np.random.rand(surface_points.shape[0]))
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        dir = np.stack([x,y,z],1)
        radius = np.random.normal(scale=sigma, size=[surface_points.shape[0],1])
        sample_points = surface_points + radius * dir

        random_points = np.concatenate(
            [2.0 * np.random.rand(int(1.5*(1.0-ratio)*self.opt.num_sample_surface), 3) - 1.0, np.ones((int(1.5*(1.0-ratio)*self.opt.num_sample_surface), 1))],
            1)  # [-1,1]
        random_points = np.matmul(random_points, inv(calib).T)[:, :3]
        sample_points = np.concatenate([sample_points, random_points], 0)
        np.random.shuffle(sample_points)

        ptsh = np.matmul(np.concatenate([sample_points, np.ones((sample_points.shape[0],1))], 1), calib.T)[:, :3]
        inbb = (ptsh[:, 0] >= -1) & (ptsh[:, 0] <= 1) & (ptsh[:, 1] >= -1) & \
               (ptsh[:, 1] <= 1) & (ptsh[:, 2] >= -1) & (ptsh[:, 2] <= 1)

        sample_points = sample_points[inbb]
        sample_points = sample_points[:self.opt.num_sample_surface]

        if sample_points.shape[0] != self.opt.num_sample_surface:# + int(self.opt.uniform_ratio * self.num_sample_inout):
            raise IOError('unable to sample sufficient number of points %s' % subject)

        inside = mesh.contains(sample_points)

        ptsh = ptsh[inbb][:self.opt.num_sample_surface]
        x = (self.load_size * (0.5 * ptsh[:,0] + 0.5)).astype(np.int32).clip(0, self.load_size-1)
        y = (self.load_size * (0.5 * ptsh[:,1] + 0.5)).astype(np.int32).clip(0, self.load_size-1)
        idx = y * self.load_size + x
        inmask = mask.reshape(-1)[idx] > 0
        inside = inside & inmask
        
        inside_points = sample_points[inside]
        outside_points = sample_points[np.logical_not(inside)]

        samples = np.concatenate([inside_points, outside_points], 0).T # [3, N]
        labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1)
        ratio = outside_points.shape[0]/samples.shape[0]

        samples = torch.Tensor(samples).float()
        labels = torch.Tensor(labels).float()
        
        del mesh
        gc.collect()

        sample_data['samples'] = torch.cat([sample_data['samples'], samples],1)
        sample_data['labels'] = torch.cat([sample_data['labels'], labels],1)
        sample_data['ratio'] = 1.0 - sample_data['labels'].sum().item() / sample_data['labels'].size(1)
        return sample_data

    def get_item(self, index):
        # in case of IO error, use random sampling instead
        subject = ''
        try:
            sid = index % len(self.subjects)
            tmp = index // len(self.subjects)

            # test use pitch == 0 only
            # also train doesn't use yaw == 0
            vid = tmp % len(self.yaw_list)
            pid = tmp // len(self.yaw_list)

            # name of the subjects 'rp_xxxx_xxx'
            subject = self.subjects[sid]
            res = {
                'name': subject,
                'mesh_path': os.path.join(self.OBJ, subject + '.obj'),
                'sid': sid,
                'vid': vid,
                'pid': pid,
                'b_min': self.B_MIN,
                'b_max': self.B_MAX,
            }
            render_data = self.get_render(sid, num_views=self.num_views, view_id=vid,
                                        pid=pid, random_sample=self.opt.random_multiview)
            sample_data = self.get_sample(subject, render_data['calib'][0].numpy(), render_data['mask'][0].numpy()) 
            sample_data = self.get_otf_sampling(subject, render_data['calib'][0].numpy(), render_data['mask'][0].numpy(), sample_data)
            
            # p = sample_data['samples'].t().numpy()
            # calib = render_data['calib'][0].numpy()
            # mask = (255.0*(0.5*render_data['img'][0].permute(1,2,0).numpy()[:,:,::-1]+0.5)).astype(np.uint8)
            # # mask = 255.0*np.stack(3*[render_data['mask'][0,0].numpy()],2)
            # p = np.matmul(np.concatenate([p, np.ones((p.shape[0],1))], 1), calib.T)[:, :3]
            # pts = 512*(0.5*p[sample_data['labels'].numpy().reshape(-1) == 1.0]+0.5)
            # for p in pts:
            #     mask = cv2.circle(mask, (int(p[0]),int(p[1])), 2, (0,255.0,0), -1)
            # mask = cv2.putText(mask, res['name'], (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), lineType=cv2.LINE_AA) 
            # cv2.imwrite('tmp.png', mask)
            # #cv2.waitKey(10)
            # exit()
            res.update(render_data)
            res.update(sample_data)
            if self.num_sample_normal:
                normal_data = self.get_normal_sampling(subject, render_data['calib'][0].numpy())
                res.update(normal_data)
            if self.num_sample_color:
                color_data = self.get_color_sampling(subject, view_id=vid)
                res.upate(color_data)
            return res
        except Exception as e:
            for i in range(10):
                try:
                    return self.get_item(index=random.randint(0, self.__len__() - 1)) 
                except Exception as e:
                    continue