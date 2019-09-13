import os
import random

import numpy as np 
from PIL import Image, ImageOps
from PIL.ImageFilter import GaussianBlur
import cv2
import torch
import trimesh

from torch.utils.data import Dataset
import torchvision.transforms as transforms

class EvalDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, projection='orthogonal'):
        self.opt = opt
        self.projection_mode = projection

        self.root = self.opt.dataroot
        self.RENDER = os.path.join(self.root, 'RENDER')
        self.MASK = os.path.join(self.root, 'MASK')
        self.PARAM = os.path.join(self.root, 'PARAM')
        self.OBJ = os.path.join(self.root, 'GEO', 'OBJ')

        self.phase = 'val'
        self.load_size = self.opt.loadSize

        self.num_views = self.opt.num_views

        self.max_view_angle = 360
        self.interval = 1
        self.subjects = self.get_subjects()

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def get_subjects(self):
        var_file = os.path.join(self.root, 'val.txt')
        if os.path.exists(var_file):
            var_subjects = np.loadtxt(var_file, dtype=str)
            return sorted(list(var_subjects))
        all_subjects = os.listdir(self.RENDER)
        return sorted(list(all_subjects))


    def __len__(self):
        return len(self.subjects) * self.max_view_angle // self.interval

    def get_render(self, subject, num_views, pitch=0, view_id=None, random_sample=False):
        '''
        Return render data
        args:
            subject: subject name
            num_views: number of views
            pitch: pitch angle (default: 0)
            view_id: the first view id. if None, select randomly
        return:
            'img': None, # [num_views, C, H, W] input images
            'calib': None, # [num_views, 4, 4] calibration matrix
            'extrinsic': None, # [num_views, 4, 4] extrinsic matrix
            'mask': None, # [num_views, 1, H, W] segmentation masks
        '''
        if view_id is None:
            view_id = np.random.randint(self.max_view_angle)
        # views are sampled evenly unless random_sample is enabled
        view_ids = [(view_id + self.max_view_angle // num_views * offset) % self.max_view_angle
                    for offset in range(num_views)]
        if random_sample:
            view_ids = np.random.choices(self.max_view_angle, num_views, replace=False)

        calib_list = []
        render_list = []
        mask_list = []
        extrinsic_list = []

        for vid in view_ids:
            param_path = os.path.join(self.PARAM, subject, '%d_%d_%02d.npy' % (vid, pitch, 0))
            render_path = os.path.join(self.RENDER, subject, '%d_%d_%02d.jpg' % (vid, pitch, 0))
            mask_path = os.path.join(self.MASK, subject, '%d_%d_%02d.jpg' % (vid, pitch, 0)) 

            # load calibration data
            param = np.load(param_path, allow_pickle=True)
            # pixel unit / world unit
            ortho_ratio = param.item().get('ortho_ratio')
            # world unit / model unit
            scale = param.item().get('scale')
            # camera center world coordinate
            center = param.item().get('center')
            # model rotation
            R = param.item().get('R')

            translate = -np.matmul(R, center).reshape(3, 1)
            extrinsic = np.concatenate([R, translate], axis=1)
            extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
            # make sure camera space matches with pixel coordinates
            scale_intrinsic = np.identity(4)
            scale_intrinsic[0, 0] = scale / ortho_ratio
            scale_intrinsic[1, 1] = -scale / ortho_ratio # due to discripancy between OpenGL and CV
            scale_intrinsic[2, 2] = scale / ortho_ratio
            # transformation from pixe space to normalized space [0, 1]
            ndc_intrinsic = np.identity(4)
            ndc_intrinsic[0, 0] = 1.0 / float(self.opt.loadSize // 2)
            ndc_intrinsic[1, 1] = 1.0 / float(self.opt.loadSize // 2)
            ndc_intrinsic[2, 2] = 1.0 / float(self.opt.loadSize // 2)
            # transformation in normalized coordinates
            trans_intrinsic = np.identity(4)

            mask = Image.open(mask_path).convert('L')
            render = Image.open(render_path).convert('RGB')

            intrinsic = np.matmul(trans_intrinsic, np.matmul(ndc_intrinsic, scale_intrinsic))
            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
            extrinsic = torch.Tensor(extrinsic).float()

            mask = transforms.Resize(self.load_size)(mask)
            mask = transforms.ToTensor()(mask).float()
            mask_list.append(mask)

            render = self.to_tensor(render)
            render = mask.expand_as(render) * render

            render_list.append(render)
            calib_list.append(calib)
            extrinsic_list.append(extrinsic)

        return {
            'img': torch.stack(render_list, dim=0),
            'calib': torch.stack(calib_list, dim=0),
            'extrinsic': torch.stack(extrinsic_list, dim=0),
            'mask': torch.stack(mask_list, dim=0)
        }

    def get_item(self, index):
        # in case of IO error, use random sampling instead
        subject = ''
        try:
            sid = index % len(self.subjects)
            vid = (index // len(self.subjects)) * self.interval
            # name of the subjects 'rp_xxxx_xxx'
            subject = self.subjects[sid]
            res = {
                'name': subject,
                'mesh_path': os.path.join(self.OBJ, subject + '.obj'),
                'sid': sid,
                'vid': vid,
            }
            render_data = self.get_render(subject, num_views=self.num_views, view_id=vid,
                                          random_sample=self.opt.random_multiview)
            res.update(render_data)
            return res
        except Exception as e:
            print(e)
            return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)