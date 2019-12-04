import os
import sys
import random
import json
import math

import numpy as np 
from PIL import Image, ImageOps
from PIL.ImageFilter import GaussianBlur
import cv2
import trimesh
from numpy.linalg import inv

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from lib.sample_util import *

def loadPoses(root, subjects):
    dic = {}
    for sub in subjects:
        dic[sub] = np.load(os.path.join(root, '%s_100k.npy' % sub))

    return dic


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


def crop_image(img, rect):
    x, y, w, h = rect

    left = abs(x) if x < 0 else 0
    top = abs(y) if y < 0 else 0
    right = abs(img.shape[1]-(x+w)) if x + w >= img.shape[1] else 0
    bottom = abs(img.shape[0]-(y+h)) if y + h >= img.shape[0] else 0
    
    if img.shape[2] == 4:
        color = [0, 0, 0, 0]
    else:
        color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    x = x + left
    y = y + top

    return new_img[y:(y+h),x:(x+w),:]


def fullbody_crop(pts):
    pts = pts[pts[:,2] > 0.2]
    pmax = pts.max(0)
    pmin = pts.min(0)

    center = (0.5 * (pmax[:2] + pmin[:2])).astype(np.int)
    radius = int(0.65 * max(pmax[0]-pmin[0], pmax[1]-pmin[1]))

    x1 = center[0] - radius
    x2 = center[0] + radius
    y1 = center[1] - radius
    y2 = center[1] + radius

    return (x1, y1, x2-x1, y2-y1)

    
class EvalRPDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, projection='orthogonal'):
        self.opt = opt
        self.phase = 'val'
        self.projection_mode = projection

        self.root = self.opt.dataroot

        self.RENDER = os.path.join(self.root, 'RENDER')
        self.PARAM = os.path.join(self.root, 'PARAM')
        self.OBJ = os.path.join(self.root, 'GEO', 'OBJ')
        self.POSE2D = os.path.join(self.root, 'POSE2D', 'json')
        
        self.B_MIN = np.array([-128, -28, -128])
        self.B_MAX = np.array([128, 228, 128])

        try:
            file = open(os.path.join(self.root,'info.txt'),'r')
        except:
            raise IOError('%s does not exist!' % os.path.join(self.root,'info.txt'))

        lines = [f.strip() for f in file.readlines()]
        self.yaw_list = []
        self.pitch_list = []
        for l in lines:
            tmp = l.split(',')
            if tmp[0] == 'pitch' and self.phase in tmp[1]:
                self.pitch_list = [int(i) for i in tmp[2].strip('[]').split(' ')]
            if tmp[0] == 'yaw' and self.phase in tmp[1]:
                self.yaw_list = [int(i) for i in tmp[2].strip('[]').split(' ')]
        self.load_size = self.opt.loadSize

        self.subjects = sorted([f for f in os.listdir(self.RENDER)])
        
        self.crop_func = fullbody_crop

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.to_tensor_512 = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.to_tensor_mask = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor()
        ])

    
    def __len__(self):
        return len(self.subjects) * len(self.yaw_list) * len(self.pitch_list)

    def get_item(self, index):
        # in case of IO error, use random sampling instead
        subject = ''
        sid = index % len(self.subjects)
        tmp = index // len(self.subjects)

        # test use pitch == 0 only
        # also train doesn't use yaw == 0
        vid = tmp % len(self.yaw_list)
        pid = tmp // len(self.yaw_list)

        # name of the subjects 'rp_xxxx_xxx'
        subject = self.subjects[sid]


        vid = self.yaw_list[vid]
        pid = self.pitch_list[pid]

        param_path = os.path.join(self.PARAM, subject, '%d_%d_%02d.npy' % (vid, pid, 0))
        render_path = os.path.join(self.RENDER, subject, '%d_%d_%02d.png' % (vid, pid, 0))

        with open(os.path.join(self.POSE2D, subject, '%d_%d_%02d_keypoints.json' % (vid, pid, 0))) as json_file:
            data = json.load(json_file)['people'][0]
            keypoints = np.array(data['pose_keypoints_2d']).reshape(-1,3)

            flags = keypoints[:,2] > 0.5
            
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
        # make sure camera space matches with pixel coordinates [-loadSize//2, loadSize]
        scale_intrinsic = np.identity(4)
        scale_intrinsic[0, 0] = scale / ortho_ratio
        scale_intrinsic[1, 1] = -scale / ortho_ratio # due to discripancy between OpenGL and CV
        scale_intrinsic[2, 2] = scale / ortho_ratio

        im = cv2.imread(render_path, cv2.IMREAD_UNCHANGED)
        h, w = im.shape[:2]

        # transformation from pixe space to normalized space [-1, 1]
        ndc_intrinsic = np.identity(4)
        scale_im2ndc = 1.0 / float(w // 2)
        ndc_intrinsic[0, 0] = scale_im2ndc
        ndc_intrinsic[1, 1] = scale_im2ndc
        ndc_intrinsic[2, 2] = scale_im2ndc
        
        # transformation in normalized coordinates
        trans_intrinsic = np.identity(4)

        intrinsic = np.matmul(scale_intrinsic, ndc_intrinsic)

        trans_mat = np.identity(4)
        rect = self.crop_func(keypoints)

        im = crop_image(im, rect)

        scale = w / rect[2]
        trans_mat *= scale
        trans_mat[3,3] = 1.0
        trans_mat[0, 3] = -scale*(rect[0] + rect[2]//2 - w//2) * scale_im2ndc
        trans_mat[1, 3] = -scale*(rect[1] + rect[3]//2 - h//2) * scale_im2ndc
        
        im = im / 255.0
        im[:,:,:3] /= im[:,:,3:] + 1e-8
        im = (255.0 * im).astype(np.uint8)[:,:,[2,1,0,3]]

        intrinsic = np.matmul(trans_mat, intrinsic)
        
        render = Image.fromarray(im[:,:,:3]).convert('RGB')
        mask = Image.fromarray(im[:,:,3]).convert('L')
                
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        calib = torch.Tensor(projection_matrix).float()

        calib_world = torch.Tensor(np.matmul(projection_matrix,np.matmul(intrinsic, extrinsic))).float()
        extrinsic = torch.Tensor(extrinsic).float()

        render = self.to_tensor(render)
        mask = self.to_tensor_mask(mask)

        render = mask.expand_as(render) * render

        render_512 = torch.nn.Upsample(size=(512,512),mode='bilinear')(render[None])
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])

        return {
            'name': '%s_%d_%d_%02d' % (subject, vid, pid, 0),
            'img': render.unsqueeze(0),
            'img_512': render_512,
            'calib': calib.unsqueeze(0),
            'calib_world': calib_world.unsqueeze(0),            
            'b_min': B_MIN,
            'b_max': B_MAX,
        }

    def __getitem__(self, index):
        return self.get_item(index)