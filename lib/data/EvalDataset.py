# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
import random

import numpy as np 
from PIL import Image, ImageOps
from PIL.ImageFilter import GaussianBlur
import cv2
import torch
import json

from torch.utils.data import Dataset
import torchvision.transforms as transforms

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

class EvalDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, projection='orthogonal'):
        self.opt = opt
        self.projection_mode = projection

        self.root = self.opt.dataroot
        self.img_files = sorted([os.path.join(self.root,f) for f in os.listdir(self.root) if f.split('.')[-1] in ['png', 'jpeg', 'jpg', 'PNG', 'JPG', 'JPEG'] and os.path.exists(os.path.join(self.root,f.replace('.%s' % (f.split('.')[-1]), '_rect.txt')))])
        self.IMG = os.path.join(self.root)

        self.phase = 'val'
        self.load_size = self.opt.loadSize

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # only used in case of multi-person processing
        self.person_id = 0

    def __len__(self):
        return len(self.img_files)

    def get_n_person(self, index):
        rect_path = self.img_files[index].replace('.%s' % (self.img_files[index].split('.')[-1]), '_rect.txt')
        rects = np.loadtxt(rect_path, dtype=np.int32)

        return rects.shape[0] if len(rects.shape) == 2 else 1

    def get_item(self, index):
        img_path = self.img_files[index]
        rect_path = self.img_files[index].replace('.%s' % (self.img_files[index].split('.')[-1]), '_rect.txt')
        # Name
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if im.shape[2] == 4:
            im = im / 255.0
            im[:,:,:3] /= im[:,:,3:] + 1e-8
            im = im[:,:,3:] * im[:,:,:3] + 0.5 * (1.0 - im[:,:,3:])
            im = (255.0 * im).astype(np.uint8)
        h, w = im.shape[:2]
        
        intrinsic = np.identity(4)

        trans_mat = np.identity(4)

        rects = np.loadtxt(rect_path, dtype=np.int32)
        if len(rects.shape) == 1:
            rects = rects[None]
        pid = min(rects.shape[0]-1, self.person_id)

        rect = rects[pid].tolist()
        im = crop_image(im, rect)

        scale_im2ndc = 1.0 / float(w // 2)
        scale = w / rect[2]
        trans_mat *= scale
        trans_mat[3,3] = 1.0
        trans_mat[0, 3] = -scale*(rect[0] + rect[2]//2 - w//2) * scale_im2ndc
        trans_mat[1, 3] = scale*(rect[1] + rect[3]//2 - h//2) * scale_im2ndc
        
        intrinsic = np.matmul(trans_mat, intrinsic)
        im_512 = cv2.resize(im, (512, 512))
        im = cv2.resize(im, (self.load_size, self.load_size))

        image_512 = Image.fromarray(im_512[:,:,::-1]).convert('RGB')
        image = Image.fromarray(im[:,:,::-1]).convert('RGB')
        
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        calib = torch.Tensor(projection_matrix).float()

        calib_world = torch.Tensor(intrinsic).float()

        # image
        image_512 = self.to_tensor(image_512)
        image = self.to_tensor(image)
        return {
            'name': img_name,
            'img': image.unsqueeze(0),
            'img_512': image_512.unsqueeze(0),
            'calib': calib.unsqueeze(0),
            'calib_world': calib_world.unsqueeze(0),
            'b_min': B_MIN,
            'b_max': B_MAX,
        }

    def __getitem__(self, index):
        return self.get_item(index)
