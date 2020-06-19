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

def face_crop(pts):
    flag = pts[:,2] > 0.2

    mshoulder = pts[1,:2]
    rear = pts[18,:2]
    lear = pts[17,:2]
    nose = pts[0,:2]

    center = np.copy(mshoulder)
    center[1] = min(nose[1] if flag[0] else 1e8, lear[1] if flag[17] else 1e8, rear[1] if flag[18] else 1e8)

    ps = []
    pts_id = [0, 15, 16, 17, 18]
    cnt = 0
    for i in pts_id:
        if flag[i]:
            ps.append(pts[i,:2])
            if i in [17, 18]:
                cnt += 1

    ps = np.stack(ps, 0)
    if ps.shape[0] <= 1:
        raise IOError('key points are not properly set')
    if ps.shape[0] <= 3 and cnt != 2:
        center = ps[-1]
    else:
        center = ps.mean(0)
    radius = int(1.4*np.max(np.sqrt(((ps - center[None,:])**2).reshape(-1,2).sum(0))))


    # radius = np.max(np.sqrt(((center[None] - np.stack([]))**2).sum(0))
    # radius = int(1.0*abs(center[1] - mshoulder[1]))
    center = center.astype(np.int)

    x1 = center[0] - radius
    x2 = center[0] + radius
    y1 = center[1] - radius
    y2 = center[1] + radius

    return (x1, y1, x2-x1, y2-y1)

def upperbody_crop(pts):
    flag = pts[:,2] > 0.2

    mshoulder = pts[1,:2]
    ps = []
    pts_id = [8]
    for i in pts_id:
        if flag[i]:
            ps.append(pts[i,:2])

    center = mshoulder
    if len(ps) == 1:
        ps = np.stack(ps, 0)
        radius = int(0.8*np.max(np.sqrt(((ps - center[None,:])**2).reshape(-1,2).sum(1))))
    else:
        ps = []
        pts_id = [0, 2, 5]
        ratio = [0.4, 0.3, 0.3]
        for i in pts_id:
            if flag[i]:
                ps.append(pts[i,:2])
        ps = np.stack(ps, 0)
        radius = int(0.8*np.max(np.sqrt(((ps - center[None,:])**2).reshape(-1,2).sum(1)) / np.array(ratio)))

    center = center.astype(np.int)

    x1 = center[0] - radius
    x2 = center[0] + radius
    y1 = center[1] - radius
    y2 = center[1] + radius

    return (x1, y1, x2-x1, y2-y1)

def fullbody_crop(pts):
    flags = pts[:,2] > 0.5      #openpose
    # flags = pts[:,2] > 0.2  #detectron
    check_id = [11,19,21,22]
    cnt = sum(flags[check_id])

    if cnt == 0:
        center = pts[8,:2].astype(np.int)
        pts = pts[pts[:,2] > 0.5][:,:2]
        radius = int(1.45*np.sqrt(((center[None,:] - pts)**2).sum(1)).max(0))
        center[1] += int(0.05*radius)
    else:
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


class EvalWPoseDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, projection='orthogonal'):
        self.opt = opt
        self.projection_mode = projection

        self.root = self.opt.dataroot
        self.img_files = sorted([os.path.join(self.root,f) for f in os.listdir(self.root) if f.split('.')[-1] in ['png', 'jpeg', 'jpg', 'PNG', 'JPG', 'JPEG'] and os.path.exists(os.path.join(self.root,f.replace('.%s' % (f.split('.')[-1]), '_keypoints.json')))])
        self.IMG = os.path.join(self.root)

        self.phase = 'val'
        self.load_size = self.opt.loadSize

        if self.opt.crop_type == 'face':
            self.crop_func = face_crop
        elif self.opt.crop_type == 'upperbody':
            self.crop_func = upperbody_crop
        else:
            self.crop_func = fullbody_crop

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
        joint_path = self.img_files[index].replace('.%s' % (self.img_files[index].split('.')[-1]), '_keypoints.json')
        # Calib
        with open(joint_path) as json_file:
            data = json.load(json_file)
            return len(data['people'])            

    def get_item(self, index):
        img_path = self.img_files[index]
        joint_path = self.img_files[index].replace('.%s' % (self.img_files[index].split('.')[-1]), '_keypoints.json')
        # Name
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        # Calib
        with open(joint_path) as json_file:
            data = json.load(json_file)
            if len(data['people']) == 0:
                raise IOError('non human found!!')
            
            # if True, the person with the largest height will be chosen. 
            # set to False for multi-person processing
            if True:
                selected_data = data['people'][0]
                height = 0
                if len(data['people']) != 1:
                    for i in range(len(data['people'])):
                        tmp = data['people'][i]
                        keypoints = np.array(tmp['pose_keypoints_2d']).reshape(-1,3)

                        flags = keypoints[:,2] > 0.5 #openpose
                        # flags = keypoints[:,2] > 0.2  #detectron
                        if sum(flags) == 0:
                            continue
                        bbox = keypoints[flags]
                        bbox_max = bbox.max(0)
                        bbox_min = bbox.min(0)

                        if height < bbox_max[1] - bbox_min[1]:
                            height = bbox_max[1] - bbox_min[1]
                            selected_data = tmp
            else:
                pid = min(len(data['people'])-1, self.person_id)
                selected_data = data['people'][pid]

            keypoints = np.array(selected_data['pose_keypoints_2d']).reshape(-1,3)

            flags = keypoints[:,2] > 0.5   #openpose
            # flags = keypoints[:,2] > 0.2    #detectron

            nflag = flags[0]
            mflag = flags[1]

            check_id = [2, 5, 15, 16, 17, 18]
            cnt = sum(flags[check_id])
            if self.opt.crop_type == 'face' and (not (nflag and cnt > 3)):
                print('Waring: face should not be backfacing.')
            if self.opt.crop_type == 'upperbody' and (not (mflag and nflag and cnt > 3)):
                print('Waring: upperbody should not be backfacing.')
            if self.opt.crop_type == 'fullbody' and sum(flags) < 15:
                print('Waring: not sufficient keypoints.')

        im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if im.shape[2] == 4:
            im = im / 255.0
            im[:,:,:3] /= im[:,:,3:] + 1e-8
            im = im[:,:,3:] * im[:,:,:3] + 0.5 * (1.0 - im[:,:,3:])
            im = (255.0 * im).astype(np.uint8)
        h, w = im.shape[:2]
        
        intrinsic = np.identity(4)

        trans_mat = np.identity(4)
        rect = self.crop_func(keypoints)

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
