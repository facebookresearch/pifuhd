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


class FRLFaceDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train', projection='orthogonal'):
        self.opt = opt
        self.phase = phase
        self.is_train = phase == 'train'
        self.projection_mode = projection

        self.root = self.opt.dataroot

        self.exp_list = ['neutral_no_hair_net']

        self.RENDER = os.path.join(self.root, 'RENDER')
        self.PARAM = os.path.join(self.root, 'PARAM')
        self.SAMPLE = os.path.join(self.root, 'GEO', 'SAMPLE')
        self.OBJ = os.path.join(self.root, 'GEO', 'MESH')
        self.BG = os.path.join(self.root, 'BG')
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
            if tmp[0] == 'pitch' and phase in tmp[1]:
                self.pitch_list = [int(i) for i in tmp[2].strip('[]').split(' ')]
            if tmp[0] == 'yaw' and phase in tmp[1]:
                self.yaw_list = [int(i) for i in tmp[2].strip('[]').split(' ')]
        
        self.bg_list = [f for f in os.listdir(self.BG) if '.jpg' in f]

        self.load_size = self.opt.loadSize

        self.num_views = self.opt.num_views

        self.num_sample_inout = self.opt.num_sample_inout
        self.num_sample_color = self.opt.num_sample_color
        self.num_sample_normal = self.opt.num_sample_normal

        self.subjects = self.get_subjects()
        
        self.crop_func = face_crop

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, 
            saturation=opt.aug_sat, hue=opt.aug_hue),
            transforms.RandomGrayscale(opt.aug_gry),
        ])
    
    def get_subjects(self):
        all_subjects = set(os.listdir(self.RENDER))
        if os.path.exists(os.path.join(self.root, 'val.txt')):
            arr = np.atleast_1d(np.loadtxt(os.path.join(self.root, 'val.txt'), dtype=str))
            var_subjects = set(arr)
        else:
            var_subjects = set([])

        if self.phase == 'all':
            return sorted(list(all_subjects))
        elif self.is_train:
            return sorted(list(all_subjects - var_subjects))
        elif len(var_subjects) != 0:
            return sorted(list(var_subjects))
        else:
            return sorted(list(all_subjects))

    def __len__(self):
        return len(self.subjects) * len(self.exp_list) * len(self.yaw_list)

    def get_render(self, sid, eid, num_views, pid=0, view_id=None, random_sample=False):
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
        subject = self.subjects[sid]
        exp = self.exp_list[eid]
        if view_id is None:
            view_id = random.choice(self.yaw_list)
        # views are sampled evenly unless random_sample is enabled
        view_ids = [self.yaw_list[(view_id + len(self.yaw_list) // num_views * offset) % len(self.yaw_list)]
                    for offset in range(num_views)]
        if random_sample:
            view_ids = np.random.choices(self.yaw_list, num_views)

        pitch = self.pitch_list[pid]

        calib_list = []
        render_list = []
        mask_list = []
        extrinsic_list = []
        file_list = []
        for vid in view_ids:
            param_path = os.path.join(self.PARAM, subject, exp, '%d_%d.npy' % (vid, pitch))
            render_path = os.path.join(self.RENDER, subject, exp, '%d_%d.png' % (vid, pitch))

            with open(os.path.join(self.POSE2D, subject, exp, '%d_%d_keypoints.json' % (vid, pitch))) as json_file:
                data = json.load(json_file)['people'][0]
                keypoints = np.array(data['pose_keypoints_2d']).reshape(-1,3)

            # load calibration data
            param = np.load(param_path, allow_pickle=True)
            # pixel unit / world unit
            ortho_ratio = param.item().get('ortho_ratio')
            # world unit / model unit
            scale = 1.0 / param.item().get('scale')
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
            
            intrinsic = np.matmul(trans_mat, intrinsic)
            im = cv2.resize(im, (self.load_size, self.load_size))
        
            im = im / 255.0
            im[:,:,:3] /= im[:,:,3:] + 1e-8
            im = (255.0 * im).astype(np.uint8)[:,:,[2,1,0,3]]

            render = Image.fromarray(im[:,:,:3]).convert('RGB')
            mask = Image.fromarray(im[:,:,3]).convert('L')
                
            tw, th = self.load_size, self.load_size
            dx, dy = 0, 0
            if self.phase == 'train' and self.num_views < 2:
                # pad images
                w, h = render.size

                # random flip
                if self.opt.random_flip and np.random.rand() > 0.5 and self.is_train:
                    intrinsic[0, :] *= -1
                    render = transforms.RandomHorizontalFlip(p=1.0)(render)
                    mask = transforms.RandomHorizontalFlip(p=1.0)(mask)

                # random scale 
                if self.opt.random_scale:
                    rand_scale = random.uniform(0.95, 1.05)
                    w = int(rand_scale * w)
                    h = int(rand_scale * h)
                    render = render.resize((w, h), Image.BILINEAR)
                    mask = mask.resize((w, h), Image.NEAREST)
                    intrinsic[:3,:] *= rand_scale
                
                if self.opt.random_rotate:
                    rotate_degree = (random.random()-0.5) * 10.0
                    theta = -(rotate_degree/180.) * np.pi
                    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                        [np.sin(theta),  np.cos(theta)]])
                    center = np.array([[render.size[0]/2,render.size[1]/2]])
                    rot_matrix = np.identity(4)
                    rot_matrix[:2,:2] = rotMatrix

                    intrinsic = np.matmul(rot_matrix, intrinsic)
                    render = render.rotate(rotate_degree, Image.BILINEAR)
                    mask = mask.rotate(rotate_degree, Image.BILINEAR)

                # # random translate in the pixel space
                if self.opt.random_trans:
                    pad_size = int(0.08 * self.load_size)
                    render = ImageOps.expand(render, pad_size, fill=0)
                    mask = ImageOps.expand(mask, pad_size, fill=0)

                    w, h = render.size
                    dx = random.randint(-int(round((w-tw)/4.0)),
                                        int(round((w-tw)/4.0)))
                    dy = random.randint(-int(round((h-th)/4.0)),
                                        int(round((h-th)/4.0)))
                    trans_intrinsic = np.identity(4)
                    intrinsic[0, 3] += -dx / float(self.opt.loadSize // 2)
                    intrinsic[1, 3] += -dy / float(self.opt.loadSize // 2)

            w, h = render.size    
            x1 = int(round((w - tw) / 2.)) + dx
            y1 = int(round((h - th) / 2.)) + dy

            render = render.crop((x1, y1, x1 + tw, y1 + th))
            mask = mask.crop((x1, y1, x1 + tw, y1 + th))

            if self.opt.random_bg and len(self.bg_list) != 0 and self.is_train:                
                bg_path = os.path.join(self.BG, random.choice(self.bg_list))
            else:
                uid = sid * len(self.yaw_list) * (view_id * len(self.pitch_list) + pitch) 
                bg_path = os.path.join(self.BG, self.bg_list[uid % len(self.bg_list)])
            bg = Image.open(bg_path).convert('RGB')

            render = Image.composite(render, bg, mask)

            if self.phase == 'train' and self.num_views < 2:
                # image augmentation
                render = self.aug_trans(render)

                if self.opt.aug_blur > 0.00001:
                    blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                    render = render.filter(blur)

            # intrinsic = np.matmul(trans_intrinsic, np.matmul(ndc_intrinsic, scale_intrinsic))
            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
            extrinsic = torch.Tensor(extrinsic).float()

            render = self.to_tensor(render)

            mask = transforms.ToTensor()(mask).float()
            mask_list.append(mask)

            if not self.opt.random_bg or len(self.bg_list) == 0:                
                render = mask.expand_as(render) * render

            # img = render.permute(1,2,0).numpy()[:,:,::-1]
            # cv2.imshow('image', (0.5*img + 0.5))
            # cv2.waitKey(1)

            render_list.append(render)
            calib_list.append(calib)
            extrinsic_list.append(extrinsic)
            file_list.append(render_path)

        return {
            'img': torch.stack(render_list, dim=0),
            'calib': torch.stack(calib_list, dim=0),
            'extrinsic': torch.stack(extrinsic_list, dim=0),
            'mask': torch.stack(mask_list, dim=0),
            'files': file_list
        }

    def get_sample(self, subject, exp, calib):
        return self.load_points_sample(subject, exp, calib)
    
    def load_points_sample(self, subject, exp, calib, num_files=1):
        '''
        load points from precomputed numpy array
        inside/outside points are stored with %05d.in.xyz and %05d.ou.xyz 
        args:
            subject: subject name 'rp_xxxx_xxx'
            mode: sampling mode
        return:
            'samples': [3, N]
            'labels': [1, N]
        '''
        if not self.is_train:
            random.seed(1991)
            np.random.seed(1991)
        SAMPLE_DIR = os.path.join(self.SAMPLE, subject, exp)

        rand_idx = np.random.randint(num_files)
        pts = np.load(os.path.join(SAMPLE_DIR, 'occ%04d.npy' % rand_idx))

        ptsh = np.matmul(np.concatenate([pts[:,:3], np.ones((pts.shape[0],1))], 1), calib.T)[:, :3]
        inbb = (ptsh[:, 0] >= -1) & (ptsh[:, 0] <= 1) & (ptsh[:, 1] >= -1) & \
               (ptsh[:, 1] <= 1) & (ptsh[:, 2] >= -1) & (ptsh[:, 2] <= 1)
        pts = pts[inbb]

        idx = np.random.randint(0, pts.shape[0], size=(self.opt.num_sample_inout+self.opt.num_sample_surface,))

        pts = pts[idx]

        in_mask = (pts[:,3] >= 0.5)
        out_mask = np.logical_not(in_mask)

        pts = pts[:,:3]
        in_pts = pts[in_mask]
        out_pts = pts[out_mask]

        samples = np.concatenate([in_pts, out_pts], 0)
        labels = np.concatenate([np.ones((in_pts.shape[0], 1)), np.zeros((out_pts.shape[0], 1))], 0)
        ratio = float(out_pts.shape[0])/float(samples.shape[0])

        if ratio > 0.99:
            raise IOError('invalid data sample')
        if samples.shape[0] != self.opt.num_sample_inout + self.opt.num_sample_surface:
            raise IOError('unable to sample sufficient number of points')

        samples = torch.Tensor(samples.T).float()
        labels = torch.Tensor(labels.T).float()

        return {
            'samples': samples,
            'labels': labels,
            'ratio': ratio
        }
    
    def get_item(self, index):
        # in case of IO error, use random sampling instead
        subject = ''
        # try:
        sid = index % len(self.subjects)
        tmp = index // len(self.subjects)

        vid = tmp % len(self.yaw_list)
        pid = 0 # tmp // len(self.yaw_list)

        eid = 0 # TODO

        # name of the subjects 'rp_xxxx_xxx'
        subject = self.subjects[sid]
        expression = self.exp_list[eid]
        res = {
            'name': subject,
            'mesh_path': os.path.join(self.OBJ, subject, expression + '.ply'),
            'sid': sid,
            'vid': vid,
            'pid': pid,
            'b_min': self.B_MIN,
            'b_max': self.B_MAX,
        }
        render_data = self.get_render(sid, eid, num_views=self.num_views, view_id=vid,
                                        pid=pid, random_sample=self.opt.random_multiview)
        sample_data = self.get_sample(subject, expression, render_data['calib'][0].numpy())        
        # p = sample_data['samples'].t().numpy()
        # calib = render_data['calib'][0].numpy()
        # mask = (255.0*(0.5*render_data['img'][0].permute(1,2,0).numpy()[:,:,::-1]+0.5)).astype(np.uint8)
        # # mask = 255.0*np.stack(3*[render_data['mask'][0,0].numpy()],2)
        # p = np.matmul(np.concatenate([p, np.ones((p.shape[0],1))], 1), calib.T)[:, :3]
        # pts = 512*(0.5*p[sample_data['labels'].numpy().reshape(-1) == 1.0]+0.5)
        # for p in pts:
        #     mask = cv2.circle(mask, (int(p[0]),int(p[1])), 2, (0,255.0,0), -1)
        # mask = cv2.putText(mask, render_data['files'][0], (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), lineType=cv2.LINE_AA) 
        # cv2.imshow('tmp.img', mask)
        # # exit()
        # cv2.waitKey(1000)
        res.update(render_data)
        res.update(sample_data)
        return res
        # except Exception as e:
        #     for i in range(10):
        #         try:
        #             return self.get_item(index=random.randint(0, self.__len__() - 1)) 
        #         except Exception as e:
        #             continue

    def __getitem__(self, index):
        return self.get_item(index)
