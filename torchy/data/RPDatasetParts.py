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

    # 1106: changed radius from 0.8 to 1.2
    center = mshoulder
    if len(ps) == 1:
        ps = np.stack(ps, 0)
        radius = int(1.2*np.max(np.sqrt(((ps - center[None,:])**2).reshape(-1,2).sum(1))))
    else:
        ps = []
        pts_id = [0, 2, 5]
        ratio = [0.4, 0.3, 0.3]
        for i in pts_id:
            if flag[i]:
                ps.append(pts[i,:2])
        ps = np.stack(ps, 0)
        radius = int(1.2*np.max(np.sqrt(((ps - center[None,:])**2).reshape(-1,2).sum(1)) / np.array(ratio)))

    center = center.astype(np.int)

    x1 = center[0] - radius
    x2 = center[0] + radius
    y1 = center[1] - radius
    y2 = center[1] + radius

    return (x1, y1, x2-x1, y2-y1)

# def upperbody_crop(pts):
#     mshoulder = pts[1,:2]
#     nflag = pts[0,2] > 0.2
#     lflag = pts[17,2] > 0.2
#     rflag = pts[18,2] > 0.2

#     rear = pts[18,:2]
#     lear = pts[17,:2]
#     top = pts[0,:2]
#     if not nflag and lflag and rflag:
#         top = 0.5 * (rear + lear)
#     elif lflag:
#         top = lear
#     elif rflag:
#         top = rear


#     center = mshoulder
#     radius = int(2.5*np.sqrt(((center - top)**2).sum(0)))
#     center = center.astype(np.int)

#     x1 = center[0] - radius
#     x2 = center[0] + radius
#     y1 = center[1] - radius
#     y2 = center[1] + radius

#     return (x1, y1, x2-x1, y2-y1)

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

    
class RPDatasetParts(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train', projection='orthogonal'):
        self.opt = opt
        self.phase = phase
        self.is_train = phase == 'train'
        self.projection_mode = projection

        self.root = self.opt.dataroot

        self.PARAM_WB = os.path.join(self.root, '../hf_human_big', 'PARAM')

        self.RENDER = os.path.join(self.root, 'RENDER')
        self.MASK = os.path.join(self.root, 'MASK')
        self.PARAM = os.path.join(self.root, 'PARAM')
        self.UV_RENDER = os.path.join(self.root, 'UV_RENDER')
        self.UV_MASK = os.path.join(self.root, 'UV_MASK')
        self.UV_NORMAL = os.path.join(self.root, 'UV_NORMAL')
        self.UV_POS = os.path.join(self.root, 'UV_POS')
        self.VOL = os.path.join(self.root, 'VOL')
        self.OBJ = os.path.join(self.root, 'GEO', 'OBJ')
        self.SAMPLE = os.path.join(self.root, 'GEO', 'SAMPLE')
        self.TSDF = os.path.join(self.root, 'GEO', 'TSDF')
        self.BG = os.path.join(self.root, 'BG')
        self.POSE2D = os.path.join(self.root, 'POSE2D', 'json')
        self.POSE3D = os.path.join(self.root, 'POSE')
        
        self.B_MIN = np.array([-128, -28, -128])
        self.B_MAX = np.array([128, 228, 128])
        # self.B_MIN = np.array([-120, -20, -64])
        # self.B_MAX = np.array([120, 220, 64])

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

        self.num_sample_surface = self.opt.num_sample_surface
        self.num_sample_inout = self.opt.num_sample_inout
        self.num_sample_color = self.opt.num_sample_color
        self.num_sample_normal = self.opt.num_sample_normal

        self.subjects = self.get_subjects()

        self.poses = loadPoses(self.POSE3D, self.subjects)
        
        if self.opt.crop_type == 'face':
            self.crop_func = face_crop
        elif self.opt.crop_type == 'upperbody':
            self.crop_func = upperbody_crop
        else:
            self.crop_func = fullbody_crop

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.to_tensor_mask = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor()
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
        return len(self.subjects) * len(self.yaw_list) * len(self.pitch_list)

    def get_render(self, sid, num_views, pid=0, view_id=None, random_sample=False):
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
        if view_id is None:
            view_id = random.choice(self.yaw_list)
        # views are sampled evenly unless random_sample is enabled
        view_ids = [self.yaw_list[(view_id + len(self.yaw_list) // num_views * offset) % len(self.yaw_list)]
                    for offset in range(num_views)]
        if random_sample:
            view_ids = np.random.choices(self.yaw_list, num_views)

        pitch = self.pitch_list[pid]

        pose3d = self.poses[subject]

        calib_list = []
        render_list = []
        mask_list = []
        extrinsic_list = []
        file_list = []
        for vid in view_ids:
            param_path = os.path.join(self.PARAM, subject, '%d_%d_%02d.npy' % (vid, pitch, 0))
            render_path = os.path.join(self.RENDER, subject, '%d_%d_%02d.png' % (vid, pitch, 0))

            with open(os.path.join(self.POSE2D, subject, '%d_%d_%02d_keypoints.json' % (vid, pitch, 0))) as json_file:
                data = json.load(json_file)['people'][0]
                keypoints = np.array(data['pose_keypoints_2d']).reshape(-1,3)

                flags = keypoints[:,2] > 0.5

                nflag = flags[0]
                mflag = flags[1]

                check_id = [2, 5, 15, 16, 17, 18]
                cnt = sum(flags[check_id])
                if self.opt.crop_type == 'face' and (not (nflag and cnt > 3)):
                    raise IOError('face should be front')
                if self.opt.crop_type == 'upperbody' and (not (mflag and nflag and cnt > 3)):
                    raise IOError('face should be front')
                if self.opt.crop_type == 'fullbody' and sum(flags) < 15:
                    raise IOError('sufficient keypoints are not detected')
                

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

            # for rendering with different resolution
            param_wb_path = os.path.join(self.PARAM_WB, subject, '%d_%d_%02d.npy' % (vid, pitch, 0))
            param_wb = np.load(param_wb_path, allow_pickle=True)

            # camera center world coordinate
            dp = np.matmul(R, (center - param_wb.item().get('center'))[:,None])[:,0]
            dp[0] *= -1.0
            s = scale / param_wb.item().get('scale')
            keypoints[:,:2] = s * (keypoints[:,:2] + param_wb.item().get('scale') * dp[None,:2] / ortho_ratio - 512) + 512

            trans_mat = np.identity(4)
            rect = self.crop_func(keypoints)

            im = crop_image(im, rect)

            scale = w / rect[2]
            trans_mat *= scale
            trans_mat[3,3] = 1.0
            trans_mat[0, 3] = -scale*(rect[0] + rect[2]//2 - w//2) * scale_im2ndc
            trans_mat[1, 3] = -scale*(rect[1] + rect[3]//2 - h//2) * scale_im2ndc
            
            if self.opt.random_body_chop and np.random.rand() > 0.5 and self.is_train:
                y_offset = random.randint(-int(0.1*im.shape[0]),0)
                if y_offset != 0:
                    im[y_offset:,:,3] = 0.0

            im = im / 255.0
            im[:,:,:3] /= im[:,:,3:] + 1e-8
            im = (255.0 * im).astype(np.uint8)[:,:,[2,1,0,3]]

            intrinsic = np.matmul(trans_mat, intrinsic)
            
            im = cv2.resize(im, (512, 512))

            render = Image.fromarray(im[:,:,:3]).convert('RGB')
            mask = Image.fromarray(im[:,:,3]).convert('L')
                
            tw, th = 512, 512
            dx, dy = 0, 0
            if self.is_train and self.num_views < 2:
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
                    mask = mask.resize((w, h), Image.BILINEAR)
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
                    pad_size = int(0.08 * tw)
                    render = ImageOps.expand(render, pad_size, fill=0)
                    mask = ImageOps.expand(mask, pad_size, fill=0)

                    w, h = render.size
                    dx = random.randint(-int(round((w-tw)/4.0)),
                                        int(round((w-tw)/4.0)))
                    dy = random.randint(-int(round((h-th)/4.0)),
                                        int(round((h-th)/4.0)))
                    trans_intrinsic = np.identity(4)
                    intrinsic[0, 3] += -dx / float(tw // 2)
                    intrinsic[1, 3] += -dy / float(th // 2)

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

            if not self.num_sample_color and self.is_train and self.num_views < 2:
                # image augmentation
                render = self.aug_trans(render)

                if self.opt.aug_blur > 0.00001:
                    blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                    render = render.filter(blur)

            # intrinsic = np.matmul(trans_intrinsic, np.matmul(ndc_intrinsic, scale_intrinsic))
            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
            extrinsic = torch.Tensor(extrinsic).float()

            render = self.to_tensor(render)

            mask = self.to_tensor_mask(mask)
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

    def get_sample(self, subject, calib, mask=None):
        return self.load_points_sample(subject, calib, mask, self.opt.sigma)
    
    def load_points_sample(self, subject, calib, mask, sigma, num_files=100):
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
            sigma = 20.0

        SAMPLE_DIR = os.path.join(self.SAMPLE, subject)

        rand_idx = np.random.randint(num_files)
        pts = np.load(os.path.join(SAMPLE_DIR, 'sdf%04d.npy' % rand_idx))

        ptsh = np.matmul(np.concatenate([pts[:,:3], np.ones((pts.shape[0],1))], 1), calib.T)[:, :3]
        inbb = (ptsh[:, 0] >= -1) & (ptsh[:, 0] <= 1) & (ptsh[:, 1] >= -1) & \
               (ptsh[:, 1] <= 1) & (ptsh[:, 2] >= -1) & (ptsh[:, 2] <= 1)
        pts = pts[inbb]

        prob = np.exp(-((pts[:,3]*20.0)**2)/(2.0*sigma*sigma))

        ptsh = np.matmul(np.concatenate([pts[:,:3], np.ones((pts.shape[0],1))], 1), calib.T)[:, :3]
        if mask is not None:
            x = (self.load_size * (0.5 * ptsh[:,0] + 0.5)).astype(np.int32).clip(0, self.load_size-1)
            y = (self.load_size * (0.5 * ptsh[:,1] + 0.5)).astype(np.int32).clip(0, self.load_size-1)
            idx = y * self.load_size + x
            inmask = mask.reshape(-1)[idx] > 0.0
            prob_mask = (1.0-self.opt.mask_ratio) * mask.reshape(-1)[idx] + self.opt.mask_ratio
            prob = prob_mask * prob
        idx = np.random.choice(np.arange(0,pts.shape[0]),self.opt.num_sample_inout,p=prob/prob.sum())
        pts = pts[idx]

        in_mask = (pts[:,3] <= 0)
        out_mask = np.logical_not(in_mask)
        dist = pts[:,3]
        pts = pts[:,:3]

        theta = 2.0 * math.pi * np.random.rand(pts.shape[0])
        phi = np.arccos(1 - 2 * np.random.rand(pts.shape[0]))
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        dir = np.stack([x,y,z],1)
        radius = np.abs(20.0*dist).clip(max=1.0)*np.random.rand(pts.shape[0])
        pts += radius[:,None] * dir
        in_pts = pts[in_mask]
        out_pts = pts[out_mask]

        # rand_pts = np.concatenate(
        #     [2.0 * np.random.rand(self.num_sample_inout, 3) - 1.0, np.ones((self.num_sample_inout, 1))],
        #     1)  # [-1,1]
        # x = (self.load_size * (0.5 * rand_pts[:,0] + 0.5)).astype(np.int32).clip(0, self.load_size-1)
        # y = (self.load_size * (0.5 * rand_pts[:,1] + 0.5)).astype(np.int32).clip(0, self.load_size-1)
        # idx = y * self.load_size + x
        # inmask = mask.reshape(-1)[idx] > 0.0
        # rand_pts = np.matmul(rand_pts, inv(calib).T)[:, :3]

        # in_max = in_pts.max(0) + 5.0
        # in_min = in_pts.min(0) - 5.0
        # bbox_in = np.logical_and(np.logical_and(inmask, rand_pts[:,2] < in_max[2]), rand_pts[:,2] > in_min[2])
        # rand_pts = rand_pts[np.logical_not(bbox_in)][:int(self.opt.uniform_ratio*self.num_sample_inout)]
        # out_pts = np.concatenate([out_pts, rand_pts], 0)

        samples = np.concatenate([in_pts, out_pts], 0)
        labels = np.concatenate([np.ones((in_pts.shape[0], 1)), np.zeros((out_pts.shape[0], 1))], 0)
        ratio = float(out_pts.shape[0])/float(samples.shape[0])

        if ratio > 0.99:
            raise IOError('invalid data sample')

        if samples.shape[0] != self.opt.num_sample_inout:# + int(self.opt.uniform_ratio * self.num_sample_inout):
            raise IOError('unable to sample sufficient number of points')

        samples = torch.Tensor(samples.T).float()
        labels = torch.Tensor(labels.T).float()

        return {
            'samples': samples,
            'labels': labels,
            'ratio': ratio
        }

    def get_normal_sampling(self, subject, calib):
        uv_pos_path = os.path.join(self.UV_POS, subject, '%02d.exr' % (0))
        uv_normal_path = os.path.join(self.UV_NORMAL, subject, '%02d.png' % (0))
        uv_mask_path = os.path.join(self.UV_MASK, subject, '%02d.png' % (0)) 

        mask = cv2.imread(uv_mask_path)
        mask = mask[:, :, 0] != 0

        uv_normal = cv2.imread(uv_normal_path)
        uv_normal = cv2.cvtColor(uv_normal, cv2.COLOR_BGR2RGB) / 255.0
        uv_normal = 2.0 * uv_normal - 1.0

        # position map
        uv_pos = cv2.imread(uv_pos_path, 2 | 4)[:, :, ::-1]

        # flatten these images to select only surface pixels
        mask = mask.reshape((-1))
        uv_normal = uv_normal.reshape((-1, 3))
        uv_pos = uv_pos.reshape((-1, 3))

        surface_points = uv_pos[mask].T
        surface_normals = uv_normal[mask].T

        ptsh = np.matmul(np.concatenate([surface_points.T, np.ones((surface_points.shape[1],1))], 1), calib.T)[:, :3]

        inbb = (ptsh[:, 0] >= -1) & (ptsh[:, 0] <= 1) & (ptsh[:, 1] >= -1) & \
               (ptsh[:, 1] <= 1) & (ptsh[:, 2] >= -1) & (ptsh[:, 2] <= 1)

        surface_points = surface_points[:,inbb]
        surface_normals = surface_normals[:,inbb]

        if self.num_sample_normal:
            sample_list = random.sample(range(0, surface_points.shape[1] - 1), self.num_sample_normal)
            surface_points = surface_points[:,sample_list]
            surface_normals = surface_normals[:,sample_list]
        
        normals = torch.Tensor(surface_normals).float()
        samples = torch.Tensor(surface_points).float()

        return {
            'samples_nml': samples,
            'labels_nml': normals
        }

    def get_point_sampling(self, subject, calibs, sample_data=None, num_files=100):
        if not self.is_train:
            random.seed(1991)
            np.random.seed(1991)

        SAMPLE_DIR = os.path.join(self.SAMPLE, self.opt.sampling_mode, subject)

        rand_idx = np.random.randint(num_files)
        npy_file = os.path.join(SAMPLE_DIR, '%05d.io.npy' % rand_idx)

        pts = np.load(npy_file, allow_pickle=True).item()

        sample_points = np.concatenate([pts['in'],pts['out']], 0)
        labels = np.concatenate([np.ones((pts['in'].shape[0],1)), np.zeros((pts['out'].shape[0], 1))], 0)

        ptsh = np.matmul(calibs, np.concatenate([sample_points.T, np.ones((1,sample_points.shape[0]))],0))[:3,:]
        inbb = (ptsh > -1) & (ptsh < 1)
        inbb = inbb[0] & inbb[1]

        sample_points_sub = sample_points[inbb]
        labels_sub = labels[inbb]

        if sample_points_sub.shape[0] == 0:
            raise IOError('no overlap in the projection.')

        if self.num_sample_surface:
            sample_list = random.choices(range(0, sample_points_sub.shape[0] - 1), k=self.num_sample_surface)
            sample_points_sub = sample_points_sub[sample_list]
            labels_sub = labels_sub[sample_list]
        
        labels = torch.Tensor(labels_sub.T).float()
        samples = torch.Tensor(sample_points_sub.T).float()

        if sample_data is not None:
            sample_data['samples'] = torch.cat([sample_data['samples'], samples],1)
            sample_data['labels'] = torch.cat([sample_data['labels'], labels],1)
            sample_data['ratio'] = 1.0 - sample_data['labels'].sum().item() / sample_data['labels'].size(1)
            return sample_data
        else:
            return {
                'samples': samples,
                'labels': labels,
                'ratio': 1 - labels.sum().item() /labels.size(1)
            }

    def get_color_sampling(self, subject, calib, vid, pid=0):
        view_id = self.yaw_list[vid]
        pitch = self.pitch_list[pid]

        uv_render_path = os.path.join(self.UV_RENDER, subject, '%d_%d_%02d.jpg' % (view_id, pitch, 0))
        uv_pos_path = os.path.join(self.UV_POS, subject, '%02d.exr' % (0))
        uv_normal_path = os.path.join(self.UV_NORMAL, subject, '%02d.png' % (0))
        uv_mask_path = os.path.join(self.UV_MASK, subject, '%02d.png' % (0)) 

        # segmentation mask for the uv render
        uv_mask = cv2.imread(uv_mask_path)
        uv_mask = uv_mask[:, :, 0] != 0
        # take the color of every pixel as the ground truth color at the point
        uv_render = cv2.imread(uv_render_path)
        uv_render = cv2.cvtColor(uv_render, cv2.COLOR_BGR2RGB) / 255.0
        # surface normal is used to perturb the points perpendicularly 
        uv_normal = cv2.imread(uv_normal_path)
        uv_normal = cv2.cvtColor(uv_normal, cv2.COLOR_BGR2RGB) / 255.0
        uv_normal = 2.0 * uv_normal - 1.0
        # position map
        uv_pos = cv2.imread(uv_pos_path, 2 | 4)[:, :, ::-1]

        # flatten these images to select only surface pixels
        uv_mask = uv_mask.reshape((-1))
        uv_render = uv_render.reshape((-1, 3))
        uv_normal = uv_normal.reshape((-1, 3))
        uv_pos = uv_pos.reshape((-1, 3))

        surface_points = uv_pos[uv_mask].T
        surface_normals = uv_normal[uv_mask].T
        surface_colors= uv_render[uv_mask].T

        ptsh = np.matmul(np.concatenate([surface_points.T, np.ones((surface_points.shape[1],1))], 1), calib.T)[:, :3]

        inbb = (ptsh[:, 0] >= -1) & (ptsh[:, 0] <= 1) & (ptsh[:, 1] >= -1) & \
               (ptsh[:, 1] <= 1) & (ptsh[:, 2] >= -1) & (ptsh[:, 2] <= 1)
        surface_points = surface_points[:,inbb]
        surface_normals = surface_normals[:,inbb]
        surface_colors = surface_colors[:,inbb]

        if self.num_sample_color:
            sample_list = random.sample(range(0, surface_points.shape[1] - 1), self.num_sample_color)
            surface_points = surface_points[:,sample_list]
            surface_normals = surface_normals[:,sample_list]
            surface_colors = surface_colors[:,sample_list]
        
        # points are perturbed perpendicularly to the surface 
        normal = torch.Tensor(surface_normals).float()
        offset = torch.normal(mean=torch.zeros((1, normal.size(1))), std=self.opt.sigma) * normal
        samples = torch.Tensor(surface_points).float() + offset

        # normalize color to [-1, 1]
        rgbs_color = 2.0 * torch.Tensor(surface_colors).float() - 1.0

        return {
            'color_samples': samples,
            'rgbs': rgbs_color
        }
    
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
            res.update(render_data)
            if not self.num_sample_color and self.opt.num_sample_inout:
                sample_data = self.get_sample(subject, render_data['calib'][0].numpy(), render_data['mask'][0].numpy()) 
                sample_data = self.get_point_sampling(subject, render_data['calib'][0].numpy(), sample_data)
                res.update(sample_data)
            elif not self.num_sample_color:
                sample_data = self.get_point_sampling(subject, render_data['calib'][0].numpy())
                res.update(sample_data)
            # p = sample_data['samples'].t().numpy()
            # calib = render_data['calib'][0].numpy()
            # mask = (255.0*(0.5*render_data['img'][0].permute(1,2,0).numpy()[:,:,::-1]+0.5)).astype(np.uint8)
            # p = np.matmul(np.concatenate([p, np.ones((p.shape[0],1))], 1), calib.T)[:, :3]
            # pts = mask.shape[0]*(0.5*p[sample_data['labels'].numpy().reshape(-1) == 1.0]+0.5)
            # for p in pts:
            #     mask = cv2.circle(mask, (int(p[0]),int(p[1])), 2, (0,255.0,0), -1)
            # # mask = cv2.putText(mask, render_data['files'][0], (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), lineType=cv2.LINE_AA) 
            # cv2.imwrite('tmp.png', mask)
            # exit()
            # cv2.waitKey(1000)
            
            
            if self.num_sample_normal:
                normal_data = self.get_normal_sampling(subject, render_data['calib'][0].numpy())
                res.update(normal_data)
            if self.num_sample_color:
                color_data = self.get_color_sampling(subject, render_data['calib'][0].numpy(), vid=vid)
                res.update(color_data)
            return res
        except Exception as e:
            for i in range(10):
                try:
                    return self.get_item(index=random.randint(0, self.__len__() - 1)) 
                except Exception as e:
                    continue

    def __getitem__(self, index):
        return self.get_item(index)

def test(is_train=True):

    max_yaw_angle = 10
    max_pitch_angle = 5
    interval_yaw = 2
    interval_pitch = 5
    yaw_size = max_yaw_angle
    intv_yaw = interval_yaw
    pitch_size = 2 * max_pitch_angle + 1
    intv_pitch = interval_pitch

    yaw_val = yaw_size - yaw_size // intv_yaw if is_train else yaw_size // intv_yaw
    pitch_val = pitch_size - pitch_size // intv_pitch - 1 if is_train else pitch_size // intv_pitch + 1
        
    for tmp in range(0, yaw_val * pitch_val):
        if is_train:
            vid = tmp % yaw_val
            pid = tmp // yaw_val
            vid = intv_yaw * (vid // (intv_yaw - 1)) + vid % (intv_yaw - 1) + 1
            pid = intv_pitch * (pid // (intv_pitch- 1)) + pid % (intv_pitch - 1) + 1 - max_pitch_angle
        else:
            vid = tmp % yaw_val
            pid = tmp // yaw_val
            vid = intv_yaw * vid
            pid = intv_pitch * pid - max_pitch_angle

if __name__ in '__main__':
    test(True)