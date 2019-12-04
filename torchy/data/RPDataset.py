import os
import sys
import random

import numpy as np 
from PIL import Image, ImageOps
from PIL.ImageFilter import GaussianBlur
import cv2
import trimesh

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def loadPoses(root, subjects):
    dic = {}
    for sub in subjects:
        dic[sub] = np.load(os.path.join(root, '%s_100k.npy' % sub))

    return dic

class RPDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train', projection='orthogonal'):
        self.opt = opt
        self.phase = phase
        self.is_train = phase == 'train'
        self.projection_mode = projection

        self.root = self.opt.dataroot
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
        self.POSE = os.path.join(self.root, 'POSE')
        
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

        self.num_sample_inout = self.opt.num_sample_inout
        self.num_sample_color = self.opt.num_sample_color
        self.num_sample_normal = self.opt.num_sample_normal

        self.subjects = self.get_subjects()

        self.poses = loadPoses(self.POSE, self.subjects)

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
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
        return len(self.subjects) * len(self.yaw_list) * len(self.pitch_list)

    def get_render(self, sid, pid=0, view_id=0):
        '''
        Return render data
        args:
            subject: subject name
            pitch: pitch angle (default: 0)
            view_id: the first view id. if None, select randomly
        return:
            'img': None, # [C, H, W] input images
            'calib': None, # [4, 4] calibration matrix
            'extrinsic': None, # [4, 4] extrinsic matrix
            'mask': None, # [1, H, W] segmentation masks
        '''
        subject = self.subjects[sid]

        vid = self.yaw_list[view_id]

        pitch = self.pitch_list[pid]

        pose3d = self.poses[subject]

        calib_list = []
        render_list = []
        mask_list = []
        extrinsic_list = []

        param_path = os.path.join(self.PARAM, subject, '%d_%d_%02d.npy' % (vid, pitch, 0))
        render_path = os.path.join(self.RENDER, subject, '%d_%d_%02d.png' % (vid, pitch, 0))

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

        # transformation from pixe space to normalized space [-1, 1]
        ndc_intrinsic = np.identity(4)
        ndc_intrinsic[0, 0] = 1.0 / float(self.opt.loadSize // 2)
        ndc_intrinsic[1, 1] = 1.0 / float(self.opt.loadSize // 2)
        ndc_intrinsic[2, 2] = 1.0 / float(self.opt.loadSize // 2)
        
        # transformation in normalized coordinates
        trans_intrinsic = np.identity(4)

        intrinsic = np.matmul(ndc_intrinsic, scale_intrinsic)
        calib = np.matmul(intrinsic, extrinsic)
        pose2d = np.matmul(calib, np.concatenate([pose3d, np.ones_like(pose3d[:,:1])], 1).T)

        im = cv2.imread(render_path, cv2.IMREAD_UNCHANGED) / 255.0
        im[:,:,:3] /= im[:,:,3:] + 1e-8
        im = (255.0 * im).astype(np.uint8)[:,:,[2,1,0,3]]
        if self.opt.random_body_chop and np.random.rand() > 0.5 and self.is_train:
            y_offset = random.randint(-100,0)
            if y_offset != 0:
                im[y_offset:,:,3] = 0.0
        render = Image.fromarray(im[:,:,:3]).convert('RGB')
        mask = Image.fromarray(im[:,:,3]).convert('L')

        if self.opt.random_bg and len(self.bg_list) != 0:
            if self.is_train:
                bg_path = os.path.join(self.BG, random.choice(self.bg_list))
            else:
                uid = sid * len(self.yaw_list) * (view_id * len(self.pitch_list) + pitch) 
                bg_path = os.path.join(self.BG, self.bg_list[uid % len(self.bg_list)])
            bg = Image.open(bg_path).convert('RGB')

            render = Image.composite(render, bg, mask)
            
        if self.phase == 'train':
            # pad images
            pad_size = int(0.1 * self.load_size)
            render = ImageOps.expand(render, pad_size, fill=0)
            mask = ImageOps.expand(mask, pad_size, fill=0)

            w, h = render.size
            tw, th = self.load_size, self.load_size

            # random flip
            if self.opt.random_flip and np.random.rand() > 0.5 and self.is_train:
                scale_intrinsic[0, 0] *= -1
                render = transforms.RandomHorizontalFlip(p=1.0)(render)
                mask = transforms.RandomHorizontalFlip(p=1.0)(mask)

            # random scale 
            if self.opt.random_scale:
                rand_scale = random.uniform(0.8, 1.4)
                w = int(rand_scale * w)
                h = int(rand_scale * h)
                render = render.resize((w, h), Image.BILINEAR)
                mask = mask.resize((w, h), Image.NEAREST)
                scale_intrinsic *= rand_scale
                scale_intrinsic[3, 3] = 1
            
            if self.opt.random_rotate:
                rotate_degree = (random.random()-0.5) * 2 * 10.0
                theta = -(rotate_degree/180.) * np.pi
                rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                    [np.sin(theta),  np.cos(theta)]])
                center = np.array([[render.size[0]/2,render.size[1]/2]])
                rot_matrix = np.identity(4)
                rot_matrix[:2,:2] = rotMatrix

                scale_intrinsic = np.matmul(rot_matrix, scale_intrinsic)
                render = render.rotate(rotate_degree, Image.BILINEAR)
                mask = mask.rotate(rotate_degree, Image.BILINEAR)

            # random translate in the pixel space
            if self.opt.random_trans:
                padding = int(1.1 * tw - w)
                if padding > 0:
                    render = ImageOps.expand(render, border=(padding,padding), fill=0)
                    mask = ImageOps.expand(mask, border=(padding, padding), fill=0)
                w, h = render.size
                dx = random.randint(-int(round((w-tw)/8.0)),
                                    int(round((w-tw)/8.0)))
                dy = random.randint(-int(round((h-th)/8.0)),
                                    int(round((h-th)/8.0)))
                trans_intrinsic[0, 3] = (-dx) / float(self.opt.loadSize // 2)
                trans_intrinsic[1, 3] = (-dy) / float(self.opt.loadSize // 2)
            else:
                dx = 0
                dy = 0

            x1 = int(round((w - tw) / 2.)) + dx
            y1 = int(round((h - th) / 2.)) + dy

            render = render.crop((x1, y1, x1 + tw, y1 + th))
            mask = mask.crop((x1, y1, x1 + tw, y1 + th))

            # image augmentation
            render = self.aug_trans(render)

            if self.opt.aug_blur > 0.00001:
                blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                render = render.filter(blur)

        intrinsic = np.matmul(trans_intrinsic, np.matmul(ndc_intrinsic, scale_intrinsic))
        calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
        extrinsic = torch.Tensor(extrinsic).float()

        render = self.to_tensor(render)

        mask = transforms.Resize(self.load_size)(mask)
        mask = transforms.ToTensor()(mask).float()

        if not self.opt.random_bg or len(self.bg_list) == 0:                
            render = mask.expand_as(render) * render

        return {
            'img': render,
            'calib': calib,
            'extrinsic': extrinsic,
            'mask': mask
        }

    def select_sampling_method(self, subject, calib):
        mode = self.opt.sampling_mode if self.is_train else 'uniform_10k'
        return self.load_points_sample(subject, mode)
    
    def load_points_sample(self, subject, mode, num_samples=0, num_files=100):
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
        SAMPLE_DIR = os.path.join(self.SAMPLE, mode, subject)

        rand_idx = np.random.randint(num_files)
        inside_file = os.path.join(SAMPLE_DIR, '%05d.in.xyz' % rand_idx)
        rand_idx = np.random.randint(num_files)
        outside_file = os.path.join(SAMPLE_DIR, '%05d.ou.xyz' % rand_idx)

        in_pts = np.load(inside_file)
        out_pts = np.load(outside_file)

        if num_samples <= 0:
            num_samples = self.num_sample_inout
        rand_indices = np.random.randint(len(in_pts), size=num_samples // 2)
        in_pts = in_pts[rand_indices]

        rand_indices = np.random.randint(len(out_pts), size=num_samples // 2)
        out_pts = out_pts[rand_indices]

        samples = np.concatenate([in_pts, out_pts], 0)
        labels = np.concatenate([np.ones((in_pts.shape[0], 1)), np.zeros((out_pts.shape[0], 1))], 0)    

        samples = torch.Tensor(samples.T).float()
        labels = torch.Tensor(labels.T).float()

        return {
            'samples': samples,
            'labels': labels,
        }

    def get_normal_sampling(self, subject):
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

    def get_color_sampling(self, subject, vid, pitch=0):
        uv_render_path = os.path.join(self.UV_RENDER, subject, '%d_%d_%02d.png' % (vid, pitch, 0))
        uv_mask_path = os.path.join(self.UV_MASK, subject, '%d_%d_%02d.png' % (vid, pitch, 0)) 
        uv_pos_path = os.path.join(self.UV_POS, subject, '%02d.exr' % (0))
        uv_normal_path = os.path.join(self.UV_NORMAL, subject, '%02d.png' % (0))

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

        if self.num_sample_color:
            sample_list = random.sample(range(0, surface_points.shape[0] - 1), self.num_sample_color)
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
            render_data = self.get_render(sid, view_id=vid,
                                        pid=pid)
            sample_data = self.select_sampling_method(subject, render_data['calib'].numpy(), render_data['mask'].numpy())

            # for debug only 
            # p = sample_data['samples'].t().numpy()
            # calib = render_data['calib'][0].numpy()
            # mask = 255.0*np.stack(3*[render_data['mask'][0,0].numpy()],2)
            # p = np.matmul(np.concatenate([p, np.ones((p.shape[0],1))], 1), calib.T)[:, :3]
            # pts = 512*(0.5*p[sample_data['labels'].numpy().reshape(-1) == 1.0]+0.5)
            # print(pts.shape)
            # for p in pts:
            #     mask = cv2.circle(mask, (int(p[0]),int(p[1])), 2, (0,255,0), -1)
            # print(mask.shape)
            # cv2.imwrite('tmp.png', mask)
            # exit()

            res.update(render_data)
            res.update(sample_data)
            if self.num_sample_normal:
                normal_data = self.get_normal_sampling(subject)
                res.update(normal_data)
            if self.num_sample_color:
                color_data = self.get_color_sampling(subject, view_id=vid)
                res.upate(color_data)
            return res
        except Exception as e:
            print(e)
            return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)