import os
import random

import numpy as np 
from PIL import Image, ImageOps
from PIL.ImageFilter import GaussianBlur
import cv2
import trimesh

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

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

        # where does this come from??
        self.B_MIN = np.array([-120, -20, -64])
        self.B_MAX = np.array([120, 220, 64])

        self.load_size = self.opt.loadSize

        self.num_views = self.opt.num_views

        self.num_sample_inout = self.opt.num_sample_inout
        self.num_sample_color = self.opt.num_sample_color

        self.max_yaw_angle = 360
        self.max_pitch_angle = self.opt.max_pitch # that's for positive and negative
        self.mean_pitch = self.opt.mean_pitch

        self.subjects = self.get_subjects()

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
            var_subjects = set(np.loadtxt(os.path.join(self.root, 'val.txt'), dtype=str))
        else:
            var_subjects = set([])

        if self.is_train:
            return sorted(list(all_subjects - var_subjects))
        elif len(var_subjects) != 0:
            return sorted(list(var_subjects))
        else:
            return sorted(list(all_subjects))

    def __len__(self):
        pitch_size = 2 * self.max_pitch_angle + 1
        pitch_val = pitch_size - 1 if self.is_train else 1
        yaw_val = self.max_yaw_angle - 1 if self.is_train else self.max_yaw_angle
        return len(self.subjects) * yaw_val * pitch_val 

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
            view_id = np.random.randint(self.max_yaw_angle)
        # views are sampled evenly unless random_sample is enabled
        view_ids = [(view_id + self.max_yaw_angle // num_views * offset) % self.max_yaw_angle
                    for offset in range(num_views)]
        if random_sample:
            view_ids = np.random.choices(self.max_yaw_angle, num_views, replace=False)

        calib_list = []
        render_list = []
        mask_list = []
        extrinsic_list = []

        for vid in view_ids:
            param_path = os.path.join(self.PARAM, subject, '%d_%d_%02d.npy' % (vid, pitch, 0))
            render_path = os.path.join(self.RENDER, subject, '%d_%d_%02d.jpg' % (vid, pitch, 0))
            mask_path = os.path.join(self.MASK, subject, '%d_%d_%02d.png' % (vid, pitch, 0)) 

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

            mask = Image.open(mask_path).convert('L')
            render = Image.open(render_path).convert('RGB')

            if self.phase == 'train' and self.num_views < 2:
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
                    rand_scale = random.uniform(0.9, 1.1)
                    w = int(rand_scale * w)
                    h = int(rand_scale * h)
                    render = render.resize((w, h), Image.BILINEAR)
                    mask = mask.resize((w, h), Image.NEAREST)
                    scale_intrinsic *= rand_scale
                    scale_intrinsic[3, 3] = 1

                # random translate in the pixel space
                if self.opt.random_trans:
                    dx = random.randint(-int(round((w-tw)/10.0)),
                                        int(round((w-tw)/10.0)))
                    dy = random.randint(-int(round((h-th)/10.0)),
                                        int(round((h-th)/10.0)))
                else:
                    dx = 0
                    dy = 0

                trans_intrinsic[0, 3] = -dy / float(self.opt.loadSize // 2)
                trans_intrinsic[1, 3] = -dy / float(self.opt.loadSize // 2)

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

    def get_color_sampling(self, subject, view_id, pitch=0):
        uv_render_path = os.path.join(self.UV_RENDER, subject, '%d_%d_%02d.jpg' % (vid, pitch, 0))
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
        normal = torch.Tensor(surface_normal).float()
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
            if self.is_train:
                yaw_val = self.max_yaw_angle - 1
                vid = tmp % yaw_val + 1
                pid = tmp // yaw_val - self.max_pitch_angle + self.mean_pitch
                pid = pid if pid < 0 else pid + 1
            else:
                yaw_val = self.max_yaw_angle
                vid = tmp % yaw_val
                pid = self.mean_pitch
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
            render_data = self.get_render(subject, num_views=self.num_views, view_id=vid,
                                            pitch=pid, random_sample=self.opt.random_multiview)
            sample_data = self.select_sampling_method(subject, render_data['calib'][0].numpy())

            res.update(render_data)
            res.update(sample_data)
            if self.num_sample_color:
                color_data = self.get_color_sampling(subject, view_id=vid)
                res.upate(color_data)
            return res
        except Exception as e:
            print(e)
            return self.get_item(index=random.randint(0, self.__len__() - 1))

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
        print(vid, pid)

if __name__ in '__main__':
    test(True)