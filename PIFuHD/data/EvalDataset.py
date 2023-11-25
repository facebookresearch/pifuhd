# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
from abc import abstractmethod

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class EvalDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, items, projection='orthogonal'):
        self.opt = opt
        self.projection_mode = projection

        self.root = self.opt.dataroot
        self.items = items
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
        return len(self.items)

    def get_item(self, index):
        im = self.items[index].img
        img_name = self.items[index].name

        if im.shape[2] == 4:
            im = im / 255.0
            im[:, :, :3] /= im[:, :, 3:] + 1e-8
            im = im[:, :, 3:] * im[:, :, :3] + 0.5 * (1.0 - im[:, :, 3:])
            im = (255.0 * im).astype(np.uint8)
        h, w = im.shape[:2]

        intrinsic = np.identity(4)
        trans_mat = np.identity(4)

        rect = self.get_human_box(index)
        im = self.crop_human_box(im, rect)

        scale_im2ndc = 1.0 / float(w // 2)
        scale = w / rect[2]
        trans_mat *= scale
        trans_mat[3, 3] = 1.0
        trans_mat[0, 3] = -scale * (rect[0] + rect[2] // 2 - w // 2) * scale_im2ndc
        trans_mat[1, 3] = scale * (rect[1] + rect[3] // 2 - h // 2) * scale_im2ndc

        intrinsic = np.matmul(trans_mat, intrinsic)
        im_512 = cv2.resize(im, (512, 512))
        im = cv2.resize(im, (self.load_size, self.load_size))

        image_512 = Image.fromarray(im_512[:, :, ::-1]).convert('RGB')
        image = Image.fromarray(im[:, :, ::-1]).convert('RGB')

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

    @abstractmethod
    def get_n_person(self, index):
        pass

    @abstractmethod
    def get_human_box(self, index):
        pass

    @abstractmethod
    def crop_human_box(self, image, rect):
        pass
