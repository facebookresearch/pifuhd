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
        self.img_files = sorted([os.path.join(self.root,f) for f in os.listdir(self.root) if '.png' in f])
        self.IMG = os.path.join(self.root)

        self.phase = 'val'
        self.load_size = self.opt.loadSize

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.img_files)

    def get_item(self, index):
        img_path = self.img_files[index]
        # Name
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        # Calib
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        calib = torch.Tensor(projection_matrix).float()

        # image
        image = Image.open(img_path).convert('RGB')
        image = self.to_tensor(image)
        return {
            'name': img_name,
            'img': image.unsqueeze(0),
            'calib': calib.unsqueeze(0),
            'b_min': B_MIN,
            'b_max': B_MAX,
        }

    def __getitem__(self, index):
        return self.get_item(index)