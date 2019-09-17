import os
import random

import numpy as np 
from numpy.linalg import inv
from PIL import Image, ImageOps
from PIL.ImageFilter import GaussianBlur
import cv2
import trimesh
from trimesh.proximity import *
import matplotlib
import matplotlib.cm as cm

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from tqdm import tqdm
from .RPDataset import RPDataset

def scalar2color(x, min=-1.0, max=1.0):
    
    norm = matplotlib.colors.Normalize(vmin=min, vmax=max, clip=True)
    # use jet colormap
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)

    colors = []
    for v in x:
        colors.append(mapper.to_rgba(v))

    return np.stack(colors, 0)[:,:3]

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

class RPTSDFDataset(RPDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):
        RPDataset.__init__(self, opt, phase=phase)
    
    def select_sampling_method(self, subject, calib):
        mode = self.opt.sampling_mode
        return self.load_points_tsdf(subject, mode, calib)

    def load_points_tsdf(self, subject, mode, calib, num_samples=0, num_files=20):
        '''
        load points and tsdf from precomputed numpy array
        '''
        TSDF_DIR = os.path.join(self.TSDF, mode, subject)

        rand_idx = np.random.randint(num_files)
        tsdf_file = os.path.join(TSDF_DIR, '%05d.xyzd.npy' % rand_idx)

        tsdf = np.load(tsdf_file)

        inbb = np.matmul(np.concatenate([tsdf[:,:3], np.ones((tsdf.shape[0],1))], 1), calib.T)[:, :3]
        inbb = (inbb[:, 0] >= -1) & (inbb[:, 0] <= 1) & (inbb[:, 1] >= -1) & \
               (inbb[:, 1] <= 1) & (inbb[:, 2] >= -1) & (inbb[:, 2] <= 1)
        
        tsdf = tsdf[inbb]
        np.random.shuffle(tsdf)

        # this balance positive and negative sdf, which seems critical for stable training
        tsdf_pos = tsdf[tsdf[:,3] >= 0]
        tsdf_neg = tsdf[tsdf[:,3] < 0]

        if num_samples <= 0:
            num_samples = self.num_sample_inout

        nin = tsdf_pos.shape[0]
        tsdf_pos = tsdf_pos[
                        :num_samples // 2] if nin > num_samples // 2 else tsdf_pos
        tsdf_neg = tsdf_neg[
                         :num_samples // 2] if nin > num_samples // 2 else tsdf_neg[
                             :(num_samples - nin)]    

        tsdf = np.concatenate([tsdf_pos, tsdf_neg], 0) # [3, N]

        samples = tsdf[:,:3]
        labels = 0.5 * tsdf[:,3:] + 0.5 # rescale to [0, 1]

        # C = scalar2color(labels[:,0])
        # save_points_color('tmp.obj', samples, C)
        # exit()
        samples = torch.Tensor(samples.T).float()
        labels = torch.Tensor(labels.T).float()

        return {
            'samples': samples,
            'labels': labels,
        }

