import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json 
import numpy as np
import cv2
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from numpy.linalg import inv

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from torchy.data import *
from torchy.model import *
from torchy.geometry import index
import torchy.networks as networks

from PIL import Image
import torchvision.transforms as transforms

parser = BaseOptions()

def gen_mesh(res, net, cuda, data, save_path, thresh=0.5, use_octree=True, components=False):
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    net.filter(image_tensor)
    if net.opt.use_front_normal:
        image_tensor = torch.cat([image_tensor, net.nmlF], 0)
    if net.opt.use_back_normal:
        image_tensor = torch.cat([image_tensor, net.nmlB], 0)
    b_min = data['b_min']
    b_max = data['b_max']
    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        cv2.imwrite(save_img_path, save_img)

        verts, faces, _, _ = reconstruction(
            net, cuda, calib_tensor, res, b_min, b_max, thresh, use_octree=use_octree, num_samples=100000)
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
        if 'calib_world' in data:
            calib_world = data['calib_world'].numpy()[0]
            verts = np.matmul(np.concatenate([verts, np.ones_like(verts[:,:1])],1), inv(calib_world).T)[:,:3]

        xyz_tensor = net.projection(verts_tensor, calib_tensor[:1])
        uv = xyz_tensor[:, :2, :]
        color = index(image_tensor[:1], uv).detach().cpu().numpy()[0].T
        color = color * 0.5 + 0.5
        save_obj_mesh_with_color(save_path, verts, faces, color)
    except Exception as e:
        print(e)


def recon(opt):
    # load checkpoints
    state_dict_path = None
    if opt.load_netG_checkpoint_path is not None:
        state_dict_path = opt.load_netG_checkpoint_path
    elif opt.resume_epoch < 0:
        state_dict_path = '%s/%s_train_latest' % (opt.checkpoints_path, opt.name)
        opt.resume_epoch = 0
    else:
        state_dict_path = '%s/%s_train_epoch_%d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)
    
    state_dict = None
    if state_dict_path is not None and os.path.exists(state_dict_path):
        print('Resuming from ', state_dict_path)
        state_dict = torch.load(state_dict_path)    
        if 'opt' in state_dict:
            print('Warning: opt is overwritten.')
            dataroot = opt.dataroot
            resolution = opt.resolution
            results_path = opt.results_path
            opt = state_dict['opt']
            opt.dataroot = dataroot
            opt.resolution = resolution
            opt.results_path = results_path
    else:
        raise Exception('failed loading state dict!', state_dict_path)
    
    parser.print_options(opt)

    cuda = torch.device('cuda:%d' % opt.gpu_id)

    # test_dataset = EvalDataset(opt)
    test_dataset = EvalWPoseDataset(opt)

    print('test data size: ', len(test_dataset))
    projection_mode = test_dataset.projection_mode

    try:
        if opt.use_aio_normal:
            netG = HGPIFuNetwNMLAIO(opt, projection_mode).to(device=cuda)
        else:
            netG = HGPIFuNetwNML(opt, projection_mode).to(device=cuda)
    except:
        netG = HGPIFuNetwNML(opt, projection_mode).to(device=cuda)

    def set_eval():
        netG.eval()

    # load checkpoints
    if state_dict is not None:
        if 'model_state_dict' in state_dict:
            netG.load_state_dict(state_dict['model_state_dict'])
        else: # this is deprecated but keep it for now.
            netG.load_state_dict(state_dict)

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s/recon' % (opt.results_path, opt.name), exist_ok=True)

    ## test
    with torch.no_grad():
        set_eval()

        print('generate mesh (test) ...')
        for test_data in tqdm(test_dataset):
            save_path = '%s/%s/recon/result_%s.obj' % (opt.results_path, opt.name, test_data['name'])
            gen_mesh(opt.resolution, netG, cuda, test_data, save_path, components=opt.use_compose)

def reconWrapper(args=None):
    opt = parser.parse(args)
    recon(opt)

if __name__ == '__main__':
    reconWrapper()
  
