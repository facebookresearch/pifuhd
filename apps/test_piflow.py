import os
import argparse
import time
import numpy as np
import math
import random
import trimesh

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchdiffeq import odeint_adjoint as odeint

from lib.options import BaseOptions
from lib.visualizer import Visualizer
from lib.mesh_util import *
from lib.sample_util import *
from torchy.data import *
from torchy.model import *
from torchy.geometry import index

from PIL import Image
import torchvision.transforms as transforms

parser = argparse.ArgumentParser('PIFlow train')
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--dataroot', type=str)
parser.add_argument('--loadSize', type=int, default=512)
parser.add_argument('--batch_time', type=int, default=2)
parser.add_argument('--num_sample_inout', type=int, default=1000)
parser.add_argument('--sigma', type=float, default=0.05)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--n_threads', type=int, default=10)
args = parser.parse_args()

class VideoDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, projection='orthogonal'):
        self.opt = opt
        self.projection_mode = projection

        self.root = self.opt.dataroot
        self.img_files = sorted([os.path.join(self.root,f) for f in os.listdir(self.root) if '.png' in f])
        self.img_files = self.img_files[:30]
        self.IMG = os.path.join(self.root)

        self.phase = 'train'
        self.load_size = self.opt.loadSize

        self.mesh = trimesh.load(os.path.join(self.root, 'result_0000.obj'))

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.img_files)

    def get_space_batch(self):
        img_path = self.img_files[0]
        # Name
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # image
        image = Image.open(img_path).convert('RGB')
        image = self.to_tensor(image)

        # Calib
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        calib_tensor = torch.Tensor(projection_matrix).float()

        surface_points, fid = trimesh.sample.sample_surface(self.mesh, self.opt.num_sample_inout)
        theta = 2.0 * math.pi * np.random.rand(surface_points.shape[0])
        phi = np.arccos(1 - 2 * np.random.rand(surface_points.shape[0]))
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        dir = np.stack([x,y,z],1)
        radius = np.random.normal(scale=self.opt.sigma, size=[surface_points.shape[0],1])
        sample_points = surface_points + radius * dir
        sample_points = torch.Tensor(sample_points.T).float()

        return {
            'img': image[None],
            'calib': calib_tensor[None],
            'samples': sample_points[None]
        }

    def get_time_batch(self):
        idxs = sorted(random.sample(range(1,len(self.img_files)), self.opt.batch_time))

        datas = {}
        for i in idxs:
            data = self.get_item(i)
            for key in data.keys():
                if key not in datas:
                    datas[key] = [data[key]]
                else:
                    datas[key].append(data[key])

        for key in datas.keys():
            datas[key] = torch.stack(datas[key], 0)
        
        return datas
        
    def get_start_mesh(self):
        verts = torch.Tensor(self.mesh.vertices.T).float()
        
        # Calib
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        calib_tensor = torch.Tensor(projection_matrix).float()

        return {
            'vertices': verts,
            'faces': self.mesh.faces,
            'calib': calib_tensor
        }

    def get_item(self, index):
        if index == 0:
            index = random.randint(1,len(self.img_files))
        img_path = self.img_files[index]
        
        # Name
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # image
        image = Image.open(img_path).convert('RGB')
        image = self.to_tensor(image)
        
        t = float(index)/len(self.img_files)
        return {
            'img': image,
            'time': torch.Tensor([t]),
        }

    def __getitem__(self, index):
        return self.get_item(index)

class PIFlow(nn.Module):
    def __init__(self):
        super(PIFlow, self).__init__()

        filter_channels = [4, 1024, 512, 256, 128, 3]
        self.net = MLP(filter_channels, res_layers=[2,3,4], last_op=nn.Tanh())

    def forward(self, t, y):
        '''
        args:
            t: (Scalar) time
            y: [B, C, N] current value (e.g., positions)
        return:
            dydt at t=t
        '''
        t = 2.0 * t - 1.0 # to range [-1, 1]
        if len(t.size()) > 0:
            return self.net(torch.cat([y, t[None,:,None].expand_as(y[:,:1,:])],1))
        else:
            return self.net(torch.cat([y, t[None,None,None].expand_as(y[:,:1,:])],1))            

if __name__ == '__main__':
    # load checkpoints
    state_dict_path = args.checkpoint
    
    state_dict = None
    if state_dict_path is not None and os.path.exists(state_dict_path):
        print('Resuming from ', state_dict_path)
        state_dict = torch.load(state_dict_path)    
        if 'opt' in state_dict:
            print('Warning: opt is overwritten.')
            opt = state_dict['opt']
    else:
        raise Exception('failed loading state dict!', state_dict_path)
    
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    dataset = VideoDataset(args)

    data_loader = DataLoader(dataset,
                             batch_size=args.batch_time, shuffle=True,
                             num_workers=args.n_threads, pin_memory=False)

    print('data size: ', len(dataset))
    projection_mode = dataset.projection_mode

    netG = HGPIFuNet(opt, projection_mode).to(device=cuda)

    def set_train():
        netG.train()

    def set_eval():
        netG.eval()

    # load checkpoints
    if state_dict is not None:
        if 'model_state_dict' in state_dict:
            netG.load_state_dict(state_dict['model_state_dict'])
        else: # this is deprecated but keep it for now.
            netG.load_state_dict(state_dict)

    flow = PIFlow().to(cuda)
    optimizer = optim.Adam(flow.parameters(), lr=1e-3)

    # train flow network
    set_eval()
    for epoch in range(args.n_epoch):        
        loss_hist = []
        for it in range(len(dataset)//args.batch_time):
            time_data = dataset.get_time_batch()

            space_data = dataset.get_space_batch()
            image_src_tensor = space_data['img'].to(device=cuda)
            sample_tensor = space_data['samples'].to(device=cuda)
            calib_tensor = space_data['calib'].to(device=cuda)

            optimizer.zero_grad()

            image_time_tensor = time_data['img'].to(device=cuda)
            time_tensor = time_data['time'].float().to(device=cuda)
            time_tensor = torch.cat([torch.zeros_like(time_tensor[:1]), time_tensor], 0)
            image_tensor = torch.cat([image_src_tensor, image_time_tensor], 0)
            netG.filter(image_tensor)
            
            pos_tensor = odeint(flow, sample_tensor, time_tensor, rtol=1e-3, atol=1e-5) # returns (Bt, Bs, C, N)

            pos_tensor = pos_tensor[1:].view(-1, *pos_tensor.size()[2:])
            pos_tensor = torch.cat([sample_tensor, pos_tensor], 0)
            netG.query(pos_tensor, calib_tensor.expand(pos_tensor.size(0),-1,-1))

            preds = netG.get_preds()
            loss = nn.MSELoss()(preds[1:], preds[:1].expand_as(preds[1:]).detach())
            loss.backward()
            optimizer.step()

            loss_hist.append(loss.item())

        print('epoch %d loss: %.4f' % (epoch, np.average(loss_hist)))

    mesh_data = dataset.get_start_mesh()
    sample_tensor = mesh_data['vertices'][None].to(device=cuda)
    calib_tensor = mesh_data['calib'][None].to(device=cuda)
    time_tensor = torch.Tensor([0,0.5,1.0]).to(device=cuda)
    pos_tensor = odeint(flow, sample_tensor, time_tensor)
    vertices = pos_tensor[1,0].t().detach().cpu().numpy()
    save_obj_mesh('test1.obj', vertices, mesh_data['faces'][:,::-1])
    vertices = pos_tensor[2,0].t().detach().cpu().numpy()
    save_obj_mesh('test2.obj', vertices, mesh_data['faces'][:,::-1])
