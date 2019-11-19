import torch
import torch.nn as nn
import torch.nn.functional as F 

from .HGFilters import *
from .HG3D import *
from .UNet3D import *
from .SimpleNet3D import *
from ..geometry import index3d

class VolumetricEncoder(nn.Module):
    def __init__(self, opt):
        super(VolumetricEncoder, self).__init__()

        self.opt = opt
        self.vol_ch_in = opt.vol_ch_in
        self.vol_ch_out = opt.vol_ch_out

        self.image2vol = HGFilter(1, opt.vol_hg_depth, 3, 32 * self.vol_ch_in, opt.vol_norm, 'ave_pool', False)
        if self.opt.vol_net == 'hg':
            self.vol_enc = HGFilter3D(self.vol_ch_in, 1, 3, self.vol_ch_out, opt.vol_norm, 'ave_pool', False)
        elif self.opt.vol_net == 'unet':
            self.vol_enc = UnetGenerator(self.vol_ch_in, self.vol_ch_out, ngf=2*self.vol_ch, norm=opt.vol_norm)
        elif self.opt.vol_net == 'simple':
            self.vol_enc = Simple3DNet(self.vol_ch_in, self.vol_ch_out, 2, opt.vol_norm)
        else:
            raise NameError('unkown encoder type %s' % self.opt.vol_net)

        self.vol_feat = None
    
    def filter(self, x):
        '''
        given input images, obtain the correponding volumetric feature.
        args:
            x: [B, 3, H, W] input image
        param:       
            vol_feat: [B, C, D', H', W']
        '''
        # assuming x is [512 x 512] -> [128 x 128]
        x = nn.Upsample(scale_factor=0.25, mode='bilinear')(x)
        y = self.image2vol(x)[0][-1]
        y = y.view(y.size(0), self.vol_ch_in, -1, y.size(2), y.size(3))

        self.vol_feat = self.vol_enc(y)

    def forward(self, xyz, calibs=None, index_feat=None):
        # print(xyz[0,:,:].max(1)[0], xyz[0,:,:].min(1)[0])
        # normalize z value 
        # xyz[:,2] = xyz[:,2] * (self.opt.loadSize // 4) / self.opt.z_size

        # print(xyz[0,:,:].max(1)[0])
        return index3d(self.vol_feat, xyz)
