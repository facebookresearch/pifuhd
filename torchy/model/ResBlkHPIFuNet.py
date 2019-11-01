import torch
import torch.nn as nn
import torch.nn.functional as F 
from .BasePIFuNet import BasePIFuNet
import functools
from .MLP import MLP
from .DepthNormalizer import DepthNormalizer
from ..net_util import *

from ..geometry import index, orthogonal, perspective

class ResBlkHPIFuNet(BasePIFuNet):
    def __init__(self,
                 opt,
                 netG,
                 projection_mode='orthogonal',
                 criteria={'clr': nn.MSELoss()}):
        super(ResBlkHPIFuNet, self).__init__(
            projection_mode=projection_mode,
            criteria=criteria)

        self.name = 'resblk_pifu'
        self.opt = opt
        self.num_views = self.opt.num_views

        norm_type = get_norm_layer(norm_type=opt.norm)
        self.image_filter = ResBlkFilter(opt, norm_layer=norm_type)

        self.mlp = MLP(
            filter_channels=self.opt.mlp_dim_color,
            num_views=self.num_views,
            res_layers=self.opt.mlp_res_layers,
            norm=self.opt.mlp_norm,
            last_op=nn.Tanh(),
            compose=False)

        self.normalizer = DepthNormalizer(opt)

        init_net(self)

        self.netG = netG

        for p in self.netG.parameters():
            p.requires_grad=False

    def filter(self, images, with_netG=False):
        '''
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B, C, H, W]
        '''
        if with_netG:
            self.netG.filter(images)
        self.im_feat = self.image_filter(images)
    
    def query(self, points, calibs, transforms=None, labels=None):
        '''
        given 3d points, we obtain 2d projection of these given the camera matrices.
        filter needs to be called beforehand.
        the prediction is stored to self.preds
        args:
            points: [B, 3, N] 3d points in world space
            calibs: [B, 3, 4] calibration matrices for each image
            transforms: [B, 2, 3] image space coordinate transforms
            labels: [B, C, N] ground truth labels (for supervision only)
        return:
            [B, C, N] prediction
        '''
        if labels is not None:
            self.labels = labels

        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]

        self.netG.query(points=points, calibs=calibs)
        z_feat = self.netG.phi.detach()
        point_local_feat_list = [self.index(self.im_feat, xy), z_feat]

        point_local_feat = torch.cat(point_local_feat_list, 1)
        self.preds = self.mlp(point_local_feat)[0]

    def train(self, mode=True):
        r"""Sets the module in training mode."""      
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.netG.eval()
        return self

    def get_error(self):
        '''
        return the loss given the ground truth labels and prediction
        '''
        error = {}
        error['Err(clr)'] = self.criteria['clr'](self.preds, self.labels)
 
        return error
  
    def forward(self, images, points, calibs, labels):
        # NET C
        # Phase 1: image filter
        self.filter(images, with_netG=True)

        # Phase 2: point query
        self.query(points=points, calibs=calibs, labels=labels)

        # get the prediction
        res = self.get_preds()
        # get the error
        err = self.get_error()

        return err, res

class ResBlk(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, last=False):
        super(ResBlk, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, last)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, last=False):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        if last:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)
    
    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResBlkFilter(nn.Module):
    def __init__(self, opt, input_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=6, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResBlkFilter, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                      norm_layer(ngf),
                      nn.ReLU(True)]
        
        n_downsampling = 2
        for i in range(n_downsampling): # add downsampling layers
            mult = 2 ** i 
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                           norm_layer(ngf * mult * 2),
                           nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            if i == n_blocks - 1:
                model += [ResBlk(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                 use_dropout=use_dropout, use_bias=use_bias, last=True)]
            else:
                model += [ResBlk(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                 use_dropout=use_dropout, use_bias=use_bias)]
        
        self.model = nn.Sequential(*model)

    def forward(self, image):
        return self.model(image)