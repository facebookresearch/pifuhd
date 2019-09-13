import torch
import torch.nn as nn
import torch.nn.functional as F 
from .BasePIFuNet import BasePIFuNet
import functools
from .MLP import MLP
from .DepthNormalizer import DepthNormalizer
from ..net_util import *

from ..geometry import index, orthogonal, perspective

class ResBlkPIFuNet(BasePIFuNet):
    def __init__(self,
                 projection_mode='orthogonal',
                 error_term=nn.MSELoss()):
        super(ResBlkPIFuNet, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)

        self.name = 'resblk_pifu'
        self.opt = opt

        norm_type = get_norm_layer(norm_type=opt.norm)
        self.image_filter = ResBlkFilter(opt, norm_layer=norm_type)

        self.mlp = MLP(
            filter_channels=self.mlp_dim_color,
            num_views=self.opt.num_views,
            no_residual=self.opt.no_residual,
            last_op=nn.Tanh())
        
        self.normalizer = DepthNormalizer(opt)

        init_net(self)

    def filter(self, images, f_volumes=None):
        '''
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B, C, H, W]
        '''
        self.im_feat = self.image_filter(images, f_volumes)

    def attach(self, im_feat):
        self.im_feat = torch.cat([im_feat, self.im_feat], 1)
    
    def query(self, points, calibs, trasnforms=None, labels=None):
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
        z = xyz[:, 2:3, :]

        z_feat = self.normalizer(z)

        point_local_feat_list = [self.index(self.im_feat, xy), z_feat]

        point_local_feat = torch.cat(point_local_feat_list, 1)

        self.preds = self.mlp(point_local_feat)

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
    def __init__(self, opt, input_nc=3, output_nc=256, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
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
                                        
        if opt.use_tanh:
            model += [nn.Tanh()]
        
        self.model = nn.Sequential(*model)

    def forward(self, image):
        return self.model(input)