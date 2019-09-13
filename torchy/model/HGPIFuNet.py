import torch
import torch.nn as nn
import torch.nn.functional as F 
from .BasePIFuNet import BasePIFuNet
from .MLP import MLP
from .DepthNormalizer import DepthNormalizer
from .HGFilters import *
from ..net_util import init_net

class HGPIFuNet(BasePIFuNet):
    '''
    HGPIFu uses stacked hourglass as an image encoder.
    '''

    def __init__(self, 
                 opt, 
                 projection_mode='orthogonal',
                 error_term=nn.MSELoss(),
                 ):
        super(HGPIFuNet, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)

        self.name = 'hg_pifu'

        self.opt = opt
        self.num_views = self.opt.num_views

        self.image_filter = HGFilter(opt)

        self.mlp = MLP(
            filter_channels=self.opt.mlp_dim,
            num_views=self.num_views,
            no_residual=self.opt.no_residual,
            last_op=nn.Sigmoid())

        self.normalizer = DepthNormalizer(opt)

        self.im_feat_list = []
        self.tmpx = None
        self.normx = None

        self.intermediate_preds_list = []

        init_net(self)
    
    def filter(self, images):
        '''
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B, C, H, W]
        '''
        self.im_feat_list, self.normx = self.image_filter(images)
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]
        
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
        z = xyz[:, 2:3, :]

        z_feat = self.normalizer(z, calibs=calibs)

        self.intermediate_preds_list = []

        for im_feat in self.im_feat_list:
            point_local_feat_list = [self.index(im_feat, xy), z_feat]
            
            point_local_feat = torch.cat(point_local_feat_list, 1)

            pred = self.mlp(point_local_feat)
            self.intermediate_preds_list.append(pred)
        
        self.preds = self.intermediate_preds_list[-1]

    def get_im_feat(self):
        '''
        return the image filter in the last stack
        return:
            [B, C, H, W]
        '''
        return self.im_feat_list[-1]

    def get_error(self):
        '''
        return the loss given the ground truth labels and prediction
        '''
        error = 0
        for preds in self.intermediate_preds_list:
            error += self.error_term(preds, self.labels)
        return error

    