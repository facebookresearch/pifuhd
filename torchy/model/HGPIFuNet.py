import torch
import torch.nn as nn
import torch.nn.functional as F 
from .BasePIFuNet import BasePIFuNet
from .MLP import MLP
from .DepthNormalizer import DepthNormalizer
from .HGFilters import *
from .VolumetricEncoder import *
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
        self.image_filter = HGFilter(opt.num_stack, opt.hg_depth, opt.hg_dim, 
                                     opt.norm, opt.hg_down, opt.use_sigmoid)

        self.mlp = MLP(
            filter_channels=self.opt.mlp_dim,
            num_views=self.num_views,
            res_layers=self.opt.mlp_res_layers,
            last_op=nn.Sigmoid() if not self.opt.use_tsdf else nn.Tanh())

        if self.opt.sp_enc_type == 'vol_enc':
            self.spatial_enc = VolumetricEncoder(opt)
        elif self.opt.sp_enc_type == 'z':
            self.spatial_enc = DepthNormalizer(opt)
        else:
            raise NameError('unknown spatial encoding type')

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
        if self.opt.sp_enc_type == 'vol_enc':
            self.spatial_enc.filter(images)

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

        sp_feat = self.spatial_enc(xyz, calibs=calibs)

        self.intermediate_preds_list = []

        for im_feat in self.im_feat_list:
            if self.opt.sp_enc_type == 'vol_enc' and self.opt.sp_no_pifu:
                point_local_feat = sp_feat
            else:
                point_local_feat_list = [self.index(im_feat, xy), sp_feat]            
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

    