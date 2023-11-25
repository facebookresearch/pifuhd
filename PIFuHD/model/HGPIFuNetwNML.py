# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from .BasePIFuNet import BasePIFuNet
from .MLP import MLP
from .DepthNormalizer import DepthNormalizer
from .HGFilters import HGFilter
from ..net_util import init_net
from ..networks import define_G
import cv2

class HGPIFuNetwNML(BasePIFuNet):
    '''
    HGPIFu uses stacked hourglass as an image encoder.
    '''

    def __init__(self, 
                 opt, 
                 projection_mode='orthogonal',
                 criteria={'occ': nn.MSELoss()}
                 ):
        super(HGPIFuNetwNML, self).__init__(
            projection_mode=projection_mode,
            criteria=criteria)

        self.name = 'hg_pifu'

        in_ch = 3
        try:
            if opt.use_front_normal:
                in_ch += 3
            if opt.use_back_normal:
                in_ch += 3
        except:
            pass
        self.opt = opt
        self.image_filter = HGFilter(opt.num_stack, opt.hg_depth, in_ch, opt.hg_dim, 
                                     opt.norm, opt.hg_down, False)

        self.mlp = MLP(
            filter_channels=self.opt.mlp_dim,
            merge_layer=self.opt.merge_layer,
            res_layers=self.opt.mlp_res_layers,
            norm=self.opt.mlp_norm,
            last_op=nn.Sigmoid())

        self.spatial_enc = DepthNormalizer(opt)

        self.im_feat_list = []
        self.tmpx = None
        self.normx = None
        self.phi = None

        self.intermediate_preds_list = []

        init_net(self)

        self.netF = None
        self.netB = None
        try:
            if opt.use_front_normal:
                self.netF = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")
            if opt.use_back_normal:
                self.netB = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")
        except:
            pass
        self.nmlF = None
        self.nmlB = None

    def loadFromHGHPIFu(self, net):
        hgnet = net.image_filter
        pretrained_dict = hgnet.state_dict()            
        model_dict = self.image_filter.state_dict()

        pretrained_dict = {k: v for k, v in hgnet.state_dict().items() if k in model_dict}                    

        for k, v in pretrained_dict.items():                      
            if v.size() == model_dict[k].size():
                model_dict[k] = v

        not_initialized = set()
               
        for k, v in model_dict.items():
            if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                not_initialized.add(k.split('.')[0])
        
        print('not initialized', sorted(not_initialized))
        self.image_filter.load_state_dict(model_dict) 

        pretrained_dict = net.mlp.state_dict()            
        model_dict = self.mlp.state_dict()

        pretrained_dict = {k: v for k, v in net.mlp.state_dict().items() if k in model_dict}                    

        for k, v in pretrained_dict.items():                      
            if v.size() == model_dict[k].size():
                model_dict[k] = v

        not_initialized = set()
               
        for k, v in model_dict.items():
            if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                not_initialized.add(k.split('.')[0])
        
        print('not initialized', sorted(not_initialized))
        self.mlp.load_state_dict(model_dict) 

    def filter(self, images):
        '''
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B, C, H, W]
        '''
        nmls = []
        # if you wish to train jointly, remove detach etc.
        with torch.no_grad():
            if self.netF is not None:
                self.nmlF = self.netF.forward(images).detach()
                nmls.append(self.nmlF)
            if self.netB is not None:
                self.nmlB = self.netB.forward(images).detach()
                nmls.append(self.nmlB)
        if len(nmls) != 0:
            nmls = torch.cat(nmls,1)
            if images.size()[2:] != nmls.size()[2:]:
                nmls = nn.Upsample(size=images.size()[2:], mode='bilinear', align_corners=True)(nmls)
            images = torch.cat([images,nmls],1)


        self.im_feat_list, self.normx = self.image_filter(images)

        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]
        
    def query(self, points, calibs, transforms=None, labels=None, update_pred=True, update_phi=True):
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
        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]

        # if the point is outside bounding box, return outside.
        in_bb = (xyz >= -1) & (xyz <= 1)
        in_bb = in_bb[:, 0, :] & in_bb[:, 1, :] & in_bb[:, 2, :]
        in_bb = in_bb[:, None, :].detach().float()

        if labels is not None:
            self.labels = in_bb * labels

        sp_feat = self.spatial_enc(xyz, calibs=calibs)

        intermediate_preds_list = []

        phi = None
        for i, im_feat in enumerate(self.im_feat_list):
            point_local_feat_list = [self.index(im_feat, xy), sp_feat]       
            point_local_feat = torch.cat(point_local_feat_list, 1)
            pred, phi = self.mlp(point_local_feat)
            pred = in_bb * pred

            intermediate_preds_list.append(pred)
        
        if update_phi:
            self.phi = phi

        if update_pred:
            self.intermediate_preds_list = intermediate_preds_list
            self.preds = self.intermediate_preds_list[-1]

    def calc_normal(self, points, calibs, transforms=None, labels=None, delta=0.01, fd_type='forward'):
        '''
        return surface normal in 'model' space.
        it computes normal only in the last stack.
        note that the current implementation use forward difference.
        args:
            points: [B, 3, N] 3d points in world space
            calibs: [B, 3, 4] calibration matrices for each image
            transforms: [B, 2, 3] image space coordinate transforms
            delta: perturbation for finite difference
            fd_type: finite difference type (forward/backward/central) 
        '''
        pdx = points.clone()
        pdx[:,0,:] += delta
        pdy = points.clone()
        pdy[:,1,:] += delta
        pdz = points.clone()
        pdz[:,2,:] += delta

        if labels is not None:
            self.labels_nml = labels

        points_all = torch.stack([points, pdx, pdy, pdz], 3)
        points_all = points_all.view(*points.size()[:2],-1)
        xyz = self.projection(points_all, calibs, transforms)
        xy = xyz[:, :2, :]

        im_feat = self.im_feat_list[-1]
        sp_feat = self.spatial_enc(xyz, calibs=calibs)

        point_local_feat_list = [self.index(im_feat, xy), sp_feat]            
        point_local_feat = torch.cat(point_local_feat_list, 1)

        pred = self.mlp(point_local_feat)[0]

        pred = pred.view(*pred.size()[:2],-1,4) # (B, 1, N, 4)

        # divide by delta is omitted since it's normalized anyway
        dfdx = pred[:,:,:,1] - pred[:,:,:,0]
        dfdy = pred[:,:,:,2] - pred[:,:,:,0]
        dfdz = pred[:,:,:,3] - pred[:,:,:,0]

        nml = -torch.cat([dfdx,dfdy,dfdz], 1)
        nml = F.normalize(nml, dim=1, eps=1e-8)

        self.nmls = nml

    def get_im_feat(self):
        '''
        return the image filter in the last stack
        return:
            [B, C, H, W]
        '''
        return self.im_feat_list[-1]


    def get_error(self, gamma):
        '''
        return the loss given the ground truth labels and prediction
        '''
        error = {}
        error['Err(occ)'] = 0
        for preds in self.intermediate_preds_list:
            error['Err(occ)'] += self.criteria['occ'](preds, self.labels, gamma)
        
        error['Err(occ)'] /= len(self.intermediate_preds_list)
        
        if self.nmls is not None and self.labels_nml is not None:
            error['Err(nml)'] = self.criteria['nml'](self.nmls, self.labels_nml)

        return error

    def forward(self, images, points, calibs, labels, gamma, points_nml=None, labels_nml=None):
        self.filter(images)
        self.query(points, calibs, labels=labels)
        if points_nml is not None and labels_nml is not None:
            self.calc_normal(points_nml, calibs, labels=labels_nml)
        res = self.get_preds()
            
        err = self.get_error(gamma)

        return err, res
