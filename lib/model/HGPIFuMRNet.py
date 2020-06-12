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
import cv2

class HGPIFuMRNet(BasePIFuNet):
    '''
    HGPIFu uses stacked hourglass as an image encoder.
    '''

    def __init__(self, 
                 opt, 
                 netG,
                 projection_mode='orthogonal',
                 criteria={'occ': nn.MSELoss()}
                 ):
        super(HGPIFuMRNet, self).__init__(
            projection_mode=projection_mode,
            criteria=criteria)

        self.name = 'hg_pifu'

        in_ch = 3
        try:
            if netG.opt.use_front_normal:
                in_ch += 3
            if netG.opt.use_back_normal:
                in_ch += 3
        except:
            pass

        self.opt = opt
        self.image_filter = HGFilter(opt.num_stack, opt.hg_depth, in_ch, opt.hg_dim, 
                                     opt.norm, 'no_down', False)

        self.mlp = MLP(
            filter_channels=self.opt.mlp_dim,
            merge_layer=-1,
            res_layers=self.opt.mlp_res_layers,
            norm=self.opt.mlp_norm,
            last_op=nn.Sigmoid())

        self.im_feat_list = []
        self.preds_interm = None
        self.preds_low = None
        self.w = None
        self.gamma = None

        self.intermediate_preds_list = []

        init_net(self)

        self.netG = netG

    def train(self, mode=True):
        r"""Sets the module in training mode."""      
        self.training = mode
        for module in self.children():
            module.train(mode)
        if not self.opt.train_full_pifu:
            self.netG.eval()
        return self

    def filter_global(self, images):
        '''
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B1, C, H, W]
        '''
        if self.opt.train_full_pifu:
            self.netG.filter(images)
        else:
            with torch.no_grad():
                self.netG.filter(images)

    def filter_local(self, images, rect=None):
        '''
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B1, B2, C, H, W]
        '''
        nmls = []
        try:
            if self.netG.opt.use_front_normal:
                nmls.append(self.netG.nmlF)
            if self.netG.opt.use_back_normal:
                nmls.append(self.netG.nmlB)
        except:
            pass

        if len(nmls):
            nmls = nn.Upsample(size=(self.opt.loadSizeBig,self.opt.loadSizeBig), mode='bilinear', align_corners=True)(torch.cat(nmls,1))
            
            # it's kind of damn way.
            if rect is None:
                images = torch.cat([images, nmls[:,None].expand(-1,images.size(1),-1,-1,-1)], 2)
            else:
                nml = []
                for i in range(rect.size(0)):
                    for j in range(rect.size(1)):
                        x1, y1, x2, y2 = rect[i,j]
                        tmp = nmls[i,:,y1:y2,x1:x2]
                        nml.append(nmls[i,:,y1:y2,x1:x2])
                nml = torch.stack(nml, 0).view(*rect.shape[:2],*nml[0].size())
                images = torch.cat([images, nml], 2)

        self.im_feat_list, self.normx = self.image_filter(images.view(-1,*images.size()[2:]))
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]
        
    def query(self, points, calib_local, calib_global=None, transforms=None, labels=None):
        '''
        given 3d points, we obtain 2d projection of these given the camera matrices.
        filter needs to be called beforehand.
        the prediction is stored to self.preds
        args:
            points: [B1, B2, 3, N] 3d points in world space
            calibs_local: [B1, B2, 4, 4] calibration matrices for each image
            calibs_global: [B1, 4, 4] calibration matrices for each image
            transforms: [B1, 2, 3] image space coordinate transforms
            labels: [B1, B2, C, N] ground truth labels (for supervision only)
        return:
            [B, C, N] prediction
        '''
        if calib_global is not None:
            B = calib_local.size(1)
        else:
            B = 1
            points = points[:,None]
            calib_global = calib_local
            calib_local = calib_local[:,None]

        ws = []
        preds = []
        preds_interm = []
        preds_low = []
        gammas = []
        newlabels = []
        for i in range(B):
            xyz = self.projection(points[:,i], calib_local[:,i], transforms)
            
            xy = xyz[:, :2, :]

            # if the point is outside bounding box, return outside.
            in_bb = (xyz >= -1) & (xyz <= 1)
            in_bb = in_bb[:, 0, :] & in_bb[:, 1, :]
            in_bb = in_bb[:, None, :].detach().float()

            self.netG.query(points=points[:,i], calibs=calib_global)
            preds_low.append(torch.stack(self.netG.intermediate_preds_list,0))

            if labels is not None:
                newlabels.append(in_bb * labels[:,i])
                with torch.no_grad():
                    ws.append(in_bb.size(2) / in_bb.view(in_bb.size(0),-1).sum(1))
                    gammas.append(1 - newlabels[-1].view(newlabels[-1].size(0),-1).sum(1) / in_bb.view(in_bb.size(0),-1).sum(1))

            z_feat = self.netG.phi
            if not self.opt.train_full_pifu:
                z_feat = z_feat.detach()
                        
            intermediate_preds_list = []
            for j, im_feat in enumerate(self.im_feat_list):
                point_local_feat_list = [self.index(im_feat.view(-1,B,*im_feat.size()[1:])[:,i], xy), z_feat]
                point_local_feat = torch.cat(point_local_feat_list, 1)
                pred = self.mlp(point_local_feat)[0]
                pred = in_bb * pred
                intermediate_preds_list.append(pred)

            preds_interm.append(torch.stack(intermediate_preds_list,0))
            preds.append(intermediate_preds_list[-1])

        self.preds = torch.cat(preds,0)
        self.preds_interm = torch.cat(preds_interm, 1) # first dim is for intermediate predictions
        self.preds_low = torch.cat(preds_low, 1) # first dim is for intermediate predictions
        
        if labels is not None:
            self.w = torch.cat(ws,0)
            self.gamma = torch.cat(gammas,0)
            self.labels = torch.cat(newlabels,0)

    def calc_normal(self, points, calib_local, calib_global, transforms=None, labels=None, delta=0.001, fd_type='forward'):
        '''
        return surface normal in 'model' space.
        it computes normal only in the last stack.
        note that the current implementation use forward difference.
        args:
            points: [B1, B2, 3, N] 3d points in world space
            calibs_local: [B1, B2, 4, 4] calibration matrices for each image
            calibs_global: [B1, 4, 4] calibration matrices for each image
            transforms: [B1, 2, 3] image space coordinate transforms
            labels: [B1, B2, 3, N] ground truth normal
            delta: perturbation for finite difference
            fd_type: finite difference type (forward/backward/central) 
        '''
        B = calib_local.size(1)

        if labels is not None:
            self.labels_nml = labels.view(-1,*labels.size()[2:])

        im_feat = self.im_feat_list[-1].view(-1,B,*self.im_feat_list[-1].size()[1:])

        nmls = []
        for i in range(B):
            points_sub = points[:,i]
            pdx = points_sub.clone()
            pdx[:,0,:] += delta
            pdy = points_sub.clone()
            pdy[:,1,:] += delta
            pdz = points_sub.clone()
            pdz[:,2,:] += delta

            points_all = torch.stack([points_sub, pdx, pdy, pdz], 3)
            points_all = points_all.view(*points_sub.size()[:2],-1)
            xyz = self.projection(points_all, calib_local[:,i], transforms)
            xy = xyz[:, :2, :]


            self.netG.query(points=points_all, calibs=calib_global, update_pred=False)
            z_feat = self.netG.phi
            if not self.opt.train_full_pifu:
                z_feat = z_feat.detach()

            point_local_feat_list = [self.index(im_feat[:,i], xy), z_feat]            
            point_local_feat = torch.cat(point_local_feat_list, 1)
            pred = self.mlp(point_local_feat)[0]

            pred = pred.view(*pred.size()[:2],-1,4) # (B, 1, N, 4)

            # divide by delta is omitted since it's normalized anyway
            dfdx = pred[:,:,:,1] - pred[:,:,:,0]
            dfdy = pred[:,:,:,2] - pred[:,:,:,0]
            dfdz = pred[:,:,:,3] - pred[:,:,:,0]

            nml = -torch.cat([dfdx,dfdy,dfdz], 1)
            nml = F.normalize(nml, dim=1, eps=1e-8)

            nmls.append(nml)
        
        self.nmls = torch.stack(nmls,1).view(-1,3,points.size(3))

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

        error = {}
        if self.opt.train_full_pifu:
            if not self.opt.no_intermediate_loss:
                error['Err(occ)'] = 0.0
                for i in range(self.preds_low.size(0)):
                    error['Err(occ)'] += self.criteria['occ'](self.preds_low[i], self.labels, self.gamma, self.w)
                error['Err(occ)'] /= self.preds_low.size(0)

            error['Err(occ:fine)'] = 0.0
            for i in range(self.preds_interm.size(0)):
                error['Err(occ:fine)'] += self.criteria['occ'](self.preds_interm[i], self.labels, self.gamma, self.w)
            error['Err(occ:fine)'] /= self.preds_interm.size(0)

            if self.nmls is not None and self.labels_nml is not None:
                error['Err(nml:fine)'] = self.criteria['nml'](self.nmls, self.labels_nml)
        else:
            error['Err(occ:fine)'] = 0.0
            for i in range(self.preds_interm.size(0)):
                error['Err(occ:fine)'] += self.criteria['occ'](self.preds_interm[i], self.labels, self.gamma, self.w)
            error['Err(occ:fine)'] /= self.preds_interm.size(0)

            if self.nmls is not None and self.labels_nml is not None:
                error['Err(nml:fine)'] = self.criteria['nml'](self.nmls, self.labels_nml)
        
        return error


    def forward(self, images_local, images_global, points, calib_local, calib_global, labels, points_nml=None, labels_nml=None, rect=None):
        self.filter_global(images_global)
        self.filter_local(images_local, rect)
        self.query(points, calib_local, calib_global, labels=labels)
        if points_nml is not None and labels_nml is not None:
            self.calc_normal(points_nml, calib_local, calib_global, labels=labels_nml)
        res = self.get_preds()
            
        err = self.get_error()

        return err, res
