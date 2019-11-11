import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from .BasePIFuNet import BasePIFuNet
from .MLP import MLP
from .DepthNormalizer import DepthNormalizer
from .HGFilters import *
from .VolumetricEncoder import *
from ..net_util import init_net

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

        self.opt = opt
        self.num_views = self.opt.num_views
        self.image_filter = HGFilter(opt.num_stack, opt.hg_depth, opt.hg_dim, 
                                     opt.norm, opt.hg_down, False, False, 1)

        self.mlp = MLP(
            filter_channels=self.opt.mlp_dim,
            merge_layer=-1,
            num_views=1,
            res_layers=self.opt.mlp_res_layers,
            norm=self.opt.mlp_norm,
            last_op=nn.Sigmoid(),
            compose=False)

        self.im_feat_list = []
        self.tmpx = None
        self.normx = None
        self.w = None

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

    def filter_local(self, images):
        '''
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B1, B2, C, H, W]
        '''
        self.im_feat_list, self.normx = self.image_filter(images.view(-1,*images.size()[2:]))
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]
        
    def query(self, points, calib_local, calib_global=None, transforms=None, labels=None):
        '''
        given 3d points, we obtain 2d projection of these given the camera matrices.
        filter needs to be called beforehand.
        the prediction is stored to self.preds
        args:
            points: [B1, 3, N] 3d points in world space
            calibs_local: [B1, B2, 4, 4] calibration matrices for each image
            calibs_global: [B1, 4, 4] calibration matrices for each image
            transforms: [B1, 2, 3] image space coordinate transforms
            labels: [B1, C, N] ground truth labels (for supervision only)
        return:
            [B, C, N] prediction
        '''
        if calib_global is not None:
            B = calib_local.size(1)
            xyz = self.projection(points[:,None].expand(-1,B,-1,-1).reshape(-1,*points.size()[1:]), calib_local.view(-1,4,4), transforms)
            xy = xyz[:, :2, :]

            # if the point is outside bounding box, return outside.
            in_bb = (xyz >= -1) & (xyz <= 1)
            in_bb = in_bb[:, 0, :] & in_bb[:, 1, :]
            in_bb = in_bb[:, None, :].detach().float()

            self.netG.query(points=points, calibs=calib_global, labels=labels)

            if labels is not None:
                self.labels = in_bb * labels[:,None].expand(-1,B,-1,-1).reshape(-1,*labels.size()[1:])
                with torch.no_grad():
                    self.w = in_bb.size(2) / in_bb.view(in_bb.size(0),-1).sum(1)
                    self.gamma = 1 - self.labels.view(self.labels.size(0),-1).sum(1) / in_bb.view(in_bb.size(0),-1).sum(1)
            self.intermediate_preds_list = []

            z_feat = self.netG.phi
            if not self.opt.train_full_pifu:
                z_feat = z_feat.detach()
            z_feat = z_feat[:,None].expand(-1,B,-1,-1).reshape(-1,*z_feat.size()[1:])

            for i, im_feat in enumerate(self.im_feat_list):
                point_local_feat_list = [self.index(im_feat, xy), z_feat]
                point_local_feat = torch.cat(point_local_feat_list, 1)
                pred = self.mlp(point_local_feat)[0]
                pred = in_bb * pred

                self.intermediate_preds_list.append(pred)
            
            self.preds = self.intermediate_preds_list[-1]
        else:
            calib_global = calib_local.view(-1,4,4)
            xyz = self.projection(points, calib_local.view(-1,4,4), transforms)
            xy = xyz[:, :2, :]

            # if the point is outside bounding box, return outside.
            in_bb = (xyz >= -1) & (xyz <= 1)
            in_bb = in_bb[:, 0, :] & in_bb[:, 1, :]
            in_bb = in_bb[:, None, :].detach().float()

            self.netG.query(points=points, calibs=calib_global, labels=labels)

            if labels is not None:
                self.labels = in_bb * labels
                with torch.no_grad():
                    self.gamma = 1 - labels.view(labels.size(0),-1).sum(1) / in_bb.view(in_bb.size(0),-1).sum(1)

            self.intermediate_preds_list = []

            z_feat = self.netG.phi
            if not self.opt.train_full_pifu:
                z_feat = z_feat.detach()

            for i, im_feat in enumerate(self.im_feat_list):
                point_local_feat_list = [self.index(im_feat, xy), z_feat]
                point_local_feat = torch.cat(point_local_feat_list, 1)
                pred = self.mlp(point_local_feat)[0]
                pred = in_bb * pred

                self.intermediate_preds_list.append(pred)
            
            self.preds = self.intermediate_preds_list[-1]

    def calc_normal(self, points, calib_local, calib_global, transforms=None, labels=None, delta=0.001, fd_type='forward'):
        '''
        return surface normal in 'model' space.
        it computes normal only in the last stack.
        note that the current implementation use forward difference.
        args:
            points: [B1, B2, 3, N] 3d points in world space
            calibs_local: [B1, B2, 3, 4] calibration matrices for each image
            calibs_global: [B1, 3, 4] calibration matrices for each image
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

    def get_error(self, gamma):
        '''
        return the loss given the ground truth labels and prediction
        '''
        error = {}
        if self.opt.train_full_pifu:
            error = self.netG.get_error(gamma)
            error['Err(occ:fine)'] = 0
            for preds in self.intermediate_preds_list:
                error['Err(occ:fine)'] += self.criteria['occ'](preds, self.labels, self.gamma, self.w)     
            error['Err(occ:fine)'] /= len(self.intermediate_preds_list)
            if self.nmls is not None and self.labels_nml is not None:
                error['Err(nml:fine)'] = self.criteria['nml'](self.nmls, self.labels_nml)
        else:
            error['Err(occ)'] = 0
            for preds in self.intermediate_preds_list:
                error['Err(occ)'] += self.criteria['occ'](preds, self.labels, self.gamma, self.w)
            
            error['Err(occ)'] /= len(self.intermediate_preds_list)
            if self.nmls is not None and self.labels_nml is not None:
                error['Err(nml)'] = self.criteria['nml'](self.nmls, self.labels_nml)
        
        return error

    def forward(self, images_local, images_global, points, calib_local, calib_global, labels, gamma, points_nml=None, labels_nml=None):
        self.filter_global(images_global)
        self.filter_local(images_local)
        self.query(points, calib_local, calib_global, labels=labels)
        if points_nml is not None and labels_nml is not None:
            self.calc_normal(points_nml, calib_local, calib_global, labels=labels_nml)
        res = self.get_preds()
            
        err = self.get_error(gamma)

        return err, res
