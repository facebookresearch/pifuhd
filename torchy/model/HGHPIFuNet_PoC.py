import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from .BasePIFuNet import BasePIFuNet
from .MLP import MLP
from .DepthNormalizer import DepthNormalizer
from .VolumetricEncoder import *
from .HGFilters_PoC import *
from ..net_util import init_net

class HGHPIFuNet(BasePIFuNet):
    '''
    HGPIFu uses stacked hourglass as an image encoder.
    '''

    def __init__(self, 
                 opt, 
                 projection_mode='orthogonal',
                 criteria={'occ': nn.MSELoss()}
                 ):
        super(HGHPIFuNet, self).__init__(
            projection_mode=projection_mode,
            criteria=criteria)

        self.name = 'hg_pifu'

        self.opt = opt
        self.num_views = self.opt.num_views
        self.image_filter = HGHFilter(opt.num_stack, opt.hg_depth, opt.hg_dim, 32,
                                     opt.norm, opt.hg_down, False, opt.hg_use_attention, opt.n_pixshuffle)

        self.mlp = MLP(
            filter_channels=self.opt.mlp_dim,
            merge_layer=self.opt.merge_layer,
            num_views=self.num_views,
            res_layers=self.opt.mlp_res_layers,
            norm=self.opt.mlp_norm,
            last_op=nn.Sigmoid(),
            compose=self.opt.use_compose)

        second_mlp = [self.opt.mlp_dim[self.opt.merge_layer+1] + 32] + self.opt.mlp_dim[1:]
        self.mlp2 = MLP(
            filter_channels=second_mlp,
            num_views=self.num_views,
            res_layers=self.opt.mlp_res_layers,
            norm=self.opt.mlp_norm,
            last_op=nn.Sigmoid(),
            compose=self.opt.use_compose)

        if self.opt.sp_enc_type == 'vol':
            self.spatial_enc = VolumetricEncoder(opt)
        elif self.opt.sp_enc_type == 'z':
            self.spatial_enc = DepthNormalizer(opt)
        else:
            raise NameError('unknown spatial encoding type')

        self.im_feat_list = []
        self.tmpx = None
        self.normx = None
        self.phi = None

        self.intermediate_preds_list = []

        init_net(self)
    
    def filter(self, images):
        '''
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B, C, H, W]
        '''
        if self.opt.sp_enc_type == 'vol':
            self.spatial_enc.filter(images)

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

        #     for i in range(xyz.size(0)):
        #         p = xyz[i].detach().cpu().numpy().T
        #         v = labels[i].detach().cpu().numpy().T

        #         cin = np.ones_like(p[v[:,0] > 0.5])
        #         cin[:,1:] = 0.0
        #         save_points_color('%04d_in.obj' % i, p[v[:,0] > 0.5], cin)
        #         cin = np.ones_like(p[v[:,0] <= 0.5])
        #         cin[:,:2] = 0.0
        #         save_points_color('%04d_out.obj' % i, p[v[:,0] <= 0.5], cin)

        # exit()

        sp_feat = self.spatial_enc(xyz, calibs=calibs)

        intermediate_preds_list = []

        phi = None
        for i, im_feat in enumerate(self.im_feat_list):

            if self.opt.sp_enc_type == 'vol' and self.opt.sp_no_pifu:
                point_local_feat = sp_feat
            # elif self.opt.imfeat_norm: # experimental
            #     point_local_feat_list = [F.normalize(self.index(im_feat, xy),dim=1,eps=1e-8), sp_feat]            
            #     point_local_feat = torch.cat(point_local_feat_list, 1)
            else:
                point_local_feat_list = [self.index(im_feat, xy), sp_feat]            
                point_local_feat = torch.cat(point_local_feat_list, 1)
            pred, phi = self.mlp(point_local_feat)
            pred = in_bb * pred
            
            intermediate_preds_list.append(pred)

        if update_phi:
            self.phi = phi
        point_local_feat_h = torch.cat([self.index(self.normx, xy), phi.detach()], 1)

        if update_pred:        
            self.intermediate_preds_list = intermediate_preds_list
            self.preds = self.mlp2(point_local_feat_h)[0]

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

        if self.opt.sp_enc_type == 'vol_enc' and self.opt.sp_no_pifu:
            point_local_feat = sp_feat
        else:
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

    # bilinear sampling doesn't support gradient backprop
    # def calc_normal(self, points, calibs, transforms=None, labels=None):
    #     '''
    #     return surface normal in 'model' space.
    #     it computes normal only in the last stack.
    #     note that the current implementation use forward difference.
    #     args:
    #         points: [B, 3, N] 3d points in world space
    #         calibs: [B, 3, 4] calibration matrices for each image
    #         transforms: [B, 2, 3] image space coordinate transforms
    #         delta: perturbation for finite difference
    #         fd_type: finite difference type (forward/backward/central) 
    #     '''
    #     if labels is not None:
    #         self.labels_nml = labels

    #     points.requires_grad = True
    #     xyz = self.projection(points, calibs, transforms)
    #     xy = xyz[:, :2, :]

    #     im_feat = self.im_feat_list[-1]
    #     sp_feat = self.spatial_enc(xyz, calibs=calibs)

    #     if self.opt.sp_enc_type == 'vol' and self.opt.sp_no_pifu:
    #         point_local_feat = sp_feat
    #     else:
    #         point_local_feat_list = [self.index(im_feat, xy), sp_feat]            
    #         point_local_feat = torch.cat(point_local_feat_list, 1)
            
    #     pred = self.mlp(point_local_feat)

    #     pred_sum = pred.sum()
        
    #     nml = -torch.autograd.grad(pred_sum, points, create_graph=True)[0]
    #     nml = F.normalize(nml, dim=1, eps=1e-8)

    #     self.nmls = nml

    def calc_comp_ids(self, points, calibs, transforms=None):
        '''
            return the component id
            NOTE: this is valid only for mlp with compose=True
        '''
        self.query(points, calibs)
        self.get_preds()

        nways = None
        if self.mlp.y_nways is not None:
            nways = self.mlp.y_nways.max(1)[1]

        return nways


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
        error['Err(occ)'] += self.criteria['occ'](self.preds, self.labels, gamma)        
        error['Err(occ)'] /= len(self.intermediate_preds_list)+1
        
        if self.nmls is not None and self.labels_nml is not None:
            error['Err(nml)'] = self.criteria['nml'](self.nmls, self.labels_nml)
        
        if self.mlp.y_nways is not None and self.opt.lambda_cmp_l1 != 0.0:
            error['Err(L1)'] = self.mlp.y_nways.abs().sum(1).mean()

        return error

    def forward(self, images, points, calibs, labels, gamma, points_nml=None, labels_nml=None):
        self.filter(images)
        self.query(points, calibs, labels=labels)
        if points_nml is not None and labels_nml is not None:
            self.calc_normal(points_nml, calibs, labels=labels_nml)
        res = self.get_preds()
            
        err = self.get_error(gamma)

        return err, res
