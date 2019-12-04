import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
from .MLP import MLP
from .DepthNormalizer import DepthNormalizer
from .ConvFilters import *
from ..net_util import init_net

class ResNetPIFuNet(BasePIFuNet):
    def __init__(self,
                 opt,
                 projection_mode='orthogonal',
                 criteria={'occ': nn.MSELoss()}
                 ):
        super(ResNetPIFuNet, self).__init__(
            projection_mode=projection_mode,
            criteria=criteria)

        self.name = 'resnet_pifu'

        self.opt = opt

        self.image_filter = ResNet('resnet34')

        self.mlp = MLP(
            filter_channels=self.opt.mlp_dim,
            merge_layer=-1,
            res_layers=self.opt.mlp_res_layers,
            norm=self.opt.mlp_norm,
            last_op=nn.Sigmoid())

        self.spatial_enc = DepthNormalizer(opt)

        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []

        init_net(self)

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        self.im_feat_list = self.image_filter(images)

    def query(self, points, calibs, transforms=None, labels=None, update_pred=True):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
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

        # This is a list of [B, Feat_i, N] features
        point_local_feat_list = [self.index(im_feat, xy) for im_feat in self.im_feat_list]
        # point_local_feat_list.append(self.global_feat.expand(-1,-1,sp_feat.size(2)))

        point_local_feat = torch.cat(point_local_feat_list + [sp_feat], 1)

        self.preds = self.mlp(point_local_feat)[0]


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

        sp_feat = self.spatial_enc(xyz, calibs=calibs)

        # This is a list of [B, Feat_i, N] features
        point_local_feat_list = [self.index(im_feat, xy) for im_feat in self.im_feat_list]

        point_local_feat = torch.cat(point_local_feat_list + [sp_feat], 1)

        pred = self.mlp(point_local_feat)[0]

        pred = pred.view(*pred.size()[:2],-1,4) # (B, 1, N, 4)

        # divide by delta is omitted since it's normalized anyway
        dfdx = pred[:,:,:,1] - pred[:,:,:,0]
        dfdy = pred[:,:,:,2] - pred[:,:,:,0]
        dfdz = pred[:,:,:,3] - pred[:,:,:,0]

        nml = -torch.cat([dfdx,dfdy,dfdz], 1)
        nml = F.normalize(nml, dim=1, eps=1e-8)

        self.nmls = nml

    def get_error(self, gamma):
        '''
        return the loss given the ground truth labels and prediction
        '''
        error = {}
        error['Err(occ)'] = self.criteria['occ'](self.preds, self.labels, gamma)

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
