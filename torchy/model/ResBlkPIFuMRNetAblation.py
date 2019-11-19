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
from ..networks import define_G
from torchvision import models


def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob > 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )



class ResBlkPIFuMRNetAblation(BasePIFuNet):
    '''
    HGPIFu uses stacked hourglass as an image encoder.
    '''

    def __init__(self, 
                 opt, 
                 projection_mode='orthogonal',
                 criteria={'occ': nn.MSELoss()}
                 ):
        super(ResBlkPIFuMRNetAblation, self).__init__(
            projection_mode=projection_mode,
            criteria=criteria)

        self.name = 'resblk_pifu_mr'

        in_ch = 3

        self.opt = opt
        self.num_views = self.opt.num_views
        self.image_filter = define_G(in_ch, opt.hg_dim, 32, "global", 4, 5, 1, 3, "instance", last_op=None)

        self.mlp = MLP(
            filter_channels=self.opt.mlp_dim,
            merge_layer=-1,
            num_views=1,
            res_layers=self.opt.mlp_res_layers,
            norm=self.opt.mlp_norm,
            last_op=nn.Sigmoid(),
            compose=False)


        if self.opt.sp_enc_type == 'z':
            self.spatial_enc = DepthNormalizer(opt)
        else:
            raise NameError('unknown spatial encoding type')


        self.im_feat = None
        self.preds_low = None
        self.gamma = None
        self.w = None

        init_net(self)

        self.global_feat = None
        if 'resnet' in self.opt.netG:
            resnet = models.resnet34(pretrained=True)
            layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.netG = nn.Sequential(layer0, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4, nn.AdaptiveAvgPool2d(1))
        else:
            self.netG = None


    def train(self, mode=True):
        r"""Sets the module in training mode."""      
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def filter_global(self, images):
        '''
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B1, C, H, W]
        '''
        if self.netG is not None:
            images = nn.Upsample(size=(256, 256), mode='bilinear')(images)
            self.global_feat = self.netG(images).view(images.size(0),-1,1)


    def filter_local(self, images, rect=None):
        '''
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B1, B2, C, H, W]
            rect: [B1, B2, 4]
        '''            
        self.im_feat = self.image_filter(images.view(-1,*images.size()[2:]))
        
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
        preds_low = []
        gammas = []
        newlabels = []

        im_feat = self.im_feat.view(-1,B,*self.im_feat.size()[1:])

        for i in range(B):
            xyz = self.projection(points[:,i], calib_local[:,i], transforms)
            
            xy = xyz[:, :2, :]

            # if the point is outside bounding box, return outside.
            with torch.no_grad():
                in_bb = (xyz >= -1) & (xyz <= 1)
                in_bb = in_bb[:, 0, :] & in_bb[:, 1, :]
                in_bb = in_bb[:, None, :].detach().float()

            z_feat = self.spatial_enc(xyz, calibs=calib_local[:,i])
            if self.global_feat is not None:
                z_feat = torch.cat([z_feat, self.global_feat.expand(-1,-1,z_feat.size(2))],1)

            if labels is not None:
                newlabels.append(in_bb * labels[:,i])
                with torch.no_grad():
                    ws.append(in_bb.size(2) / in_bb.view(in_bb.size(0),-1).sum(1))
                    gammas.append(1 - newlabels[-1].view(newlabels[-1].size(0),-1).sum(1) / in_bb.view(in_bb.size(0),-1).sum(1))

            point_local_feat_list = [self.index(im_feat[:,i], xy), z_feat]
            point_local_feat = torch.cat(point_local_feat_list, 1)
            pred = self.mlp(point_local_feat)[0]
            preds.append(in_bb * pred)

        self.preds = torch.cat(preds,0)
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

        im_feat = self.im_feat.view(-1,B,*self.im_feat.size()[1:])

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

            z_feat = self.spatial_enc(xyz, calibs=calib_local[:,i])
            if self.global_feat is not None:
                z_feat = torch.cat([z_feat, self.global_feat.expand(-1,-1,z_feat.size(2))],1)

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
            error['Err(occ:fine)'] = self.criteria['occ'](self.preds, self.labels, self.gamma, self.w)     
            if self.nmls is not None and self.labels_nml is not None:
                error['Err(nml:fine)'] = self.criteria['nml'](self.nmls, self.labels_nml)
        else:
            error['Err(occ:fine)'] = self.criteria['occ'](self.preds, self.labels, self.gamma, self.w)                 
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
