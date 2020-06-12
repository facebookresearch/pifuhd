# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F 

from ..geometry import index, orthogonal, perspective

class BasePIFuNet(nn.Module):
    def __init__(self,
                 projection_mode='orthogonal',
                 criteria={'occ': nn.MSELoss()},
                 ):
        '''
        args:
            projection_mode: orthonal / perspective
            error_term: point-wise error term 
        '''
        super(BasePIFuNet, self).__init__()
        self.name = 'base'

        self.criteria = criteria

        self.index = index
        self.projection = orthogonal if projection_mode == 'orthogonal' else perspective

        self.preds = None
        self.labels = None
        self.nmls = None
        self.labels_nml = None
        self.preds_surface = None # with normal loss only

    def forward(self, points, images, calibs, transforms=None):
        '''
        args:
            points: [B, 3, N] 3d points in world space
            images: [B, C, H, W] input images
            calibs: [B, 3, 4] calibration matrices for each image
            transforms: [B, 2, 3] image space coordinate transforms
        return:
            [B, C, N] prediction corresponding to the given points
        '''
        self.filter(images)
        self.query(points, calibs, transforms)
        return self.get_preds()

    def filter(self, images):
        '''
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B, C, H, W]
        '''
        None
    
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
        None

    def calc_normal(self, points, calibs, transforms=None, delta=0.1):
        '''
        return surface normal in 'model' space.
        it computes normal only in the last stack.
        note that the current implementation use forward difference.
        args:
            points: [B, 3, N] 3d points in world space
            calibs: [B, 3, 4] calibration matrices for each image
            transforms: [B, 2, 3] image space coordinate transforms
            delta: perturbation for finite difference
        '''
        None

    def get_preds(self):
        '''
        return the current prediction.
        return:
            [B, C, N] prediction
        '''
        return self.preds

    def get_error(self, gamma=None):
        '''
        return the loss given the ground truth labels and prediction
        '''
        return self.error_term(self.preds, self.labels, gamma)

    