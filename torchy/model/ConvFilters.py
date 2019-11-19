import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet
import torchvision.models.vgg as vgg


class ResNet(nn.Module):
    def __init__(self, model='resnet18'):
        super(ResNet, self).__init__()

        if model == 'resnet18':
            net = resnet.resnet18(pretrained=True)
        elif model == 'resnet34':
            net = resnet.resnet34(pretrained=True)
        elif model == 'resnet50':
            net = resnet.resnet50(pretrained=True)
        else:
            raise NameError('Unknown Fan Filter setting!')

        self.conv1 = net.conv1

        self.pool = net.maxpool
        self.layer0 = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, image):
        '''
        :param image: [BxC_inxHxW] tensor of input image
        :return: list of [BxC_outxHxW] tensors of output features
        '''

        y = image
        feat_pyramid = []
        y = self.layer0(y)
        feat_pyramid.append(y)
        y = self.layer1(self.pool(y))
        feat_pyramid.append(y)
        y = self.layer2(y)
        feat_pyramid.append(y)
        y = self.layer3(y)
        feat_pyramid.append(y)
        y = self.layer4(y)
        feat_pyramid.append(y)

        return feat_pyramid
