import torch
import torch.nn as nn 
import torch.nn.functional as F 
from ..net_util import *

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm='batch'):
        super(ConvBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if norm == 'batch':
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
            self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
            self.bn4 = nn.BatchNorm2d(in_planes)
        elif norm == 'group':
            self.bn1 = nn.GroupNorm(32, in_planes)
            self.bn2 = nn.GroupNorm(32, int(out_planes / 2))
            self.bn3 = nn.GroupNorm(32, int(out_planes / 4))
            self.bn4 = nn.GroupNorm(32, in_planes)
        
        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                self.bn4,
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None
    
    def forward(self, x):
        residual = x

        out1 = self.conv1(F.relu(self.bn1(x), True))
        out2 = self.conv2(F.relu(self.bn2(out1), True))
        out3 = self.conv3(F.relu(self.bn3(out2), True))

        out3 = torch.cat([out1, out2, out3], 1)

        if self.downsample is not None:
            residual = self.downsample(residual)
        
        out3 += residual

        return out3

class HourGlass(nn.Module):
    def __init__(self, depth, n_features, norm='batch'):
        super(HourGlass, self).__init__()
        self.depth = depth
        self.features = n_features
        self.norm = norm

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))
        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

    def _forward(self, level, inp):
        # upper branch
        up1 = inp 
        up1 = self._modules['b1_' + str(level)](up1)

        # lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='bilinear')

        return up1 + up2
    
    def forward(self, x):
        return self._forward(self.depth, x)
        

class HGFilter(nn.Module):
    def __init__(self, stack, depth, last_ch, norm='batch', down_type='conv64', use_sigmoid=True, use_attention=False):
        super(HGFilter, self).__init__()
        self.n_stack = stack
        self.use_sigmoid = use_sigmoid
        self.use_attention = use_attention
        self.depth = depth
        self.last_ch = last_ch
        self.norm = norm
        self.down_type = down_type

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

        if self.norm == 'batch':
            self.bn1 = nn.BatchNorm2d(64)
        elif self.norm == 'group':
            self.bn1 = nn.GroupNorm(32, 64)

        if self.down_type == 'conv64':
            self.conv2 = ConvBlock(64, 64, self.norm)
            self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        elif self.down_type == 'conv128':
            self.conv2 = ConvBlock(128, 128, self.norm)
            self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        elif self.down_type == 'ave_pool':
            self.conv2 = ConvBlock(64, 128, self.norm)
        
        self.conv3 = ConvBlock(128, 128, self.norm)
        self.conv4 = ConvBlock(128, 256, self.norm)
        
        # start stacking
        for stack in range(self.n_stack):
            self.add_module('m' + str(stack), HourGlass(self.depth, 256, self.norm))

            self.add_module('top_m_' + str(stack), ConvBlock(256, 256, self.norm))
            self.add_module('conv_last' + str(stack),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            if self.norm == 'batch':
                self.add_module('bn_end' + str(stack), nn.BatchNorm2d(256))
            elif self.norm == 'group':
                self.add_module('bn_end' + str(stack), nn.GroupNorm(32, 256))
            
            self.add_module('l' + str(stack),
                            nn.Conv2d(256, self.last_ch if not self.use_attention else self.last_ch + 1, 
                            kernel_size=1, stride=1, padding=0))
            
            if stack < self.n_stack - 1:
                self.add_module(
                    'bl' + str(stack), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module(
                    'al' + str(stack), nn.Conv2d(self.last_ch, 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)

        if self.down_type == 'ave_pool':
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        elif self.down_type == ['conv64', 'conv128']:
            x = self.conv2(x)
            x = self.down_conv2(x)
        else:
            raise NameError('unknown downsampling type')
    
        normx = x

        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        for i in range(self.n_stack):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                       (self._modules['conv_last' + str(i)](ll)), True)

            tmp_out = self._modules['l' + str(i)](ll)
            if self.use_attention:
                tmp_out = nn.Sigmoid()(tmp_out[:,-1:]) * tmp_out[:,:-1]

            if self.use_sigmoid:
                outputs.append(nn.Tanh()(tmp_out))
            else:
                outputs.append(tmp_out)
            
            if i < self.n_stack - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_
            
        return outputs, normx
    