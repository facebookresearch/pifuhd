import torch
import torch.nn as nn 
import torch.nn.functional as F 

def conv3x3(in_planes, out_planes, strd=1, padding=1, axis=0, bias=False):
    kernel = (1, 1, 1)
    kernel[axis % 3] = 3
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel,
                     stride=strd, padding=padding, bias=bias)

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm='batch'):
        super(ConvBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, int(out_planes / 2), axis=0)
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4), axis=1)
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4), axis=2)

        if norm == 'batch':
            self.bn1 = nn.BatchNorm3d(in_planes)
            self.bn2 = nn.BatchNorm3d(int(out_planes / 2))
            self.bn3 = nn.BatchNorm3d(int(out_planes / 4))
            self.bn4 = nn.BatchNorm3d(in_planes)
        elif norm == 'group':
            self.bn1 = nn.GroupNorm(32, in_planes)
            self.bn2 = nn.GroupNorm(32, int(out_planes / 2))
            self.bn3 = nn.GroupNorm(32, int(out_planes / 4))
            self.bn4 = nn.GroupNorm(32, in_planes)
        
        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                self.bn4,
                nn.ReLU(True),
                nn.Conv3d(in_planes, out_planes,
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

class HourGlass3D(nn.Module):
    def __init__(self, depth, n_features, norm='batch'):
        super(HourGlass3D, self).__init__()
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
        low1 = F.avg_pool3d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='blinear')

        return up1 + up2
    
    def forward(self, x):
        return self._forward(self.depth, x)

class HGFilter3D(nn.Module):
    def __init__(self, in_ch, stack, depth, last_ch, norm='batch', down_type='conv64', use_sigmoid=True):
        super(HGFilter3D, self).__init__()
        self.n_stack = stack
        self.use_sigmoid = use_sigmoid
        self.depth = depth
        self.last_ch = last_ch
        self.norm = norm
        self.down_type = down_type

        self.preds = None

        self.conv1 = ConvBlock(in_ch, 64, self.norm)
        
        # start stacking
        for stack in range(self.n_stack):
            self.add_module('m' + str(stack), HourGlass3D(self.depth, 64, self.norm))

            self.add_module('top_m_' + str(stack), ConvBlock(64, 64, self.norm))
            self.add_module('conv_last' + str(stack),
                            nn.Conv3d(64, 64, kernel_size=1, stride=1, padding=0))
            if self.norm == 'batch':
                self.add_module('bn_end' + str(stack), nn.BatchNorm3d(64))
            elif self.norm == 'group':
                self.add_module('bn_end' + str(stack), nn.GroupNorm(32, 64))
            
            self.add_module('l' + str(stack),
                            nn.Conv3d(64, self.last_ch, kernel_size=1, stride=1, padding=0))
            
            if stack < self.n_stack - 1:
                self.add_module(
                    'bl' + str(stack), nn.Conv3d(64, 64, kernel_size=1, stride=1, padding=0))
                self.add_module(
                    'al' + str(stack), nn.Conv3d(self.last_ch, 64, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = self.conv1(x)

        previous = x

        outputs = []
        for i in range(self.n_stack):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                       (self._modules['conv_last' + str(i)](ll)), True)

            tmp_out = self._modules['l' + str(i)](ll)
            if self.use_sigmoid:
                outputs.append(nn.Tanh()(tmp_out))
            else:
                outputs.append(tmp_out)
            
            if i < self.n_stack - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_
        
        self.preds = outputs
            
        return outputs[-1]
    