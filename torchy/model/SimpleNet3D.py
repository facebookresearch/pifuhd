import torch
import torch.nn as nn 
import torch.nn.functional as F 

class DownConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm='batch'):
        super(DownConv3D, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=4,
                        stride=2, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, True)
        if norm == 'batch':
            self.downnorm = nn.BatchNorm3d(out_ch)
        elif norm == 'group':
            self.downnorm = nn.GroupNorm(16, out_ch)

    def forward(self, x):
        return self.relu(self.downnorm(self.conv(x)))

class Simple3DNet(nn.Module):
    def __init__(self, in_ch, out_ch, out_res, norm='batch'):
        super(Simple3DNet, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.out_res = out_res

        # assuming it's 32x32x32
        self.conv1 = DownConv3D(in_ch, in_ch*2, norm) # 16x16x16
        self.conv2 = DownConv3D(in_ch*2, in_ch*4, norm) # 8x8x8
        self.conv3 = DownConv3D(in_ch*4, in_ch*4, norm) # 4x4x4

        self.pool = nn.AvgPool3d(2)
        
        self.conv_last = nn.Conv3d(in_ch*4, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        B = x.size(0)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.pool(x)
        x = self.conv_last(x)

        return x
