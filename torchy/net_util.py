import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F 
import functools

# if F.normalize suffers from instability, try this
def normalize(x, dim, eps=1e-10):
    xsq = torch.sum(x.pow(2),dim,keepdim=True).sqrt() + eps
    x = x / xsq.expand_as(x).contiguous().detach() # NOTE: without detach, grad tends to blow up
    return x

def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm, 32)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

class CustomBCELoss(nn.Module):
    def __init__(self, brock=False):
        super(CustomBCELoss, self).__init__()
        self.brock = brock

    def forward(self, pred, gt, ratio=0.7):
        if self.brock:
            x_hat = torch.clamp(pred, 1e-7, 1.0-1e-7) # prevent log(0) from happening
            x = 3.0*gt - 1.0 # rescaled to [-1,2]

            loss = -(ratio*x*torch.log(x_hat) + (1.0-ratio)*(1.0-x)*torch.log(1.0-x_hat))
        else:
            loss = -(ratio*gt*torch.log(pred) + (1.0-ratio)*(1.0-gt)*torch.log(1.0-pred))

        return loss.mean()
