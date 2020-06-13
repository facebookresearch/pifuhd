'''
MIT License

Copyright (c) 2019 Shunsuke Saito, Zeng Huang, and Ryota Natsume

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F 
import functools

def load_state_dict(state_dict, net):
    model_dict = net.state_dict()

    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}                    

    for k, v in pretrained_dict.items():                      
        if v.size() == model_dict[k].size():
            model_dict[k] = v

    not_initialized = set()
            
    for k, v in model_dict.items():
        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
            not_initialized.add(k.split('.')[0])
    
    print('not initialized', sorted(not_initialized))
    net.load_state_dict(model_dict) 

    return net
    
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

class CustomBCELoss(nn.Module):
    def __init__(self, brock=False, gamma=None):
        super(CustomBCELoss, self).__init__()
        self.brock = brock
        self.gamma = gamma

    def forward(self, pred, gt, gamma, w=None):
        x_hat = torch.clamp(pred, 1e-5, 1.0-1e-5) # prevent log(0) from happening
        gamma = gamma[:,None,None] if self.gamma is None else self.gamma
        if self.brock:
            x = 3.0*gt - 1.0 # rescaled to [-1,2]

            loss = -(gamma*x*torch.log(x_hat) + (1.0-gamma)*(1.0-x)*torch.log(1.0-x_hat))
        else:
            loss = -(gamma*gt*torch.log(x_hat) + (1.0-gamma)*(1.0-gt)*torch.log(1.0-x_hat))

        if w is not None:
            if len(w.size()) == 1:
                w = w[:,None,None] 
            return (loss * w).mean()
        else:
            return loss.mean()

class CustomMSELoss(nn.Module):
    def __init__(self, gamma=None):
        super(CustomMSELoss, self).__init__()
        self.gamma = gamma

    def forward(self, pred, gt, gamma, w=None):
        gamma = gamma[:,None,None] if self.gamma is None else self.gamma
        weight = gamma * gt + (1.0-gamma) * (1 - gt)
        loss = (weight * (pred - gt).pow(2)).mean()

        if w is not None:
            return (loss * w).mean()
        else:
            return loss.mean()

def createMLP(dims, norm='bn', activation='relu', last_op=nn.Tanh(), dropout=False):
    act = None
    if activation == 'relu':
        act = nn.ReLU()
    if activation == 'lrelu':
        act = nn.LeakyReLU()
    if activation == 'selu':
        act = nn.SELU()
    if activation == 'elu':
        act = nn.ELU()
    if activation == 'prelu':
        act = nn.PReLU()

    mlp = []
    for i in range(1,len(dims)):
        if norm == 'bn':
            mlp += [  nn.Linear(dims[i-1], dims[i]),
                    nn.BatchNorm1d(dims[i])]
        if norm == 'in':
            mlp += [  nn.Linear(dims[i-1], dims[i]),
                    nn.InstanceNorm1d(dims[i])]
        if norm == 'wn':
            mlp += [  nn.utils.weight_norm(nn.Linear(dims[i-1], dims[i]), name='weight')]
        if norm == 'none':
            mlp += [ nn.Linear(dims[i-1], dims[i])]
        
        if i != len(dims)-1:
            if act is not None:
                mlp += [act]
            if dropout:
                mlp += [nn.Dropout(0.2)]

    if last_op is not None:
        mlp += [last_op]

    return mlp