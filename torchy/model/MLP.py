import torch
import torch.nn as nn
import torch.nn.functional as F 

class MLP(nn.Module):
    def __init__(self, 
                 filter_channels, 
                 num_views=1, 
                 merge_layer=0,
                 res_layers=[],
                 norm='group',
                 last_op=None,
                 compose=False):
        super(MLP, self).__init__()

        self.filters = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.num_views = num_views
        self.merge_layer = merge_layer if merge_layer > 0 else len(filter_channels) // 2
        self.res_layers = res_layers
        self.norm = norm
        self.last_op = last_op
        self.compose = compose
        self.y_nways = None # only for part composition

        for l in range(0, len(filter_channels)-1):
            if l in self.res_layers:
                self.filters.append(nn.Conv1d(
                    filter_channels[l] + filter_channels[0],
                    filter_channels[l+1],
                    1))
            else:
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l+1],
                    1))
            if l != len(filter_channels)-2:
                if norm == 'group':
                    self.norms.append(nn.GroupNorm(32, filter_channels[l+1]))
                elif norm == 'batch':
                    self.norms.append(nn.BatchNorm1d(filter_channels[l+1]))

    def forward(self, feature):
        '''
        feature may include multiple view inputs
        args:
            feature: [B, C_in, N]
        return:
            [B, C_out, N] prediction
        '''
        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            y = f(
                y if i not in self.res_layers
                else torch.cat([y, tmpy], 1)
            )
            if i != len(self.filters)-1:
                if self.norm not in ['batch', 'group']:
                    y = F.leaky_relu(y)
                else:
                    y = F.leaky_relu(self.norms[i](y))         
            if self.num_views > 1 and i == self.merge_layer:
                y = y.view(
                    -1, self.num_views, *y.size()[1:]
                ).mean(dim=1)
                tmpy = feature.view(
                    -1, self.num_views, *feature.size()[1:]
                ).mean(dim=1)

        if self.last_op is not None:
            y = self.last_op(y)

        if self.compose and y.size(1) != 1:
            self.y_nways = y
            y = y.max(dim=1, keepdim=True)[0]

        return y

class ResBlock(nn.Module):
    def __init__(self, in_ch):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_ch, in_ch, 1)
        self.conv2 = nn.Conv1d(in_ch, in_ch, 1)
        self.conv3 = nn.Conv1d(in_ch, in_ch, 1)

    def forward(self, y):
        tmp_y = y
        y = self.conv1(F.relu(y))
        y = self.conv2(F.relu(y))

        return self.conv3(y + tmp_y)

class MLPResBlock(nn.Module):
    def __init__(self, 
                 in_ch, 
                 out_ch,
                 n_block, 
                 last_op=None):
        super(MLPResBlock, self).__init__()

        self.last_op = last_op

        self.filters = nn.ModuleList()
        for i in range(n_block):
            self.filters.append(ResBlock(512))
        
        self.conv1 = nn.Conv1d(in_ch, 512, 1)
        self.conv_last = nn.Conv1d(512, out_ch, 1)

    def forward(self, feature):
        '''
        feature may include multiple view inputs
        args:
            feature: [B, C_in, N]
        return:
            [B, C_out, N] prediction
        '''
        y = feature
        y = self.conv1(y)

        for f in self.filters:
            y = f(y)

        y = self.conv_last(F.relu(y))

        if self.last_op is not None:
            y = self.last_op(y)

        return y