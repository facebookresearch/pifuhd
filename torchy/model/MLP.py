import torch
import torch.nn as nn
import torch.nn.functional as F 

class MLP(nn.Module):
    def __init__(self, 
                 filter_channels, 
                 num_views=1, 
                 merge_layer=0,
                 no_residual=True, 
                 last_op=None):
        super(MLP, self).__init__()

        self.filters = []
        self.num_views = num_views
        self.merge_layer = merge_layer if merge_layer > 0 else len(filter_channels) // 2
        self.no_residual = no_residual
        self.last_op = last_op

        if self.no_residual:
            for l in range(0, len(filter_channels)-1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l+1],
                    1))
                self.add_module('conv%d' % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels)-1):
                if 0 != l:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l] + filter_channels[0],
                        filter_channels[l+1],
                        1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l+1],
                        1))
                self.add_module('conv%d' % l, self.filters[l])

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
            if self.no_residual:
                y = f(y)
            else:
                y = f(
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            if i != len(self.filters)-1:
                y = F.leaky_relu(y)
            
            if self.num_views > 1 and i == self.merge_layer:
                y = y.view(
                    -1, self.num_views, *y.size()[1:]
                ).mean(dim=1)
                tmpy = feature.view(
                    -1, self.num_views, *feature.size()[1:]
                ).mean(dim=1)

        if self.last_op is not None:
            y = self.last_op(y)

        return y
