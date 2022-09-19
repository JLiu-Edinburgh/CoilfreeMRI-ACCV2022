import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math

class DenseBlock2d(nn.Module):
    def __init__(self, filters, channel_reduction=2, out_channel=None, n_layers=4, norm=True, non_linearity=True, dilation=False, bias=False, stasm=None):
        super().__init__()
        self.out_channel = out_channel if out_channel is not None else filters
        self.channel_reduction = channel_reduction

        self.reduction = nn.Conv2d(filters, filters//self.channel_reduction, 3, 1, padding=1, bias=bias)

        self.stasm = stasm # otherwise, STASM_2D(filters=filters//self.channel_reduction, sci=True) 

        if dilation:
            self.dilation = [1, 2, 3]  # [1, 2, 3] or [1, 2, 5], 
        else:
            self.dilation = [1, 1, 1]

        def block(in_features, norm=True, non_linearity=True, dilation=1, bias=True):
            # to preserve the output size, padding = padding+(dilation-1)*(kernel_size-1)//2
            layers = [nn.Conv2d(in_features, filters//self.channel_reduction, 3, 1, padding=dilation, bias=bias, dilation=dilation)]

            if norm:
                layers += [nn.InstanceNorm2d(filters)]

            if non_linearity:
                layers += [nn.LeakyReLU(0.2, inplace=True)]

            return nn.Sequential(*layers)
        
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            self.blocks.add_module('dense_layer_%d'%i, block(in_features=(i+1) * filters//self.channel_reduction, norm=norm, non_linearity=non_linearity, dilation=self.dilation[i%3], bias=bias))

        self.conv = nn.Conv2d((n_layers+1) * filters//self.channel_reduction, self.out_channel, 3, 1, 1, bias=bias)

    def forward(self, inputs):
        inputs = self.reduction(inputs)

        if self.stasm is not None:
            inputs = self.stasm(inputs) # apply attention 

        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)

        out = self.conv(inputs)
        return out
