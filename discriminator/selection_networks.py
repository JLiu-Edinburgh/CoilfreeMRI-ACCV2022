import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.nn import Module, Parameter, Softmax

import math


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim, suppression=True, beta=None):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.suppression = suppression
        self.beta = beta

        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)

    def forward(self,x):
        m_batchsize, C, height, width = x.size()
        d_k_root = math.sqrt(height*width)

        query = x.view(m_batchsize, C, -1)
        key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(query, key) # batch matrix multiplication. compared to A@B, it applies to 3-D tensors
        
        if self.suppression:
            energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
            # b.expand_as(a): last dimensions of b must match those of a except that it is 1, and first dimensions are automatically unsqueezed and repeated according to a.
        else:
            energy_new = energy/d_k_root

        attention = self.softmax(energy_new)
        value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, value)
        out = out.view(m_batchsize, C, height, width)

        if self.beta is not None:
            out = self.beta*out + (1-self.beta)*x
        else:
            out = self.gamma*out + x
            
        return out
