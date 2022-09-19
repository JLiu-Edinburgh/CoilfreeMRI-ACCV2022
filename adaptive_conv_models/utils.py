import random
import math
from PIL import Image
import numpy as np

import torch
import torch.nn as nn

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim, suppression=True, beta=None):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.suppression = suppression
        self.beta = beta

        if beta is None:
            self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,input):
        if len(input.shape) == 4:
            x = input.clone()
            x = x.contiguous().unsqueeze(dim=2)
        else:
            x = input.clone()
        m_batchsize, C, depth, height, width = x.size()
        d_k_root = math.sqrt(depth*height*width)

        query = x.contiguous().view(m_batchsize, C, -1)
        key = x.contiguous().view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(query, key) # batch matrix multiplication. compared to A@B, it applies to 3-D tensors
        
        if self.suppression:
            energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
            # b.expand_as(a): last dimensions of b must match those of a except that it is 1, and first dimensions are automatically unsqueezed and repeated according to a.
        else:
            energy_new = energy/d_k_root

        attention = self.softmax(energy_new)
        value = x.contiguous().view(m_batchsize, C, -1)

        out = torch.bmm(attention, value)
        out = out.contiguous().view(m_batchsize, C, depth, height, width)

        if self.beta is not None:
            out = self.beta*out + x
        else:
            out = self.gamma*out + x
        if len(input.shape) == 4:
            out = out.contiguous().squeeze(dim=2)
        return out

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# contextual loss for transformer
''' 
2018 the Contextual Loss- Maintaining Natural Image Statistics with the Contextual Loss;
2018 The Contextual Loss for Image Transformation with Non-Aligned Data;
2017 Template Matching with Deformable Diversity Similarity.pdf;
2018 Image Inpainting via Generative Multi-column;
'''

def l1_norm(x, dim, eps=1.e-6):
    ''' input is non-negative already '''
    out = x / (x.sum(dim=dim, keepdim=True) + eps)
    return out

class Contextual_Transformer2d(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.epsilon = 1.e-6
        self.h = 0.25 # temperature parameter

        #
        self.q = nn.Conv2d(channel, channel, 1, 1, 0, bias=False)
        self.k = nn.Conv2d(channel, channel, 1, 1, 0, bias=False)
        self.v = nn.Conv2d(channel, channel, 1, 1, 0, bias=False)

        #
        self.norm = nn.Softmax(dim=-2) # for contextual attention 
        # self.norm = nn.Softmax(dim=-1) # for scaled dot-production attention 

        #
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        input = x.clone()
        # embedding
        Q = self.q(x).contiguous().flatten(start_dim=-2) # (B, C, H*W)
        K = self.k(x).contiguous().flatten(start_dim=-2)
        V = self.v(x).contiguous().flatten(start_dim=-2)
    
        ''' (B, C, C), contextual attention '''
        Q = Q / torch.norm(Q, dim=-1, keepdim=True) # L_2 normalize
        K= K / torch.norm(K, dim=-1, keepdim=True)
        M = Q @ K.transpose(-1,-2) # (B, C, C)

        # compute the relativistic distance matrix and normalize it
        # M = M / M.max(dim=-2, keepdim=True)[0] # use Cosine similarity

        M = 1. - M # use Cosine distance
        M = M / (M.max(dim=-2, keepdim=True)[0] + self.epsilon)
        M = 1.- M/self.h # for Softmax normalization

        # dual normalization
        M = self.norm(M)
        M = l1_norm(M, dim=-1) # L_1 normalization

        ''' (B, C, C), scaled dot-production attention '''
        # M = Q @ K.transpose(-1,-2) 
        # M = self.norm(M/torch.sqrt(torch.tensor(H*W, dtype=torch.float)))

        # select the relatively closest candidates
        # M_r, index = M.max(dim=-1, keepdim=True) # (B, C, 1)
        # index = index.expand(-1, -1, H*W) # (B, C, H*W)
        # V_r = V.gather(dim=1, index = index) # (B, C, H*W)
        # output = (M_r * V_r).contiguous().view(B, C, H, W)

        output = (M @ V).contiguous().view(B, C, H, W) # all attention values are used

        # torch.save(M.cpu(), 'attention_maps_%.8f.pt'% V.mean().item())
        # print(V.mean().item())

        return input + self.gamma*output



class Contextual_TransHead2d(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.epsilon = 1.e-6
        self.h = 0.25 # temperature parameter

        #
        self.q = nn.Conv2d(channel, channel, 3, 1, 1, bias=False)
        self.k = nn.Conv2d(channel, channel, 3, 1, 1, bias=False)
        self.v = nn.Conv2d(channel, channel, 3, 1, 1, bias=False)

        #
        self.norm = nn.Softmax(dim=-2) # for contextual attention 

    def forward(self, x):
        B, C, H, W = x.shape

        # embedding
        Q = self.q(x).contiguous().flatten(start_dim=-2) # (B, C, H*W)
        K = self.k(x).contiguous().flatten(start_dim=-2)
        V = self.v(x).contiguous().flatten(start_dim=-2)

        # inner-product
        Q = Q / torch.norm(Q, dim=-1, keepdim=True) # L_2 normalize
        K= K / torch.norm(K, dim=-1, keepdim=True)
        M = Q @ K.transpose(-1,-2) # (B, C, C)

        M = 1. - M # use Cosine distance
        M = M / (M.max(dim=-2, keepdim=True)[0] + self.epsilon)
        M = 1.- M/self.h # for Softmax normalization

        M = self.norm(M)
        M = l1_norm(M, dim=1) # L_1 normalization

        output = (M @ V).contiguous().view(B, C, H, W) # all attention values are used

        return output

# class Contextual_Transformer2d(nn.Module):
    # ''' multi-head version '''
#     def __init__(self, channel, heads=3):
#         super().__init__()
        
#         self.multi_head = []
#         for i in range(heads):
#             self.multi_head.append(Contextual_TransHead2d(channel))

#         self.multi_head = nn.ModuleList(self.multi_head)

#         self.fusion = nn.Conv2d(channel*heads, channel, 3, 1, 1, bias=False)

#         #
#         self.gamma = nn.Parameter(torch.zeros(1))

#     def forward(self, x):
#         # multi-head
#         multi_output = []
#         for i, h in enumerate(self.multi_head):
#             multi_output.append(h(x))

#         output = self.fusion(torch.cat(multi_output, dim=1))

#         return x + self.gamma*output


# Fourier feature https://github.com/GlassyWing/fourier-feature-networks/blob/master/demo.py
class Siren(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = 30
    def forward(self, input):
        return torch.sin(self.w * input)

class Fourier_Convolution(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        assert out_channel % 2 == 0, 'out_channel needs to be even!'
        self.B = nn.Parameter(torch.randn(1, 1, in_channel, out_channel//2)) # (1, 1, C, C'//2)
    def forward(self, input):
        input = input.permute(0,2,3,1) # (B, H, W, C)
        output = 2.* np.pi * input @ self.B # (B, H, W, C'//2)
        output = torch.cat([torch.cos(output), torch.sin(output)], dim=-1) # (B, H, W, C')
        return output.permute(0, -1, 1, 2)

def Fourier_feature_transform(input):
    _, C, _, _ = input.shape
    assert C % 2 == 0, 'in_channel needs to be even!'
    output = torch.split(input, C//2, dim=1)
    output = torch.cat([torch.cos(2.*np.pi*output[0]), torch.sin(2.*np.pi*output[1])], dim=1)
    return output

# SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks
def simAM (X, lambda_X=0.0001):
    # X: input feature [N, C, H, W]
    # spatial size
    n = X.shape[2] * X.shape[3] - 1
    # square of (t - u)
    d = (X - X.mean(dim=[2,3], keepdim=True)).pow(2)
    # d.sum() / n is channel variance
    v = d.sum(dim=[2,3], keepdim=True) / n
    # E_inv groups all importance of X
    E_inv = d / (4 * (v + lambda_X)) + 0.5
    # return attended features
    return X * nn.Sigmoid()(E_inv)
