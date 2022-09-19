import torch
import torch.nn as nn
import numpy as np

from .spatio_temporal_filtering import *
from .utils import Contextual_Transformer2d

def pixel_roll2d(x, shifts, dims, repeat=True, value=0.):
    if repeat:
        return torch.roll(x, shifts=shifts, dims=dims)
    else:
        _, _, H, W = x.shape
        output = nn.ConstantPad2d(padding=(max(0, -shifts[1]), max(0, shifts[1]), max(0, -shifts[0]), max(0, shifts[0])), value=value)(x)
        output = torch.roll(output, shifts=shifts, dims=dims)
        start_point = torch.relu(-1*torch.tensor(shifts))
        start_H, start_W = int(start_point[0].item()), int(start_point[1].item())
        return output[..., start_H:start_H+H, start_W:start_W+W]
        
class Attention_Branch(nn.Module):
    '''densely connected block'''
    def __init__(self, filters=32, out_channels=1, L=2): # L=3 brain dataset, L=3 knee dataset with channel collapse by the factor of 4
        super().__init__()
        self.out_channels = out_channels

        self.head = nn.Sequential(
            nn.Conv3d(filters, filters, (1,3,3), (1,1,1), (0,1,1), bias=False),
            nn.InstanceNorm3d(filters),
            nn.LeakyReLU(0.2, inplace=True),
            )
        #
        self.down1 = nn.Sequential(
            nn.Conv3d(filters, filters, (1,4,4), (1,2,2), (0,1,1), bias=False),
            nn.InstanceNorm3d(filters),
            nn.LeakyReLU(0.2, inplace=True),
            )
        #
        self.down2 = nn.Sequential(
            nn.Conv3d(filters, filters, (1,4,4), (1,2,2), (0,1,1), bias=False),
            nn.InstanceNorm3d(filters),
            nn.LeakyReLU(0.2, inplace=True),
            )
        #
        self.e0 = nn.Sequential(
            nn.Conv3d(filters, filters, (1,3,3), (1,1,1), (0,1,1), bias=False),
            nn.InstanceNorm3d(filters),
            nn.LeakyReLU(0.2, inplace=True),
            )
        #
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=(1,2,2), mode='nearest'),
            nn.Conv3d(filters, filters, (1,3,3), (1,1,1), (0,1,1), bias=False),
            nn.InstanceNorm3d(filters),
            nn.LeakyReLU(0.2, inplace=True),
            )
        #
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=(1,2,2), mode='nearest'),
            nn.Conv3d(2*filters, filters, (1,3,3), (1,1,1), (0,1,1), bias=False),
            nn.InstanceNorm3d(filters),
            nn.LeakyReLU(0.2, inplace=True),
            )
        #
        self.tail = nn.Sequential(
            nn.Conv3d(2*filters, self.out_channels, (1,3,3), (1,1,1), (0,1,1), bias=False),
            )

        ''' non-local spatial correlations '''
        self.epsilon = 1.e-6
        self.h = 0.25 # temperature parameter
        self.repeat = False # control the rolling mode of the pixel shifting

        self.norm = nn.Softmax(dim=1)

        self.L = L
        self.collapse = nn.Sequential(
            nn.Conv2d(filters, filters//4, 1, 1, 0, bias=False),
        )

        self.non_local = nn.Sequential(
            nn.Conv2d((2*self.L+1)**2, filters, 1, 1, 0, bias=False),

            # nn.InstanceNorm2d(filters),
        )

        # direction vectors are adopted
        self.collapse_Q = nn.Sequential(
            nn.Conv2d(filters, filters//4, 1, 1, 0, bias=False),
        )

        self.direct = [nn.Conv2d(filters//4+1, 1, 3, 1, 1, bias=False) for i in range(-1*self.L, self.L+1) for j in range(-1*self.L, self.L+1)]
        self.direct = nn.ModuleList(self.direct)

    def forward(self, input, dilation=[1, 1]):
        ''' non-local spatial correlations '''
        input = input.contiguous().squeeze(dim=2) # (B, C, H ,W)
        dilate_h, dilate_w = dilation

        input_Q = self.collapse_Q(input) # query embedding, reduce the feature depth
        input_Q = input_Q / (torch.norm(input_Q, dim=1, keepdim=True) + self.epsilon) # normalize

        input = self.collapse(input) # reduce the feature depth
        input = input / (torch.norm(input, dim=1, keepdim=True) + self.epsilon) # normalize

        pixel_shift_dot = [(pixel_roll2d(input_Q, shifts=(i*dilate_h,j*dilate_w), dims=(-2,-1), repeat=self.repeat)*input).sum(dim=1, keepdim=True) for i in range(-1*self.L, self.L+1) for j in range(-1*self.L, self.L+1)] # element-wise product, [..., (B, 1, H, W), ...]

        # compute relative cosine distance
        pixel_shift_dot = torch.cat(pixel_shift_dot, dim=1) # (B, (2*L+1)**2, H ,W)

        pixel_shift_dot = 1. - pixel_shift_dot # use Cosine distance
        pixel_shift_dot = pixel_shift_dot / (pixel_shift_dot.max(dim=1, keepdim=True)[0] + self.epsilon)
        pixel_shift_dot = 1.- pixel_shift_dot/self.h # for Softmax normalization, # (B, (2*L+1)**2, H ,W)

        # pixel_shift_dot = self.norm(pixel_shift_dot) # normalization is not used here
        
        # pixel_shift_dot = self.non_local[0](pixel_shift_dot) # (B, C', H ,W)
        # pixel_shift_dot = pixel_shift_dot.contiguous().unsqueeze(dim=2) # (B, C', 1, H ,W)
        # return self.norm(pixel_shift_dot).contiguous().unsqueeze(dim=2) # (B, (2*L+1)**2, 1, H ,W)

        # adopt directional information as well
        pixel_shift_dot = torch.split(pixel_shift_dot, 1, dim=1) # correlation information [..., (B, 1, H, W), ...]

        pixel_shift_mixed = [self.direct[(j+self.L)+(i+self.L)*(2*self.L+1)](torch.cat([(pixel_roll2d(input_Q, shifts=(i*dilate_h,j*dilate_w), dims=(-2,-1), repeat=self.repeat)-input), pixel_shift_dot[(j+self.L)+(i+self.L)*(2*self.L+1)]], dim=1)) for i in range(-1*self.L, self.L+1) for j in range(-1*self.L, self.L+1)] # [..., (B, C+1, H, W), ...] to [..., (B, 1, H, W), ...]

        pixel_shift_mixed = torch.cat(pixel_shift_mixed, dim=1) # (B, (2*L+1)**2, H, W)

        pixel_shift_mixed = nn.LeakyReLU(0.2, inplace=True)(self.non_local[0](pixel_shift_mixed)) # (B, C', H ,W)
        pixel_shift_mixed = pixel_shift_mixed.contiguous().unsqueeze(dim=2) # (B, C', 1, H ,W)

        ''' kernel prediction '''
        d0 = self.head(pixel_shift_mixed)
        d1 = self.down1(d0)
        d2 = self.down2(d1)

        _, _, D0, H0, W0 = d0.shape
        _, _, D1, H1, W1 = d1.shape

        #
        e0 = self.e0(d2)
        #
        u2 = self.up2(e0) if H1 % 2 == 0 and W1 % 2 == 0 else self.up2[1:](nn.Upsample(size=(D1,H1,W1))(e0))
        u1 = self.up1(torch.cat([d1, u2], dim=1)) if H0 % 2 == 0 and W0 % 2 == 0 else self.up1[1:](nn.Upsample(size=(D0,H0,W0))(torch.cat([d1, u2], dim=1)))
        #
        spatial_att = self.tail(torch.cat([d0, u1], dim=1))
        
        return spatial_att
        
class Shallow_Unet(nn.Module):
    '''densely connected block'''
    def __init__(self, filters=32, out_channels=1, reduction=2): # L=3 brain dataset, L=3 knee dataset with channel collapse by the factor of 4
        super().__init__()
        self.out_channels = out_channels

        self.head = nn.Sequential(
            nn.Conv2d(filters, filters//reduction, (3,3), (1,1), (1,1), bias=False),
            nn.InstanceNorm2d(filters//reduction),
            nn.LeakyReLU(0.2, inplace=True),
            )
        #
        self.down1 = nn.Sequential(
            nn.Conv2d(filters//reduction, filters//reduction//2, (4,4), (2,2), (1,1), bias=False),
            nn.InstanceNorm2d(filters//reduction//2),
            nn.LeakyReLU(0.2, inplace=True),
            )
        #
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=(2,2), mode='nearest'),
            nn.Conv2d(filters//reduction//2, filters//reduction, (3,3), (1,1), (1,1), bias=False),
            nn.InstanceNorm2d(filters//reduction),
            nn.LeakyReLU(0.2, inplace=True),
            )
        #
        self.tail = nn.Sequential(
            nn.Conv2d(2*filters//reduction, self.out_channels, (3,3), (1,1), (1,1), bias=False),
            )

    def forward(self, input):
        # ''' kernel prediction '''
        d0 = self.head(input)
        d1 = self.down1(d0)

        _, _, H0, W0 = d0.shape

        #
        u1 = self.up1(d1) if H0 % 2 == 0 and W0 % 2 == 0 else self.up1[1:](nn.Upsample(size=(H0,W0))(e0))
        #
        spatial_att = self.tail(torch.cat([d0, u1], dim=1))
        
        return spatial_att

class STASM_2D(nn.Module):
    '''spatio-temporal attentive selection module: channel-wise attention is first applied and spatial attention mask is then computed and convolved with features, namely spatially adaptive filtering'''
    def __init__(self, filters=32, sci=False, groups=1):
        super().__init__()
        self.filters = filters
        # self.multi_scale = multi_scale # indicate whether we use patch-based statistics for channel attention
        self.sci = sci

        # self.head = nn.Sequential(
        #     nn.Conv2d(in_channel, filters, 1, 1, 0, bias=False),
        #     #
        #     nn.Conv2d(filters, filters, (1,3,3), (1,1,1), (0,1,1), bias=False),
        #     nn.InstanceNorm2d(filters),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     )
        # # patch 0: global
        # self.mlp = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(filters, filters // reduction_ratio),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Linear(filters // reduction_ratio, filters)
        # )

        # self.sigmoid = nn.Sigmoid()

        if self.sci:
            self.cam = Contextual_Transformer2d(channel=self.filters)

        self.attention = Attention_Branch(self.filters, out_channels=9*groups)

        self.adaptive_filtering = AdapFilter3d(kernel_size=[1,3,3], dilation=[1,1,1])
        # 

    def forward(self, input):
        output = input.unsqueeze(dim=-3) # (B, C, 1, H, W)
        spatial_att = self.attention(output) # (B, 9, 1, H, W)

        spatial_att = adap_reshape(spatial_att, kernel_size=[1,3,3], softmax=True) # (B, 1, 1, H, W, 1, 3, 3)

        # weighted average
        output = self.adaptive_filtering(output, spatial_att, norm=None, residual=True)

        return output.squeeze(dim=-3)

