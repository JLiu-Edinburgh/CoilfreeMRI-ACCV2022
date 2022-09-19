import torch
import torch.nn as nn
import numpy as np

from .spatio_temporal_filtering import *
from .siamese_adaptive_gan import STASM_2D
from .esrgan import DenseBlock2d

class weightedAverageTerm(nn.Module):
    def __init__(self, para=0.1):
        super().__init__()
        self.para = nn.Parameter(torch.Tensor([para]))

    def forward(self, x, Sx):
        x = self.para*x + (1 - self.para)*Sx
        return x

class Multi_Level_Dense_Unet(nn.Module):
    '''densely connected block, L4: 4 levels'''
    def __init__(self, scaler_c=2, dense_dilation=False, n_stage=0, stasm=None, img_shape=(2,256,256)):
        super().__init__()
        _, self.h, self.w = img_shape
        if self.h % 2**4 != 0:
            self.scale_h = self.h // 2**3
            self.h_integer = False
        else:
            self.scale_h = self.h // 2**3
            self.h_integer = True

        if self.w % 2**4 != 0:
            self.scale_w = self.w // 2**3
            self.w_integer = False
        else:
            self.scale_w = self.w // 2**3
            self.w_integer = True

        self.dense_dilation = dense_dilation
        self.n_stage = n_stage
        # SIO to S0 before receiving connections
        self.head = nn.Sequential(
            nn.Conv2d((self.n_stage+1)*scaler_c*16, scaler_c*16, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(scaler_c*16),
            nn.LeakyReLU(0.2, inplace=True),
            )
        # S1
        self.down1 = nn.Sequential(
            nn.Conv2d((2*self.n_stage+1)*scaler_c*16, scaler_c*32, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(scaler_c*32), # w or w/o norm
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.Conv2d(scaler_c*32, scaler_c*32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(scaler_c*32),
            nn.LeakyReLU(0.2, inplace=True),
            )
        # S2
        self.down2 = nn.Sequential(
            nn.Conv2d((2*self.n_stage+1)*scaler_c*32, scaler_c*64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(scaler_c*64),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.Conv2d(scaler_c*64, scaler_c*64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(scaler_c*64),
            nn.LeakyReLU(0.2, inplace=True),
            )
        # S3
        self.down3 = nn.Sequential(
            nn.Conv2d((2*self.n_stage+1)*scaler_c*64, scaler_c*64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(scaler_c*64), # w or w/o norm
            nn.LeakyReLU(0.2, inplace=True),
            )
        # S4
        self.down4 = nn.Sequential(
            nn.Conv2d((2*self.n_stage+1)*scaler_c*64, scaler_c*64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(scaler_c*64), # w or w/o norm
            nn.LeakyReLU(0.2, inplace=True),
            )
        # S3'
        if self.h_integer and self.w_integer:
            self.up4 = nn.Sequential(
                #
                DenseBlock2d(filters=(self.n_stage+1)*scaler_c*64, channel_reduction=self.n_stage+1, out_channel=scaler_c*64, n_layers=3, norm=True, non_linearity=True, dilation=self.dense_dilation, bias=False),

                #
                nn.Upsample(scale_factor=(2,2), mode='nearest'),
                nn.Conv2d(scaler_c*64, scaler_c*64, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(scaler_c*64),
                nn.LeakyReLU(0.2, inplace=True),
                )
        else:
            self.up4 = nn.Sequential(
                #
                DenseBlock2d(filters=(self.n_stage+1)*scaler_c*64, channel_reduction=self.n_stage+1, out_channel=scaler_c*64, n_layers=3, norm=True, non_linearity=True, dilation=self.dense_dilation, bias=False),
                
                #
                nn.Upsample(size=(self.scale_h,self.scale_w), mode='nearest'),
                nn.Conv2d(scaler_c*64, scaler_c*64, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(scaler_c*64),
                nn.LeakyReLU(0.2, inplace=True),
                )
        # S2'
        self.up3 = nn.Sequential(
            DenseBlock2d(filters=(2*self.n_stage+2)*scaler_c*64, channel_reduction=2*self.n_stage+2, out_channel=scaler_c*64, n_layers=3, norm=True, non_linearity=True, dilation=self.dense_dilation, bias=False),
            #
            nn.Upsample(scale_factor=(2,2), mode='nearest'),
            nn.Conv2d(scaler_c*64, scaler_c*64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(scaler_c*64),
            nn.LeakyReLU(0.2, inplace=True),
            )
        # S1'
        self.up2 = nn.Sequential(
            DenseBlock2d(filters=(2*self.n_stage+2)*scaler_c*64, channel_reduction=2*self.n_stage+2, out_channel=scaler_c*64, n_layers=3, norm=True, non_linearity=True, dilation=self.dense_dilation, bias=False),
            #
            nn.Upsample(scale_factor=(2,2), mode='nearest'),
            nn.Conv2d(scaler_c*64, scaler_c*32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(scaler_c*32),
            nn.LeakyReLU(0.2, inplace=True),
            )
        # S0'
        self.up1 = nn.Sequential(
            DenseBlock2d(filters=(2*self.n_stage+2)*scaler_c*32, channel_reduction=2*self.n_stage+2, out_channel=scaler_c*32, n_layers=3, norm=True, non_linearity=True, dilation=self.dense_dilation, bias=False),
            #
            nn.Upsample(scale_factor=(2,2), mode='nearest'),
            nn.Conv2d(scaler_c*32, scaler_c*16, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(scaler_c*16),
            nn.LeakyReLU(0.2, inplace=True),
            )
        # SIO'
        self.tail = nn.Sequential(
            DenseBlock2d(filters=(2*self.n_stage+2)*scaler_c*16, channel_reduction=2*self.n_stage+2, out_channel=scaler_c*16, n_layers=3, norm=True, non_linearity=True, dilation=self.dense_dilation, bias=False, stasm=stasm),
            )
    def forward(self, input_dict, DC_conv=None, DC_instances=None, DC_gamma=None):
        # encoding
        input_dict['S0'] = torch.cat([self.head(input_dict['SIO']), input_dict['S0']], dim=1)
        input_dict['S1'] = torch.cat([self.down1(input_dict['S0']), input_dict['S1']], dim=1)
        input_dict['S2'] = torch.cat([self.down2(input_dict['S1']), input_dict['S2']], dim=1)
        input_dict['S3'] = torch.cat([self.down3(input_dict['S2']), input_dict['S3']], dim=1)
        input_dict['S4'] = torch.cat([self.down4(input_dict['S3']), input_dict['S4']], dim=1)
        # decoding
        input_dict['S3'] = torch.cat([self.up4(input_dict['S4']), input_dict['S3']], dim=1)
        input_dict['S2'] = torch.cat([self.up3(input_dict['S3']), input_dict['S2']], dim=1)
        input_dict['S1'] = torch.cat([self.up2(input_dict['S2']), input_dict['S1']], dim=1)
        input_dict['S0'] = torch.cat([self.up1(input_dict['S1']), input_dict['S0']], dim=1)

        if DC_conv is None:
            input_dict['SIO'] = torch.cat([self.tail(input_dict['S0']), input_dict['SIO']], dim=1)

            return input_dict
        else: 
            output_tail = self.tail(input_dict['S0'])

            output_DC = DC_conv.head[-2](output_tail)

            output_DC = DC_instances['DC'](output_DC, DC_instances['zero_filled'], DC_instances['csm'])

            output_DC = DC_conv.head[-1](output_DC)

            # residual feature shortcut
            input_dict['SIO'] = torch.cat([DC_gamma(output_tail, output_DC), input_dict['SIO']], dim=1)

            return input_dict, None # save cuda memory when intermediate results are not involved

class Multi_Level_Dense_Network(nn.Module):
    '''densely connected block'''
    def __init__(self, img_shape, out_channel=None, scaler_c=2, dense_dilation=False, stages=3, stasm=True, groups=1, data_consistency=True, L3=False):
        super().__init__()
        self.c, self.h, self.w = img_shape
        self.data_consistency = data_consistency
        self.L3 = L3
        
        self.out_channel = out_channel if out_channel is not None else self.c

        # STASM 
        if isinstance(stasm, bool) and stasm:
            self.stasm = nn.Sequential()
            for i in range(stages):
                self.stasm.add_module('stasm%d'%i, STASM_2D(filters=scaler_c*16, sci=True, groups=groups)) # added to the last dense_block in each Unet
        elif isinstance(stasm, bool) and not stasm:
            self.stasm = []
            for i in range(stages):
                self.stasm.append(None)
        else:
            self.stasm = stasm

        #
        self.expansion = nn.Sequential(
            nn.Conv2d(self.c, scaler_c*16, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(scaler_c*16, scaler_c*16, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            )

        #    
        self.unets = nn.ModuleList()
        
        for i in range(stages):
            self.unets.add_module('unet%d'%i, Multi_Level_Dense_Unet(scaler_c=scaler_c, dense_dilation=dense_dilation, n_stage=i, stasm=self.stasm[i], img_shape=img_shape))

        #
        class DC_Module(nn.Sequential):
            def __init__(self, in_channel, dc_filter, out_channel):
                super().__init__()
                self.add_module('head', nn.Sequential(
                    nn.Conv2d(in_channel+1, dc_filter, 3, 1, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(dc_filter, dc_filter, 3, 1, 1, bias=False),

                    nn.Conv2d(in_channel, 2, 3, 1, 1, bias=False), # traditional DC channel layers
                    nn.Conv2d(2, in_channel, 3, 1, 1, bias=False),
                )
                )

                self.add_module('d1', nn.Sequential(
                    nn.InstanceNorm2d(dc_filter),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(dc_filter, dc_filter, 4, 2, 1, bias=False),

                    nn.InstanceNorm2d(dc_filter),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(dc_filter, dc_filter, 3, 1, 1, bias=False)
                )
                )
                self.add_module('d2', nn.Sequential(
                    nn.InstanceNorm2d(dc_filter),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(dc_filter, dc_filter, 4, 2, 1, bias=False),

                    nn.InstanceNorm2d(dc_filter),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(dc_filter, dc_filter, 3, 1, 1, bias=False)
                )
                )
                self.add_module('d3', nn.Sequential(
                    nn.InstanceNorm2d(dc_filter),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(dc_filter, dc_filter, 4, 2, 1, bias=False),

                    nn.InstanceNorm2d(dc_filter),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(dc_filter, dc_filter, 3, 1, 1, bias=False)
                )
                )

                self.add_module('collapse_0', nn.Conv2d(2*dc_filter, out_channel, 3, 1, 1, bias=False))
                self.add_module('collapse_0_', nn.Conv2d(3*dc_filter, out_channel, 3, 1, 1, bias=False))
                self.add_module('collapse_1', nn.Conv2d(dc_filter, out_channel, 3, 1, 1, bias=False))

                self.add_module('expansion', nn.Conv2d(out_channel, in_channel, 3, 1, 1, bias=False))

                ''' Gaussian pyramid for observations and mixing blocks'''
                self.add_module('fusion', nn.Sequential(
                    nn.Conv2d(2, dc_filter, 3, 1, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(dc_filter, out_channel, 3, 1, 1, bias=False)
                )
                )

                self.add_module('head_z', nn.Conv2d(out_channel, dc_filter, 3, 1, 1, bias=False))

                self.add_module('d1_z', nn.Sequential(
                    nn.InstanceNorm2d(dc_filter),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(dc_filter, dc_filter, 4, 2, 1, bias=False)
                )
                )
                self.add_module('d2_z', nn.Sequential(
                    nn.InstanceNorm2d(dc_filter),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(dc_filter, dc_filter, 4, 2, 1, bias=False)
                )
                )
                self.add_module('d3_z', nn.Sequential(
                    nn.InstanceNorm2d(dc_filter),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(dc_filter, dc_filter, 4, 2, 1, bias=False)
                )
                )

                self.add_module('mix_h', nn.Sequential(
                    nn.Conv2d(2*dc_filter, dc_filter, 3, 1, 1, bias=False),
                    nn.InstanceNorm2d(dc_filter),
                    nn.LeakyReLU(0.2, inplace=True),
                )
                )
                self.add_module('mix_1', nn.Sequential(
                    nn.Conv2d(2*dc_filter, dc_filter, 3, 1, 1, bias=False),
                    nn.InstanceNorm2d(dc_filter),
                    nn.LeakyReLU(0.2, inplace=True),
                )
                )
                self.add_module('mix_2', nn.Sequential(
                    nn.Conv2d(2*dc_filter, dc_filter, 3, 1, 1, bias=False),
                    nn.InstanceNorm2d(dc_filter),
                    nn.LeakyReLU(0.2, inplace=True),
                )
                )
                self.add_module('mix_expansion', nn.Conv2d(dc_filter, in_channel, 3, 1, 1, bias=False))

        if not self.data_consistency:
            self.DC, self.DC_instances, self.DC_gamma = None, None, None
        else:
            self.DC = nn.ModuleList()
            self.DC_instances = []
            self.DC_gamma = nn.ModuleList()
            for i in range(stages):
                # self.DC.add_module('dc_conv_%d'%i, nn.Sequential(
                #     nn.Conv2d(scaler_c*16, self.out_channel, 3, 1, 1, bias=False),
                #     nn.Conv2d(self.out_channel, scaler_c*16, 3, 1, 1, bias=False),
                # )
                # ) # conventional DC layers
                self.DC.add_module('dc_conv_%d'%i, DC_Module(in_channel=scaler_c*16, dc_filter=8, out_channel=self.out_channel))
                #
                self.DC_instances.append({
                    'DC':None,
                    'zero_filled':None,
                    'csm':None,
                    })
                # 
                self.DC_gamma.add_module('weighted_sum_%d'%i, weightedAverageTerm(para=0.1))

        #
        self.collapse = nn.Sequential(
            nn.Conv2d((stages+1)*scaler_c*16, scaler_c*16, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(scaler_c*16, self.out_channel, 3, 1, 1, bias=False),
            nn.Tanh()
            )

        self.predictor = nn.Sequential(
            nn.Conv2d((stages+1)*scaler_c*16, scaler_c*16, 3, 1, 1, bias=False),
            # nn.Conv2d(self.out_channel, scaler_c*16, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(scaler_c*16, 2, 3, 1, 1, bias=False),

            nn.Conv2d(16, 1, 3, 1, 1, bias=False),
            ) # predictor for the DC loss

    def forward(self, input, zero_filled=None, csm=None, dc_operator=None):
        if len(input.shape) == 5 and input.shape[2] == 1:
            input = input.contiguous().squeeze(dim=2)
            squeezed = True
        else:
            squeezed = False

        zero_filled = input if zero_filled is None else zero_filled

        device = input.device
        input_dict = {
            'SIO':self.expansion(input),
            'S0':torch.FloatTensor().to(device),
            'S1':torch.FloatTensor().to(device),
            'S2':torch.FloatTensor().to(device),
            'S3':torch.FloatTensor().to(device),
            'S4':torch.FloatTensor().to(device),
            }
        #
        output_hierachi = []
        if not self.data_consistency:
            for unet in self.unets:
                input_dict = unet(input_dict)
        else:           
            for unet, dc, dc_instances, dc_gamma in zip(self.unets, self.DC, self.DC_instances, self.DC_gamma):
                # feed data consistency instances
                dc_instances['DC'] = dc_operator
                dc_instances['zero_filled'] = zero_filled
                dc_instances['csm'] = csm
                #
                input_dict, output_subnet = unet(input_dict, dc, dc_instances, dc_gamma)
                output_hierachi.append(output_subnet)

        #
        output = self.collapse(input_dict['SIO'])

        #
        if squeezed:
            output = output.unsqueeze(dim=2) # (B, 2, 1, H, W)
        
        return output, output_hierachi
