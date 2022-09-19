import torch
import torch.nn as nn
import numpy as np

class AdapFilter3d(nn.Module):
    def __init__(self, kernel_size=[3,3,3], dilation=[1,1,1]):
        """
        input: (B, C, D, H, W), adaptive kernel F: (B, C, D, H, W, d, h, w)
        """
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size, kernel_size]
        else:
            self.kernel_size = kernel_size
        if isinstance(dilation, int):
            self.dilation = [dilation, dilation, dilation]
        else:
            self.dilation = dilation
        #
        self.d, self.h, self.w = self.kernel_size
        self.N = np.prod(self.kernel_size)
        #
        self.dilate_d, self.dilate_h, self.dilate_w = self.dilation
        #
        self.w_dilated, self.h_dilated, self.d_dilated = ((self.w-1)*self.dilate_w + 1), ((self.h-1)*self.dilate_h + 1), ((self.d-1)*self.dilate_d + 1)
        self.pad3d = nn.ConstantPad3d(((self.w_dilated-1)//2, (self.w_dilated-1)//2, (self.h_dilated-1)//2, (self.h_dilated-1)//2, (self.d_dilated-1)//2, (self.d_dilated-1)//2), 0.)
        #
        self.unfold = nn.Unfold(kernel_size=(self.d*self.h, self.w), dilation=self.dilation[-1], padding=0, stride=1)
    #
    def forward(self, input, F, norm=None, act=None, residual=False): # norm=None: default self.norm; norm=False: no norm.
        """
        input: (B, C, D, H, W), adaptive kernel F: (B, C_F, D, H, W, d, h, w),
        output: (B, C, D, H, W)
        """
        x = input.clone()
        dtype = input.data.type()
        B, C, D, H, W = x.shape
        C_F = F.shape[1]
        # channel expansion for grouped filtering
        if C != C_F:
            # F = F.contiguous().expand(-1,C,-1,-1,-1,-1,-1,-1)
            assert (C > C_F and C % C_F) == 0, 'computed filters do not match features'
            F = torch.split(F, 1, dim=1)
            F = torch.cat([x.expand(-1,C//C_F,-1,-1,-1,-1,-1,-1) for x in F], dim=1)

        assert x.shape == F.shape[:-3] and list(self.kernel_size) == list(F.shape[-3:]), 'input and flitering kernel do not match'
        # padding
        x = self.pad3d(x) # note that the cloned input has been padded since here
        # unfold the input
        x = x.contiguous().view(B, C, -1, W+self.w_dilated-1)
        x = self.unfold(x) # (B, C*d*h*w, D*H*W)
        # filter
        F = F.flatten(start_dim=-3) # (B, C, D, H, W, N)
        F = F.permute(0,1,-1,2,3,4).contiguous().view(B, -1, D, H, W)
        F = F.flatten(start_dim=-3) # (B, C*N,  D*H*W)
        #
        x_F = (x * F).contiguous().view(B, C, self.d*self.h*self.w, D, H, W)
        x_F = x_F.sum(dim=2) # (B, C, D, H, W)
        #
        if norm is not None:
            x_F = norm(x_F)

        if act is not None:
            x_F = act(x_F)
            
        if residual:
            x_F = input + x_F    
        return x_F

def adap_reshape(input, kernel_size=[3,3,3], softmax=False):
    """
    reshape input of size (B, C*d*h*w, D, H, W) to be (B, C, D, H, W, d, h, w).
    """
    x = input.clone() # clone() is necessary for the reason that .view(), .reshape(), and .flatten() return the same tensor of a different shape (as y = x) and consequently some changes in returned values via inplace operators, e.g. x += 10 and nn.ReLU(inplace=True)(x) in any operations. and nn.ReLU(inplace=True) (x = x+10. not work), will cause the same changes in input values.
    if isinstance(kernel_size, int):
        d, h, w = [kernel_size, kernel_size, kernel_size]
    else:
        d, h, w = kernel_size
    B, _, D, H, W = x.shape
    x = x.contiguous().view((B, -1, d, h, w, D, H, W))
    x = x.contiguous().permute((0,1,-3,-2,-1,2,3,4)) # (B, C, D, H, W, d, h, w)

    if softmax:
        x = x.contiguous().flatten(start_dim=-3)
        x = nn.functional.softmax(x, dim=-1)
        x = x.contiguous().view(B, -1, D, H, W, d, h, w)

    return x