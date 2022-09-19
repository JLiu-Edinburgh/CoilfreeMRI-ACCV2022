import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np

from torchvision.utils import save_image

# fftshift and ifftshift
def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)
    
def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)

def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)

def torch_fft(x_under):
    # x_under_per = x_under.permute(0, 2, 3, 1)

    # assert x_under_per.size(-1) == 2

    # x_under_per = ifftshift(x_under_per, dim=(-3, -2))
    # x_zf_per = torch.fft(x_under_per, 2, normalized=True)
    # x_zf_per = fftshift(x_zf_per, dim=(-3, -2))

    # x_zf = x_zf_per.permute(0, 3, 1, 2)
    # return x_zf

    x_under = x_under.permute(0, 2, 3, 1)

    assert x_under.size(-1) == 2

    x_under = ifftshift(x_under, dim=(-3, -2))
    x_under = torch.fft(x_under, 2, normalized=True)
    x_under = fftshift(x_under, dim=(-3, -2))

    x_under = x_under.permute(0, 3, 1, 2)
    return x_under

def torch_ifft(x_under):
    # x_under_per = x_under.permute(0, 2, 3, 1)
    # assert x_under_per.size(-1) == 2

    # x_under_per = ifftshift(x_under_per, dim=(-3, -2))
    # x_zf_per = torch.ifft(x_under_per, 2, normalized=True)
    # x_zf_per = fftshift(x_zf_per, dim=(-3, -2))

    # x_zf = x_zf_per.permute(0, 3, 1, 2)
    # return x_zf 

    x_under = x_under.permute(0, 2, 3, 1)
    assert x_under.size(-1) == 2

    x_under = ifftshift(x_under, dim=(-3, -2))
    x_under = torch.ifft(x_under, 2, normalized=True)
    x_under = fftshift(x_under, dim=(-3, -2))

    x_under = x_under.permute(0, 3, 1, 2)
    return x_under 

def sigtoimage(sig):
    ''' input (B, 2, H, W), and output (B, 1, H, W)  in magnitude '''
    x_real = torch.unsqueeze(sig[:, 0, :, :], 1)
    x_imag = torch.unsqueeze(sig[:, 1, :, :], 1)
    x_image = torch.sqrt(x_real * x_real + x_imag * x_imag)
    return x_image