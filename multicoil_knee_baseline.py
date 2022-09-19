# supervised training
import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from adaptive_conv_models import *
from discriminator import *

import torch.nn as nn
import torch.nn.functional as F
import torch

from h5topng.data import transforms as T
from h5topng.common import subsample

from Vtils import get_psnr, get_ssim

import random

from h5_dataloader.hdf5_dataset import HDF5Dataset

class To_h_space:
    def __init__(self, mask=None, center_fractions=[0.04], accelerations=[8], seed=None):
        self.mask = mask
        self.seed = seed
        if mask == None:
            self.mask_func = subsample.MaskFunc(center_fractions, accelerations)

    def __call__(self, data):
        data_copy = data.clone()
        device = data.device
        # (B, 2, H, W) to complex data (B,1,H,W,2)
        data = data.unsqueeze(dim=-1).transpose(1,-1)

        # to fft domian
        data = T.fft2(data)

        # apply mask
        if self.mask == None:
            data, _ = T.apply_mask(data, self.mask_func, seed=self.seed)
        else:
            self.mask = self.mask.to(device)
            data = torch.where(self.mask == 0, torch.Tensor([0.]).to(device), data)
            # data = self.mask.to(device) * data

        # to image domain
        data = T.ifft2(data)
        return data.transpose(1,-1).squeeze(dim=-1)

class To_k_space:
    def __init__(self, mask=None, center_fractions=[0.04], accelerations=[8], seed=None):
        self.mask = mask
        self.seed = seed
        if mask == None:
            self.mask_func = subsample.MaskFunc(center_fractions, accelerations)

    def __call__(self, data):
        device = data.device
        # to complex data (B,1,H,W,2)
        data = data.unsqueeze(dim=-1).transpose(1,-1)
        # to fft domian
        data = T.fft2(data)

        # apply mask
        if self.mask == None:
            data, _ = T.apply_mask(data, self.mask_func, seed=self.seed)
        else:
            self.mask = self.mask.to(device)
            data = torch.where(self.mask == 0, torch.Tensor([0.]).to(device), data)
        
        # to (B,2,H,W)
        return data.transpose(1,-1).squeeze(dim=-1)

from utils import torch_fft, torch_ifft, sigtoimage
class Soft_Data_Consistency(nn.Module):
    '''hard DC operator, mask: (B=1, C=1, H, W)'''
    def __init__(self, mask):
        super().__init__()
        self.mask = mask
        self.mask_c = torch.ones_like(mask) - mask # complementary of support
        
    # def __call__(self, data, data_u):
    def forward(self, data, data_u):
        '''input: (B,2,H,W)'''
        device = data.device
        self.mask = self.mask.to(device)
        self.mask_c = self.mask_c.to(device)

        # # to complex data (B,1,H,W,2)
        # data = data.unsqueeze(dim=-1).transpose(1,-1)
        # data_u = data_u.unsqueeze(dim=-1).transpose(1,-1)

        # to fft domian
        # data = T.fft2(data)
        # data_u = T.fft2(data_u)

        data = torch_fft(data)
        data_u = torch_fft(data_u)

        # DC operation
        data_dc = data*self.mask_c + data_u*self.mask

        # to image domain
        data_dc = torch_ifft(data_dc)

        return data_dc

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=35, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="NYU_MRI", help="name of the dataset")
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--testroot', required=True, help='path to dataset')
parser.add_argument('--mask', default=None, help='path to dataset')
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_depth", type=int, default=1, help="size of image depth, e.g. coils")
parser.add_argument("--img_coil", type=int, default=15, help="number of coils")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=2, help="number of image channels")
parser.add_argument("--repeat_dim", type=int, default=1, help="number of random samples in test")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")

parser.add_argument("--lambda_adv", type=float, default=0.05, help="pixelwise loss weight")
parser.add_argument("--lambda_pixel", type=float, default=10., help="pixelwise reconstruction loss weight")
parser.add_argument("--lambda_latent", type=float, default=0.5, help="latent loss weight")
parser.add_argument("--lambda_vgg", type=float, default=0.5, help="perceptual loss weight")
parser.add_argument("--lambda_grad", type=float, default=10., help="gradient penalty")

parser.add_argument("--stage", type=int, default=4)

opt = parser.parse_args()
# print(opt)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

input_shape = (opt.channels, opt.img_depth, opt.img_height, opt.img_width)

# mean square normalize
def mean_square_normalize(data, thresh=0.05, ratio=0.1, dilate=1.0):
    data[data.abs()<thresh] = 0.0 # threshold
    
    shape = data.shape
    mean_square = (data**2).sum(1).sqrt().mean((-2,-1))
    mean_square = mean_square.view((shape[0],1,1,1)).repeat((1,shape[1],shape[2],shape[3]))
    
    # normalize
    data = data/mean_square*ratio 
    data = torch.tanh(data*dilate)
    return data

def sample_images(epoch, i):
    """Saves a generated sample rom the validation set"""
    generator.eval()

    img_samples = None
    attention_samples = []

    for img_B, Csm in zip(testBatch, testCsm):
        img_B, Csm = torch.split(img_B, 1, dim=0), torch.split(Csm, 1, dim=0) # [..., (1,2,256,256), ...]

        # combination of coil views for signal alignment among different methods
        img_align = coil_combine(img_B, Csm) # (B, 2, H, W)
        img_align = sigtoimage(img_align)
        _, align_max = normedabs(img_align, return_max=True)

        csm_mask = sig2rss(Csm) # binary mask used to focus on saliencies

        #
        img_A = [to_cyc(full) for full in img_B]
        img_DC = [x.clone() for x in img_A]

        img_A, img_B = sig2rss(img_A), sig2rss(img_B) # RSS, (B=1, 1, H, W)
        img_A, img_B = normedabs(img_A), normedabs(img_B) # (B, 1, H, W), (-1,1)

        # Generate samples
        with torch.no_grad():
            fake_B, _ = generator(img_A.unsqueeze(dim=2), zero_filled=sum(img_DC), csm=Csm, dc_operator=multi_coil_dc)
            fake_B = fake_B.squeeze(dim=2) # (1, C, H, W)

        # compute magnitude maps
        img_B, fake_B, img_A = (img_B+ 1.) / 2., (fake_B+ 1.) / 2., (img_A+ 1.) / 2.

        # display results
        img_B, fake_B, img_A = img_B * align_max *csm_mask, fake_B * align_max *csm_mask, img_A * align_max

        # diff
        diff = (fake_B-img_B).abs()

        # Concatenate samples horisontally
        fake_B = torch.cat([x for x in fake_B], -1) # (C, H, 2*N*W)
        diff = torch.cat([x for x in diff], -1) # (C, H, 2*N*W)

        img_sample = torch.cat((img_A.squeeze(dim=0), fake_B, img_B.squeeze(dim=0), diff), -1) # (C, H, (N+2)*W)
        img_sample = img_sample.view(1, *img_sample.shape) # (1, C, H, (N+2)*W)
        # Concatenate with previous samples vertically
        img_samples = img_sample if img_samples is None else torch.cat([img_samples, img_sample], -2) # (1, C, M*H, (N+2)*W)

    save_image(img_samples, "images/%s/Recons_epoch_%d_%d.png" % (opt.dataset_name, epoch, i), nrow=8, normalize=False)

    generator.train()


# measurement method to produce real_A from real_B
if opt.mask == None:
    mask = opt.mask
else: 
    mask = torch.load(opt.mask) # (1 ,1, 256, 1, 1)

mask = mask.transpose(-2,-3) # (1 ,1, 1, 256, 1)

to_cyc = To_h_space(mask=mask)
to_k = To_k_space(mask=mask)
# to_cyc = To_h_space(mask=None, center_fractions=[0.04], accelerations=[8]) # sampling pattern diversity
# to_k = To_k_space(mask=None, center_fractions=[0.04], accelerations=[8])

soft_dc = Soft_Data_Consistency(mask=mask.squeeze(dim=-1)) # DC opeerator

# hierarchical sub-sampling supervision operators
def make_random_sample(p, size):
    ''' p: sampling rate, between 0. and 1. '''
    mask = torch.Tensor()
    P = torch.rand(size)
    mask = torch.where(P>p, torch.tensor([0.]), torch.tensor([1.]))
    return mask

# multi-coil DC operator    
def multi_coil_dc(inputs, zero_filled, CSM):
    if isinstance(zero_filled, list):
        outputs = [complex_product(inputs, m) for m in CSM] # coil projection
        outputs = [soft_dc(x, x_dc) for x, x_dc in zip(outputs, zero_filled)] # data consistency
        outputs = coil_combine(outputs, CSM)  # coil combination
    else:
        outputs = soft_dc(inputs, zero_filled) # data consistency
    return outputs

# Loss functions
# mae_loss = torch.nn.MSELoss()
mae_loss = torch.nn.L1Loss()

def local_weighted_L1Loss(inputs, label, kernel_size=(8,8), beta=20.):
    ''' inputs (B, C, H, W) '''
    diff = (inputs - label).abs()
    diff = nn.Unfold(kernel_size=kernel_size, stride=kernel_size)(diff) # (B, C*np.prod(kernel_size), L)
    diff_mean = diff.mean(dim=1, keepdim=True) # (B, 1, L)
    diff_modulation = nn.functional.softmax(beta*diff_mean, dim=-1).detach()
    diff_mean = (diff_modulation*diff_mean).sum(dim=-1).squeeze(dim=1).mean()
    return diff_mean

eps = 1e-12
Smooth_L1 = lambda output, target: torch.sqrt((output - target)**2+eps).mean()
sig2square = lambda input: (input[:,[0]]**2 + input[:,[1]]**2) # output squared magnitude maps of shape (B, 1, H, W)
sig2rss = lambda input_list: torch.sqrt(sum([sig2square(x) for x in input_list])) # input is multi-coil data list whose elements are of shape (B, 2, H, W), and mapped to be (B, 1, H, W)
sig2abs = lambda input: torch.sqrt(sig2square(input)) 

complex_product = lambda a, b: torch.cat([a[:,[0]]*b[:,[0]]-a[:,[1]]*b[:,[1]], a[:,[1]]*b[:,[0]]+a[:,[0]]*b[:,[1]]], dim=1) # a*b.conj where a and b are of shape (B, 2, H, W)

conj_product = lambda a, b: torch.cat([a[:,[0]]*b[:,[0]]+a[:,[1]]*b[:,[1]], a[:,[1]]*b[:,[0]]-a[:,[0]]*b[:,[1]]], dim=1) # a*b.conj where a and b are of shape (B, 2, H, W)

def coil_combine(org, csm):
    ''' org and csm [..., (B, 2, H, W), ...], and return the combination of shape (B, 2, H, W) '''
    conj_proj = [conj_product(a,b) for a, b in zip(org, csm)]
    return sum(conj_proj)

def sig2normedabs(input, return_max=False):
    ''' input (B, 2, H, W), output normed magnitude map of shape (B, 1, H, W) '''
    output = sig2abs(input)
    _, _, H_in, W_in = output.shape

    # normalize
    output = output.flatten(start_dim=1,end_dim=-1) # (B, H*W)
    output_max = output.max(dim=-1, keepdim=True)[0]
    output = output / (output_max + 1.e-6)
    output = 2. * output - 1. # (-1, 1)

    output = output.contiguous().view(-1,1,H_in, W_in) # (B, 1, H, W)

    if return_max:
        return output, output_max.contiguous().view(-1, 1, 1, 1)
    else:
        return output

def normedabs(input, return_max=False):
    ''' input (B, 1, H, W), output normalized magnitude map of shape (B, 1, H, W) '''
    _, _, H_in, W_in = input.shape

    # normalize
    output = input.flatten(start_dim=1,end_dim=-1) # (B, H*W)
    output_max = output.max(dim=-1, keepdim=True)[0]
    output = output / (output_max + 1.e-6)
    output = 2. * output - 1. # (-1, 1)

    output = output.contiguous().view(-1,1,H_in, W_in) # (B, 1, H, W)

    if return_max:
        return output, output_max.contiguous().view(-1, 1, 1, 1)
    else:
        return output

# Initialize generator, encoder and discriminators
generator = Multi_Level_Dense_Network(img_shape=(1,256,256), out_channel=1, stages=opt.stage)

D_VAE =  RA_MultiDiscriminator_CBAM([1, *input_shape[2:]], p=0.1)

# VGGs
vgg = models.vgg11_bn(pretrained=True).features[:19].cuda()
for param in vgg.parameters(): 
    param.requires_grad = False # no longer parameter(), but can receive and transmit gradients; it saves computational costs and memory

VGGList = nn.ModuleList()
VGGList.add_module('vgg_0', vgg[:9])
VGGList.add_module('vgg_1', vgg[9:12])
VGGList.add_module('vgg_2', vgg[12:16])
VGGList.add_module('vgg_3', vgg[16:])


if cuda:
    generator = generator.cuda()
    D_VAE = D_VAE.cuda()
    mae_loss.cuda()
    
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

def lr_scheduler(optimizer, epoch):
    print('LR is now {}'.format(optimizer.param_groups[0]['lr']))

    """Decay learning rate by a factor of 0.5 every 5000."""
    if (epoch-1) % 10 == 0 and epoch > 10:
        power = (epoch-1)/10
        for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr * (0.8**power)
        print('LR is set to {}'.format(param_group['lr']))

if opt.epoch != 0:
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch), map_location='cpu'), strict=False)
    D_VAE.load_state_dict(torch.load("saved_models/%s/D_VAE_%d.pth" % (opt.dataset_name, opt.epoch), map_location='cpu'))

    # Load pretrained models
    # optimizer_G.load_state_dict(torch.load("saved_models/%s/optimizer_G_%d.pth" % (opt.dataset_name, opt.epoch), map_location='cpu'))
    # optimizer_D_VAE.load_state_dict(torch.load("saved_models/%s/optimizer_D_VAE_%d.pth" % (opt.dataset_name, opt.epoch), map_location='cpu'))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# prepare dataset
loader_params = {'batch_size': opt.batch_size, 'shuffle': True, 'num_workers': 2}
test_params = {'batch_size': 5, 'shuffle': False, 'num_workers': 1}
structure=['csm', 'multi_coil']

start_ = 100

# create dataloaders for training and validation
dataset = HDF5Dataset(opt.dataroot, recursive=True, load_data=False, data_cache_size=2, transform=None, structure=structure)
dataloader = DataLoader(dataset, **loader_params)

testset = HDF5Dataset(opt.testroot, recursive=True, load_data=False, data_cache_size=2, transform=None, structure=structure)
testloader = DataLoader(testset, **test_params)

# ----------
#  Training
# ----------
if __name__ == '__main__':
    # Adversarial loss
    valid = 1.
    fake = 0.

    _, [testCsm, testBatch] = next(enumerate(testloader))

    del testset, testloader

    testBatch = testBatch.type(Tensor)
    testCsm = testCsm.type(Tensor)

    prev_time = time.time()
    for epoch in range(opt.epoch+1, opt.n_epochs+opt.epoch+1):
        # sample_images(epoch=opt.epoch, i=20000)
        # break

        lr_scheduler(optimizer_G, epoch) # scheduler
        lr_scheduler(optimizer_D_VAE, epoch)

        for i, [csm, batch] in enumerate(dataloader): # (B,15,2,256,256)
            batch, csm = Variable(batch.type(Tensor)), Variable(csm.type(Tensor)) # to CUDA
            batch, csm = torch.split(batch, 1, dim=1), torch.split(csm, 1, dim=1) 
            batch, csm = [x.squeeze(dim=1) for x in batch], [x.squeeze(dim=1) for x in csm] # [..., (B,2,256,256), ...]

            if opt.epoch>0 and epoch == (opt.epoch+1) and i == 0:
                # Load pretrained models
                optimizer_G.load_state_dict(torch.load("saved_models/%s/optimizer_G_%d.pth" % (opt.dataset_name, opt.epoch), map_location='cpu'))
                optimizer_D_VAE.load_state_dict(torch.load("saved_models/%s/optimizer_D_VAE_%d.pth" % (opt.dataset_name, opt.epoch), map_location='cpu'))

            optimizer_G.zero_grad()

            with torch.cuda.amp.autocast(enabled=False):
                # Set model input
                real_B = batch # [..., (B,2,256,256), ...]

                real_A = [to_cyc(full).detach() for full in batch] # under-sample coil views
                real_DC = [x.clone() for x in real_A] # [..., (B,2,256,256), ...]
                real_K = [to_k(full).detach() for full in batch] # for DC in k-space

                real_A, full = sig2rss(real_A), sig2rss(real_B) # RSS, (B, 1, H, W)
                real_A, full = normedabs(real_A), normedabs(full) # (B, 1, H, W), (-1,1)

                del batch

                # -------------------------------
                #  Train Generator and Encoder
                # -------------------------------

                # ----------
                # cVAE-GAN
                # ----------

                # Produce output using real_A (cVAE-GAN)
                fake_B, fake_B_hierarchi = generator(real_A.unsqueeze(dim=2), zero_filled=sum(real_DC), csm=csm, dc_operator=multi_coil_dc)
                fake_B = fake_B.squeeze(dim=2) # (B, 2, H, W)
                
                '''non-uniform mean'''
                # Pixelwise loss of translated image by VAE
                alpha = 0.64 # 0.84

                # L-1
                L1_loss = mae_loss(fake_B, full)
                # SSIM
                MS_SSIM_Loss = 1. - get_ssim((fake_B+1.)/2., (full+1.)/2.) # SSIM
                
                # total pixel loss
                loss_pixel = (1-alpha)*L1_loss + alpha*MS_SSIM_Loss

                '''L_adv and L_vgg'''

                # Adversarial loss
                loss_VAE_GAN = D_VAE.compute_loss(full, fake_B, valid=fake, fake=valid, sg=False) # relativistic average

                # Total Loss (Generator + Encoder)
                loss_GE = opt.lambda_adv*loss_VAE_GAN + opt.lambda_pixel * loss_pixel

                # VGG loss
                loss_VGG = torch.Tensor(1).fill_(0.).type(Tensor)
                content_loss, gram_loss = [0., 0., 0.], [0., 0., 0.]
                lambda_gram = 0.005

                real_content = (full + 1.) / 2. # (B, 1, H, W), (0, 1)
                fake_content = (fake_B + 1.) / 2.

                real_content = real_content.expand(-1,3,-1,-1) # (B, 3, H, W)
                fake_content = fake_content.expand(-1,3,-1,-1)
                
                content_loss = []
                gram_loss = []
                lambda_gram = 0.005

                weight_list = [1., 1.5, 3., 4.5]

                for k, m in enumerate(VGGList):
                    real_content = m(real_content).detach()
                    fake_content = m(fake_content)

                    real_vgg = real_content.clone()
                    fake_vgg = fake_content.clone()

                    # content_loss += [nn.L1Loss()(real_vgg, fake_vgg)]
                    content_loss += [mae_loss(real_vgg, fake_vgg)]

                    # gram matrices
                    gram_real = real_vgg.view(real_vgg.shape[0],real_vgg.shape[1],-1) @ real_vgg.view(real_vgg.shape[0],real_vgg.shape[1],-1).transpose(-2,-1)
                    gram_fake = fake_vgg.view(fake_vgg.shape[0],fake_vgg.shape[1],-1) @ fake_vgg.view(fake_vgg.shape[0],fake_vgg.shape[1],-1).transpose(-2,-1)

                    gram_loss += [weight_list[k]*mae_loss(gram_real, gram_fake)]

                loss_VGG = sum(content_loss) + lambda_gram*sum(gram_loss)
                loss_VGG *= opt.lambda_vgg

                del real_content, fake_content, real_vgg, fake_vgg, gram_real, gram_fake

                loss_G = loss_GE + loss_VGG 

            loss_G.backward()
            optimizer_G.step()

            optimizer_D_VAE.zero_grad()

            # clone_B = torch.ones(fake_B.shape).cuda()
            # clone_B.copy_(fake_B)
            # clone_B = fake_B.new_tensor(fake_B)
            with torch.cuda.amp.autocast(enabled=False):
                loss_D_VAE = D_VAE.compute_loss(full, fake_B.detach(), valid=valid, fake=fake, sg=True) # relativistic average

                loss_D_VAE *= opt.lambda_adv 

                # gradient penalty
                loss_grad_VAE = 0.
                loss_grad_VAE = 10.*D_VAE.compute_gradient_penalty(full, fake_B.detach()) # gradient penalty
                loss_grad_VAE *= opt.lambda_adv

                loss_D = loss_D_VAE + loss_grad_VAE

            loss_D.backward()
            optimizer_D_VAE.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            sys.stdout.write(
                "\r[E %d/%d, %d/%d] [D: (%.3f, %.3f)] [G: (%.3f), pixel: (%.3f, %.3f), vgg: (%.3f, %.3f, %.3f), (%.3f, %.3f, %.3f)] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D_VAE.item(),
                    loss_grad_VAE,

                    loss_GE.item()-opt.lambda_pixel * loss_pixel.item(),

                    opt.lambda_pixel*(1-alpha)*L1_loss.item(),
                    opt.lambda_pixel*alpha*MS_SSIM_Loss.item(),

                    opt.lambda_vgg*content_loss[0],
                    opt.lambda_vgg*content_loss[1],
                    opt.lambda_vgg*content_loss[2],

                    opt.lambda_vgg*lambda_gram*gram_loss[0],
                    opt.lambda_vgg*lambda_gram*gram_loss[1],
                    opt.lambda_vgg*lambda_gram*gram_loss[2],
                    time_left,
                )
            )

            if batches_done % opt.sample_interval == 0:
                sample_images(epoch, i)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:

            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_VAE.state_dict(), "saved_models/%s/D_VAE_%d.pth" % (opt.dataset_name, epoch))
            torch.save(optimizer_G.state_dict(), "saved_models/%s/optimizer_G_%d.pth" % (opt.dataset_name, epoch))
            torch.save(optimizer_D_VAE.state_dict(), "saved_models/%s/optimizer_D_VAE_%d.pth" % (opt.dataset_name, epoch))

