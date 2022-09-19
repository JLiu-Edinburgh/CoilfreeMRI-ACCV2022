import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

import torch.autograd as autograd

from .selection_networks import CAM_Module
from .esrgan import ResidualInResidualDenseBlock

class MultiDiscriminator_CBAM(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalize=True, p=None):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                # layers.append(nn.InstanceNorm2d(out_filters, 0.8))
                layers.append(nn.InstanceNorm2d(out_filters))
                # layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            if p is not None:
                layers.append(nn.Dropout2d(p=p))

            return layers

        class ResCBAM(nn.Module):
            def __init__(self, in_channels, beta=1.):
                super().__init__()
                self.cbam = CBAM(in_channels)
                self.beta = beta
            def forward(self, x):
                # return x + self.beta*self.cbam(x)
                return x

        class Mixed_Dropout2d(nn.Sequential):
            def __init__(self, p1=0.2, p2=None):
                super().__init__()
                self.p2 = p1 if p2 is None else p2

                self.add_module('dropout2d', nn.Dropout2d(p=p1))
                self.add_module('spatial_dropout', Spatial_Dropout(p=self.p2))

        class Spatial_Dropout(nn.Module):
            def __init__(self, p=0.1):
                super().__init__()
                self.dropout = nn.Dropout2d(p=p)
            def forward(self, x):
                _, c, h, w = x.shape
                x = x.view(-1,c,h*w,1)
                x = x.permute((0,2,1,-1)) # (B, h*w, c, 1)
                x = self.dropout(x)
                x = x.transpose(1,2) # (B, c, h*w, 1)
                return x.view(-1,c,h,w) 

        channels, _, _ = input_shape
        # Extracts discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    # nn.Conv2d(channels, 32, 3, stride=1, padding=1), # 256
                    nn.Conv2d(channels, 32, 4, stride=2, padding=1), # 256
                    # ResCBAM(32),
                    
                    *discriminator_block(32, 64, normalize=True), # 128
                    # ResCBAM(64, beta=0.2),
                    Mixed_Dropout2d(p1=0., p2=0.0),
                    # Spatial_Dropout(p=0.2),
                    # CAM_Module(64, 0.2),

                    *discriminator_block(64, 128), # 64 and dropout activations
                    CAM_Module(128),
                    Mixed_Dropout2d(p1=0.1, p2=0.1),

                    *discriminator_block(128, 256), # 32
                    CAM_Module(256),
                    Mixed_Dropout2d(p1=0.1, p2=0.1),

                    *discriminator_block(256, 512), # 16
                    # ResCBAM(512),
                    # CAM_Module(512),
                    nn.Conv2d(512, 1, 3, padding=1), # 16
                    # nn.Sigmoid(),
                )
                # nn.Sequential(
                #     nn.Conv2d(channels, 32, 4, stride=2, padding=1), # no norm after the first conv 
                #     *discriminator_block(32, 64, normalize=True),
                #     *discriminator_block(64, 128),
                #     *discriminator_block(128, 256),
                #     *discriminator_block(256, 512),
                #     nn.Conv2d(512, 1, 3, padding=1)
                # ), # result in bluriness
            ) # downsample by 2^4

        class Downsample(nn.Module):
            def __init__(self, scale_factor, mode='bilinear'):
                super().__init__()
                self.scale_factor = scale_factor # scale_h_w or (scale_h, scale_w), munipulate 2D images
                self.mode = mode

            def forward(self, input):
                return nn.functional.interpolate(input, scale_factor=self.scale_factor, mode=self.mode)
                
        self.downsample = Downsample(scale_factor=0.5)
        # self.downsample = nn.AvgPool2d(channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        device= real_samples.device
        # Random weight term for interpolation between real and fake samples
    #    alpha = torch.Tensor(np.random.uniform(0., 0.2, (real_samples.size(0), 1, 1, 1))).to(device)
    #    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True).to(device)
    #    for i in range(0): # sample more points
    #        
    #        # Get random interpolation between real and fake samples
    #        interpolates = torch.cat([interpolates, (alpha * real_samples + (1 - alpha) * fake_samples)], dim=0)
    #    
    #    alpha = torch.Tensor(np.random.uniform(0.8, 1., (real_samples.size(0), 1, 1, 1))).to(device)  
    #    interpolates = torch.cat([interpolates, (alpha * real_samples + (1 - alpha) * fake_samples)], dim=0)
        
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True).to(device)
        # for i in range(1):
        #     alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
        #     interpolates = torch.cat([interpolates, (alpha * real_samples + (1 - alpha) * fake_samples)], dim=0)
        
        # outputs
        # outputs = self.forward(interpolates)

        # gradient
        gradients = []
        d_interpolates = []
        fake = []
        gradient_penalty = None
        for i, D in enumerate(self.models):
            d_interpolates.append(D(interpolates).mean(dim=(1,2,3))) # assume D returns (B,1,h,w)
            d_interpolates[i] = d_interpolates[i].unsqueeze(dim=1)
        #    fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
            fake.append(torch.ones_like(d_interpolates[i]).to(device))
            # Get gradient w.r.t. interpolates
            '''
            inputs = inputs.requires_grad_(True) is needed,

            compute gradients w.r.t inputs and integrate them using coefficients from grad_outputs (of the same shape of outputs), hence 'gradients.shape = inputs.shape';
            
            'create_graph = True' to compute autograd.grad(outputs=gradients,...) after 'gradients = autograd.grad(outputs=outputs,...)', noting 'retain_graph = True' is also needed.

            Moreover, if only_inputs is True, the function will only return a list of gradients w.r.t the specified inputs. If it’s False, then gradient w.r.t. all remaining leaves will still be computed, and will be accumulated into their .grad attribute.
            '''
            gradients.append(autograd.grad(
                outputs=d_interpolates[i],
                inputs=interpolates,
                grad_outputs=fake[i],
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0])

            gradients[i] = gradients[i].view(gradients[i].size(0), -1)
            
            gradient_penalty = ((gradients[i].norm(2, dim=1) - 1) ** 2).mean() if gradient_penalty is None else gradient_penalty + ((gradients[i].norm(2, dim=1) - 1) ** 2).mean()

            interpolates = self.downsample(interpolates)
        
        return gradient_penalty/(i+1)    

class MultiDiscriminator_shared_CBAM(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalize=True, p=None):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                # layers.append(nn.InstanceNorm2d(out_filters, 0.8))
                layers.append(nn.InstanceNorm2d(out_filters))
                # layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            if p is not None:
                layers.append(nn.Dropout2d(p=p))

            return layers

        class ResCBAM(nn.Module):
            def __init__(self, in_channels, beta=1.):
                super().__init__()
                self.cbam = CBAM(in_channels)
                self.beta = beta
            def forward(self, x):
                # return x + self.beta*self.cbam(x)
                return x

        class Mixed_Dropout2d(nn.Sequential):
            def __init__(self, p1=0.2, p2=None):
                super().__init__()
                self.p2 = p1 if p2 is None else p2

                self.add_module('dropout2d', nn.Dropout2d(p=p1))
                self.add_module('spatial_dropout', Spatial_Dropout(p=self.p2))

        class Spatial_Dropout(nn.Module):
            def __init__(self, p=0.1):
                super().__init__()
                self.dropout = nn.Dropout2d(p=p)
            def forward(self, x):
                _, c, h, w = x.shape
                x = x.view(-1,c,h*w,1)
                x = x.permute((0,2,1,-1)) # (B, h*w, c, 1)
                x = self.dropout(x)
                x = x.transpose(1,2) # (B, c, h*w, 1)
                return x.view(-1,c,h,w) 

        channels, _, _ = input_shape
        # Extracts discriminator models
        self.backbone = nn.Sequential(
                    nn.Conv2d(channels, 32, 3, stride=1, padding=1), # 256
                    
                    *discriminator_block(32, 64, normalize=True), # 128
                    Mixed_Dropout2d(p1=0., p2=0.0),

                    *discriminator_block(64, 128), # 64 and dropout activations
                    CAM_Module(128),
                    Mixed_Dropout2d(p1=0.1, p2=0.1),

                    *discriminator_block(128, 256), # 32
                    CAM_Module(256),
                    Mixed_Dropout2d(p1=0.1, p2=0.1),
                     )
        self.models = nn.ModuleList()
        self.tails = nn.ModuleList()

        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(256, 256), # downsampling block
                )
            )
            self.tails.add_module(
                "tail_%d" % i,
                nn.Sequential(
                    nn.Conv2d(256, 1, 3, padding=1), # output single channel
                )
            )    

        class Downsample(nn.Module):
            def __init__(self, scale_factor, mode='bilinear'):
                super().__init__()
                self.scale_factor = scale_factor # scale_h_w or (scale_h, scale_w), munipulate 2D images
                self.mode = mode

            def forward(self, input):
                return nn.functional.interpolate(input, scale_factor=self.scale_factor, mode=self.mode)
                
        self.downsample = Downsample(scale_factor=0.5)
        # self.downsample = nn.AvgPool2d(channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        x = self.backbone(x)
        outputs = []
        for m, t in zip(self.models, self.tails):
            x = m(x)
            outputs.append(t(x.clone()))
        return outputs

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        device= real_samples.device
        # Random weight term for interpolation between real and fake samples
    #    alpha = torch.Tensor(np.random.uniform(0., 0.2, (real_samples.size(0), 1, 1, 1))).to(device)
    #    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True).to(device)
    #    for i in range(0): # sample more points
    #        
    #        # Get random interpolation between real and fake samples
    #        interpolates = torch.cat([interpolates, (alpha * real_samples + (1 - alpha) * fake_samples)], dim=0)
    #    
    #    alpha = torch.Tensor(np.random.uniform(0.8, 1., (real_samples.size(0), 1, 1, 1))).to(device)  
    #    interpolates = torch.cat([interpolates, (alpha * real_samples + (1 - alpha) * fake_samples)], dim=0)
        
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True).to(device)
        # for i in range(1):
        #     alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
        #     interpolates = torch.cat([interpolates, (alpha * real_samples + (1 - alpha) * fake_samples)], dim=0)
        
        # outputs
        # outputs = self.forward(interpolates)

        # gradient
        gradients = []
        d_interpolates = []
        fake = []
        gradient_penalty = None

        outputs = self.forward(interpolates)
        for i, D in enumerate(outputs):
            d_interpolates.append(D.mean(dim=(1,2,3))) # assume D returns (B,1,h,w)
            d_interpolates[i] = d_interpolates[i].unsqueeze(dim=1) # (B,1)
            fake.append(torch.ones_like(d_interpolates[i]).to(device))
            # Get gradient w.r.t. interpolates
            '''
            inputs = inputs.requires_grad_(True) is needed,

            compute gradients w.r.t inputs and integrate them using coefficients from grad_outputs (of the same shape of outputs), hence 'gradients.shape = inputs.shape';
            
            'create_graph = True' to compute autograd.grad(outputs=gradients,...) after 'gradients = autograd.grad(outputs=outputs,...)', noting 'retain_graph = True' is also needed.

            Moreover, if only_inputs is True, the function will only return a list of gradients w.r.t the specified inputs. If it’s False, then gradient w.r.t. all remaining leaves will still be computed, and will be accumulated into their .grad attribute.
            '''
            gradients.append(autograd.grad(
                outputs=d_interpolates[i],
                inputs=interpolates,
                grad_outputs=fake[i],
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0])

            gradients[i] = gradients[i].view(gradients[i].size(0), -1)
            
            gradient_penalty = ((gradients[i].norm(2, dim=1) - 1) ** 2).mean() if gradient_penalty is None else gradient_penalty + ((gradients[i].norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty/(i+1)

class RA_MultiDiscriminator(nn.Module):
    '''relativistic average discriminator based on LSGAN '''
    def __init__(self, input_shape):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalize=True, p=None):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                # layers.append(nn.InstanceNorm2d(out_filters, 0.8))
                layers.append(nn.InstanceNorm2d(out_filters))
                # layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            if p is not None:
                layers.append(nn.Dropout2d(p=p))

            return layers

        class ResCBAM(nn.Module):
            def __init__(self, in_channels, beta=1.):
                super().__init__()
                self.cbam = CBAM(in_channels)
                self.beta = beta
            def forward(self, x):
                # return x + self.beta*self.cbam(x)
                return x

        class Mixed_Dropout2d(nn.Sequential):
            def __init__(self, p1=0.2, p2=None):
                super().__init__()
                self.p2 = p1 if p2 is None else p2

                self.add_module('dropout2d', nn.Dropout2d(p=p1))
                self.add_module('spatial_dropout', Spatial_Dropout(p=self.p2))

        class Spatial_Dropout(nn.Module):
            def __init__(self, p=0.1):
                super().__init__()
                self.dropout = nn.Dropout2d(p=p)
            def forward(self, x):
                _, c, h, w = x.shape
                x = x.view(-1,c,h*w,1)
                x = x.permute((0,2,1,-1)) # (B, h*w, c, 1)
                x = self.dropout(x)
                x = x.transpose(1,2) # (B, c, h*w, 1)
                return x.view(-1,c,h,w) 

        channels, _, _ = input_shape
        # Extracts discriminator models
        self.backbone = nn.Sequential(
                    nn.Conv2d(channels, 32, 3, stride=1, padding=1), # 256
                    
                    *discriminator_block(32, 64, normalize=True), # 128
                    Mixed_Dropout2d(p1=0., p2=0.0),

                    *discriminator_block(64, 128), # 64 and dropout activations
                    CAM_Module(128),
                    Mixed_Dropout2d(p1=0.1, p2=0.1),

                    *discriminator_block(128, 256), # 32
                    CAM_Module(256),
                    Mixed_Dropout2d(p1=0.1, p2=0.1),
                     )
        self.models = nn.ModuleList()
        self.tails = nn.ModuleList()

        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(256, 256), # downsampling block
                )
            )
            self.tails.add_module(
                "tail_%d" % i,
                nn.Sequential(
                    nn.Conv2d(256, 1, 3, padding=1), # output single channel
                )
            )    

        class Downsample(nn.Module):
            def __init__(self, scale_factor, mode='bilinear'):
                super().__init__()
                self.scale_factor = scale_factor # scale_h_w or (scale_h, scale_w), munipulate 2D images
                self.mode = mode

            def forward(self, input):
                return nn.functional.interpolate(input, scale_factor=self.scale_factor, mode=self.mode)
                
        self.downsample = Downsample(scale_factor=0.5)
        # self.downsample = nn.AvgPool2d(channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, y, valid=1., fake=0., sg=False):
        """Computes the MSE between model output and scalar gt"""
        # loss = sum([torch.mean((out_0 - valid) ** 2 + (out_1 - fake) ** 2) for out_0, out_1 in zip(*self.forward(x,y,sg))])
        if not sg:
            loss = sum([torch.mean((out_1 - fake) ** 2) for out_1 in self.forward(x,y,sg)[1]])
        else:
            loss = sum([torch.mean((out_0 - valid) ** 2 + (out_1 - fake) ** 2) for out_0, out_1 in zip(*self.forward(x,y,sg))])
        return loss

    def forward(self, x, y=None, sg=False):
        '''sg: False for updating D and True for updating G'''
        x = self.backbone(x)
        if y is None:
            outputs = []
            for m, t in zip(self.models, self.tails):
                x = m(x)
                outputs.append(t(x.clone()))

            return outputs
        else:
            y = self.backbone(y).detach() if sg else self.backbone(y)

            outputs_0 = []
            outputs_1 = []
            # if not sg: # relativity in feature levels
            #     for m, t in zip(self.models, self.tails):
            #         x = m(x)
            #         y = m(y)
            #         outputs_0.append(t(x-y.mean(dim=0, keepdim=True)))
            #         outputs_1.append(t(y-x.mean(dim=0, keepdim=True)))
            # else:
            #     for m, t in zip(self.models, self.tails):
            #         x = m(x)
            #         y = m(y).detach()
            #         outputs_0.append(t(x-y.mean(dim=0, keepdim=True)))
            #         outputs_1.append(t(y-x.mean(dim=0, keepdim=True)))

            # if not sg: # relativity in scores
            #     for m, t in zip(self.models, self.tails):
            #         x = m(x)
            #         y = m(y)
            #         x_score = t(x.clone())
            #         y_score = t(y.clone())
            #         outputs_0.append(x_score-y_score.mean(dim=0, keepdim=True))
            #         outputs_1.append(y_score-x_score.mean(dim=0, keepdim=True))
            # else:
            #     for m, t in zip(self.models, self.tails):
            #         x = m(x)
            #         y = m(y).detach()
            #         x_score = t(x.clone())
            #         y_score = t(y.clone()).detach()
            #         outputs_0.append(x_score-y_score.mean(dim=0, keepdim=True))
            #         outputs_1.append(y_score-x_score.mean(dim=0, keepdim=True))

            if not sg: # relativity in scores
                for m, t in zip(self.models, self.tails):
                    x = m(x).detach()
                    y = m(y)
                    x_score = t(x.clone()).detach()
                    y_score = t(y.clone())
                    outputs_1.append(y_score-x_score.mean(dim=0, keepdim=True))
            else:
                for m, t in zip(self.models, self.tails):
                    x = m(x)
                    y = m(y)
                    x_score = t(x.clone())
                    y_score = t(y.clone())
                    outputs_0.append(x_score-y_score.mean(dim=0, keepdim=True))
                    outputs_1.append(y_score-x_score.mean(dim=0, keepdim=True)) 

            return outputs_0, outputs_1

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        device= real_samples.device
        # Random weight term for interpolation between real and fake samples
    #    alpha = torch.Tensor(np.random.uniform(0., 0.2, (real_samples.size(0), 1, 1, 1))).to(device)
    #    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True).to(device)
    #    for i in range(0): # sample more points
    #        
    #        # Get random interpolation between real and fake samples
    #        interpolates = torch.cat([interpolates, (alpha * real_samples + (1 - alpha) * fake_samples)], dim=0)
    #    
    #    alpha = torch.Tensor(np.random.uniform(0.8, 1., (real_samples.size(0), 1, 1, 1))).to(device)  
    #    interpolates = torch.cat([interpolates, (alpha * real_samples + (1 - alpha) * fake_samples)], dim=0)
        
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True).to(device)
        # for i in range(1):
        #     alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
        #     interpolates = torch.cat([interpolates, (alpha * real_samples + (1 - alpha) * fake_samples)], dim=0)
        
        # outputs
        # outputs = self.forward(interpolates)

        # gradient
        gradients = []
        d_interpolates = []
        fake = []
        gradient_penalty = None

        outputs = self.forward(interpolates) # non-relativistic forward
        for i, D in enumerate(outputs):
            d_interpolates.append(D.mean(dim=(1,2,3))) # assume D returns (B,1,h,w)
            d_interpolates[i] = d_interpolates[i].unsqueeze(dim=1) # (B,1)
            fake.append(torch.ones_like(d_interpolates[i]).to(device))
            # Get gradient w.r.t. interpolates
            '''
            inputs = inputs.requires_grad_(True) is needed,

            compute gradients w.r.t inputs and integrate them using coefficients from grad_outputs (of the same shape of outputs), hence 'gradients.shape = inputs.shape';
            
            'create_graph = True' to compute autograd.grad(outputs=gradients,...) after 'gradients = autograd.grad(outputs=outputs,...)', noting 'retain_graph = True' is also needed.

            Moreover, if only_inputs is True, the function will only return a list of gradients w.r.t the specified inputs. If it’s False, then gradient w.r.t. all remaining leaves will still be computed, and will be accumulated into their .grad attribute.
            '''
            gradients.append(autograd.grad(
                outputs=d_interpolates[i],
                inputs=interpolates,
                grad_outputs=fake[i],
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0])

            gradients[i] = gradients[i].view(gradients[i].size(0), -1)
            
            gradient_penalty = ((gradients[i].norm(2, dim=1) - 1) ** 2).mean() if gradient_penalty is None else gradient_penalty + ((gradients[i].norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty/(i+1)    


class RA_MultiDiscriminator_CBAM(nn.Module):
    '''relativistic average discriminator based on LSGAN '''
    def __init__(self, input_shape, p=0.1):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalize=True, p=None):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                # layers.append(nn.InstanceNorm2d(out_filters, 0.8))
                layers.append(nn.InstanceNorm2d(out_filters))
                # layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            if p is not None:
                layers.append(nn.Dropout2d(p=p))

            return layers

        class ResCBAM(nn.Module):
            def __init__(self, in_channels, beta=1.):
                super().__init__()
                self.cbam = CBAM(in_channels)
                self.beta = beta
            def forward(self, x):
                # return x + self.beta*self.cbam(x)
                return x

        class Mixed_Dropout2d(nn.Sequential):
            def __init__(self, p1=0.2, p2=None):
                super().__init__()
                self.p2 = p1 if p2 is None else p2

                self.add_module('dropout2d', nn.Dropout2d(p=p1))
                self.add_module('spatial_dropout', Spatial_Dropout(p=self.p2))

        class Spatial_Dropout(nn.Module):
            def __init__(self, p=0.1):
                super().__init__()
                self.dropout = nn.Dropout2d(p=p)
            def forward(self, x):
                _, c, h, w = x.shape
                x = x.view(-1,c,h*w,1)
                x = x.permute((0,2,1,-1)) # (B, h*w, c, 1)
                x = self.dropout(x)
                x = x.transpose(1,2) # (B, c, h*w, 1)
                return x.view(-1,c,h,w) 

        channels, _, _ = input_shape
        # Extracts discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    # nn.Conv2d(channels, 32, 3, stride=1, padding=1), # 256
                    nn.Conv2d(channels, 32, 4, stride=2, padding=1), # 256
                    # ResCBAM(32),
                    
                    *discriminator_block(32, 64, normalize=True), # 128
                    # ResCBAM(64, beta=0.2),
                    Mixed_Dropout2d(p1=0., p2=0.0),
                    # Spatial_Dropout(p=0.2),
                    # CAM_Module(64, 0.2),

                    *discriminator_block(64, 128), # 64 and dropout activations
                    CAM_Module(128),
                    Mixed_Dropout2d(p1=p, p2=p),

                    *discriminator_block(128, 256), # 32
                    CAM_Module(256),
                    Mixed_Dropout2d(p1=p, p2=p),

                    *discriminator_block(256, 512), # 16
                    # ResCBAM(512),
                    # CAM_Module(512),
                    nn.Conv2d(512, 1, 3, padding=1), # 16
                    # nn.Sigmoid(),
                )
            ) # downsample by 2^4

        class Downsample(nn.Module):
            def __init__(self, scale_factor, mode='bilinear'):
                super().__init__()
                self.scale_factor = scale_factor # scale_h_w or (scale_h, scale_w), munipulate 2D images
                self.mode = mode

            def forward(self, input):
                return nn.functional.interpolate(input, scale_factor=self.scale_factor, mode=self.mode)
                
        self.downsample = Downsample(scale_factor=0.5)
        # self.downsample = nn.AvgPool2d(kernel_size=4, stride=2, padding=1, count_include_pad=True)

    def compute_loss(self, x, y, valid=1., fake=0., sg=False):
        """Computes the MSE between model output and scalar gt"""
        if not sg:
            loss = sum([torch.mean((out_1 - fake) ** 2) for out_1 in self.forward(x,y,sg)[1]])
        else:
            loss = sum([torch.mean((out_0 - valid) ** 2 + (out_1 - fake) ** 2) for out_0, out_1 in zip(*self.forward(x,y,sg))])
        return loss

    def forward(self, x, y=None, sg=False):
        '''sg: False for updating D and True for updating G'''
        # device = x.device
        if y is None:
            output1 = []
            for m in self.models:
                output1.append(m(x))
                x = self.downsample(x)
            return None, output1
        else:
            outputs_0 = []
            outputs_1 = []

            if not sg: # for G loss
                for m in self.models:
                    x_score = m(x).detach()
                    y_score = m(y)

                    # x_score_mean = torch.ones((1, *x_score.shape[1:])).to(device)
                    # x_score_mean.copy_(x_score.mean(dim=0, keepdim=True))

                    # outputs_1.append(y_score-x_score_mean.detach())

                    outputs_1.append(y_score-x_score.mean(dim=0, keepdim=True))

                    x = self.downsample(x)
                    y = self.downsample(y)
            else: # for D loss
                for m in self.models:
                    x_score = m(x)
                    y_score = m(y)

                    # x_score_mean = torch.ones((1, *x_score.shape[1:])).to(device)
                    # y_score_mean = torch.ones((1, *y_score.shape[1:])).to(device)
                    # x_score_mean.copy_(x_score.mean(dim=0, keepdim=True))
                    # y_score_mean.copy_(y_score.mean(dim=0, keepdim=True))

                    # outputs_0.append(x_score-y_score_mean.detach())
                    # outputs_1.append(y_score-x_score_mean.detach())

                    outputs_0.append(x_score-y_score.mean(dim=0, keepdim=True))
                    outputs_1.append(y_score-x_score.mean(dim=0, keepdim=True))

                    x = self.downsample(x)
                    y = self.downsample(y)   

            return outputs_0, outputs_1

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        device= real_samples.device
        # Random weight term for interpolation between real and fake samples
    #    alpha = torch.Tensor(np.random.uniform(0., 0.2, (real_samples.size(0), 1, 1, 1))).to(device)
    #    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True).to(device)
    #    for i in range(0): # sample more points
    #        
    #        # Get random interpolation between real and fake samples
    #        interpolates = torch.cat([interpolates, (alpha * real_samples + (1 - alpha) * fake_samples)], dim=0)
    #    
    #    alpha = torch.Tensor(np.random.uniform(0.8, 1., (real_samples.size(0), 1, 1, 1))).to(device)  
    #    interpolates = torch.cat([interpolates, (alpha * real_samples + (1 - alpha) * fake_samples)], dim=0)
        
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True).to(device)
        # for i in range(1):
        #     alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
        #     interpolates = torch.cat([interpolates, (alpha * real_samples + (1 - alpha) * fake_samples)], dim=0)
        
        # outputs
        # outputs = self.forward(interpolates)

        # gradient
        gradients = []
        d_interpolates = []
        fake = []
        gradient_penalty = None

        outputs = self.forward(interpolates)[1] # non-relativistic forward
        for i, D in enumerate(outputs):
            d_interpolates.append(D.mean(dim=(1,2,3))) # assume D returns (B,1,h,w)
            d_interpolates[i] = d_interpolates[i].unsqueeze(dim=1) # (B,1)
            fake.append(torch.ones_like(d_interpolates[i]).to(device))
            # Get gradient w.r.t. interpolates
            '''
            inputs = inputs.requires_grad_(True) is needed,

            compute gradients w.r.t inputs and integrate them using coefficients from grad_outputs (of the same shape of outputs), hence 'gradients.shape = inputs.shape';
            
            'create_graph = True' to compute autograd.grad(outputs=gradients,...) after 'gradients = autograd.grad(outputs=outputs,...)', noting 'retain_graph = True' is also needed.

            Moreover, if only_inputs is True, the function will only return a list of gradients w.r.t the specified inputs. If it’s False, then gradient w.r.t. all remaining leaves will still be computed, and will be accumulated into their .grad attribute.
            '''
            gradients.append(autograd.grad(
                outputs=d_interpolates[i],
                inputs=interpolates,
                grad_outputs=fake[i],
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0])

            gradients[i] = gradients[i].view(gradients[i].size(0), -1)
            
            gradient_penalty = ((gradients[i].norm(2, dim=1) - 1) ** 2).mean() if gradient_penalty is None else gradient_penalty + ((gradients[i].norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty/(i+1)

class ImageMap(nn.Module): # RRDBs
    def __init__(self, in_channels, out_channels, filters=64, n_layers=4, n_drb=3, n_rrdb=2, dilation=False, tanh=False):
        super().__init__()
        self.out_channels = out_channels if out_channels is not None else in_channels

        self.head = nn.Conv2d(in_channels, filters, 3, 1, 1)
        '''add activation layer'''
        # self.head = nn.Sequential(
        #     nn.Conv2d(in_channels, filters, 3, 1, 1),
        #     nn.InstanceNorm2d(filters),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     )

        self.main = nn.Sequential(
        *[ResidualInResidualDenseBlock(filters=filters, res_scale=0.2, n_layers=n_layers, n_drb=n_drb, dilation=dilation) for _ in range(n_rrdb)],
        nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.tail = nn.Conv2d(filters, self.out_channels, kernel_size=3, stride=1, padding=1)
        if tanh:
            self.tail.add_module('tanh', nn.Tanh())

    def forward(self, x):
        out = self.head(x)
        out = self.main(out) + out
        return self.tail(out)

class RA_MultiDiscriminator_Unet(nn.Module):
    '''relativistic average discriminator based on LSGAN '''
    def __init__(self, input_shape):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalize=True, p=None):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                # layers.append(nn.InstanceNorm2d(out_filters, 0.8))
                layers.append(nn.InstanceNorm2d(out_filters))
                # layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            if p is not None:
                layers.append(nn.Dropout2d(p=p))

            return layers

        class Feature_Up(nn.Module):
            def __init__(self, feature_channels):
                super().__init__()
                self.up2 = nn.Sequential(
                    nn.Upsample(scale_factor=(2,2), mode='nearest'),
                    nn.Conv2d(feature_channels, feature_channels, (3,3), (1,1), (1,1), bias=False),
                    nn.InstanceNorm2d(feature_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    )

                self.up1 = nn.Sequential(
                    nn.Conv2d(feature_channels*2, feature_channels, (3,3), (1,1), (1,1), bias=False),
                    nn.InstanceNorm2d(feature_channels),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Upsample(scale_factor=(2,2), mode='nearest'),
                    nn.Conv2d(feature_channels, feature_channels, (3,3), (1,1), (1,1), bias=False),
                    nn.InstanceNorm2d(feature_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    )

                self.pixel_tail = nn.Sequential(
                    nn.Conv2d(feature_channels*2, feature_channels, (3,3), (1,1), (1,1), bias=False),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(feature_channels, feature_channels, (3,3), (1,1), (1,1), bias=False),
                    nn.LeakyReLU(0.2, inplace=True), 

                    nn.Conv2d(feature_channels, 1, 3, padding=1),
                    )

            def forward(self, h0, h1, h2):
                u2 = self.up2(h2)
                u1 = self.up1(torch.cat([u2,h1], dim=1))
                output = self.pixel_tail(torch.cat([u1,h0], dim=1))
                return output  

        class ResCBAM(nn.Module):
            def __init__(self, in_channels, beta=1.):
                super().__init__()
                self.cbam = CBAM(in_channels)
                self.beta = beta
            def forward(self, x):
                # return x + self.beta*self.cbam(x)
                return x

        class Mixed_Dropout2d(nn.Sequential):
            def __init__(self, p1=0.2, p2=None):
                super().__init__()
                self.p2 = p1 if p2 is None else p2

                self.add_module('dropout2d', nn.Dropout2d(p=p1))
                self.add_module('spatial_dropout', Spatial_Dropout(p=self.p2))

        class Spatial_Dropout(nn.Module):
            def __init__(self, p=0.1):
                super().__init__()
                self.dropout = nn.Dropout2d(p=p)
            def forward(self, x):
                _, c, h, w = x.shape
                x = x.view(-1,c,h*w,1)
                x = x.permute((0,2,1,-1)) # (B, h*w, c, 1)
                x = self.dropout(x)
                x = x.transpose(1,2) # (B, c, h*w, 1)
                return x.view(-1,c,h,w) 

        channels, _, _ = input_shape
        feature_channels = 16

        # Extracts discriminator models
        self.extractor = ImageMap(in_channels=channels, out_channels=feature_channels, filters=16, n_layers=3, n_drb=2, n_rrdb=1, dilation=False, tanh=False)

        self.pixel_D = Feature_Up(feature_channels=feature_channels)

        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    # nn.Conv2d(channels, 32, 3, stride=1, padding=1), # 256
                    nn.Conv2d(feature_channels, 32, 4, stride=2, padding=1), # 256
                    # ResCBAM(32),
                    
                    *discriminator_block(32, 64, normalize=True), # 128
                    # ResCBAM(64, beta=0.2),
                    Mixed_Dropout2d(p1=0., p2=0.0),
                    # Spatial_Dropout(p=0.2),
                    # CAM_Module(64, beta=0.2),

                    *discriminator_block(64, 128), # 64 and dropout activations
                    CAM_Module(128),
                    Mixed_Dropout2d(p1=0.1, p2=0.1),

                    *discriminator_block(128, 256), # 32
                    CAM_Module(256),
                    Mixed_Dropout2d(p1=0.1, p2=0.1),

                    *discriminator_block(256, 512), # 16
                    # ResCBAM(512),
                    # CAM_Module(512),
                    nn.Conv2d(512, 1, 3, padding=1), # 16
                    # nn.Sigmoid(),
                )
            ) # downsample by 2^4

        class Downsample(nn.Module):
            def __init__(self, scale_factor, mode='bilinear'):
                super().__init__()
                self.scale_factor = scale_factor # scale_h_w or (scale_h, scale_w), munipulate 2D images
                self.mode = mode

            def forward(self, input):
                return nn.functional.interpolate(input, scale_factor=self.scale_factor, mode=self.mode)
                
        self.downsample = Downsample(scale_factor=0.5)
        # self.downsample = nn.AvgPool2d(channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, y, valid=1., fake=0., sg=False):
        """Computes the MSE between model output and scalar gt"""
        if not sg:
            relative_scores = self.forward(x,y,sg)
            loss = sum([torch.mean((out_1 - fake) ** 2) for out_1 in relative_scores[1]])
            return loss, relative_scores[0]
        else:
            loss = sum([torch.mean((out_0 - valid) ** 2 + (out_1 - fake) ** 2) for out_0, out_1 in zip(*self.forward(x,y,sg))])
            return loss

    def forward(self, x, y=None, sg=False):
        '''sg: False for updating D and True for updating G'''
        # device = x.device
        if y is None:
            outputs = []
            feature_outputs = []
            for m in self.models:
                x_ext = self.extractor(x)
                feature_outputs.append(x_ext)

                outputs.append(m(x_ext))
                x = self.downsample(x)

            pixel_score = self.pixel_D(*feature_outputs)    
            outputs.append(pixel_score)    
            return outputs
        else:
            outputs_0 = []
            outputs_1 = []
            feature_outputs_0 = []
            feature_outputs_1 = []

            if not sg: # for G loss
                for m in self.models:
                    x_ext = self.extractor(x).detach()
                    feature_outputs_0.append(x_ext)
                    x_score = m(x_ext).detach()

                    y_ext = self.extractor(y)
                    feature_outputs_1.append(y_ext)
                    y_score = m(y_ext)

                    # x_score_mean = torch.ones((1, *x_score.shape[1:])).to(device)
                    # x_score_mean.copy_(x_score.mean(dim=0, keepdim=True))

                    # outputs_1.append(y_score-x_score_mean.detach())

                    outputs_1.append(y_score-x_score.mean(dim=0, keepdim=True))

                    x = self.downsample(x)
                    y = self.downsample(y)

                x_pixel_score = self.pixel_D(*feature_outputs_0).detach()
                y_pixel_score = self.pixel_D(*feature_outputs_1)

                outputs_1.append(y_pixel_score-x_pixel_score.mean(dim=0, keepdim=True))

                return torch.clamp(outputs_1[-1], 0., 1.), outputs_1

            else: # for D loss
                for m in self.models:
                    x_ext = self.extractor(x)
                    feature_outputs_0.append(x_ext)
                    x_score = m(x_ext)

                    y_ext = self.extractor(y)
                    feature_outputs_1.append(y_ext)
                    y_score = m(y_ext)

                    outputs_0.append(x_score-y_score.mean(dim=0, keepdim=True))
                    outputs_1.append(y_score-x_score.mean(dim=0, keepdim=True))

                    x = self.downsample(x)
                    y = self.downsample(y)

                x_pixel_score = self.pixel_D(*feature_outputs_0)
                y_pixel_score = self.pixel_D(*feature_outputs_1)

                outputs_0.append(x_pixel_score-y_pixel_score.mean(dim=0, keepdim=True))
                outputs_1.append(y_pixel_score-x_pixel_score.mean(dim=0, keepdim=True)) 

                return outputs_0, outputs_1

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        device= real_samples.device
        
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True).to(device)

        # gradient
        gradients = []
        d_interpolates = []
        fake = []
        gradient_penalty = None

        outputs = self.forward(interpolates) # non-relativistic forward
        for i, D in enumerate(outputs):
            d_interpolates.append(D.mean(dim=(1,2,3))) # assume D returns (B,1,h,w)
            d_interpolates[i] = d_interpolates[i].unsqueeze(dim=1) # (B,1)
            fake.append(torch.ones_like(d_interpolates[i]).to(device))
            # Get gradient w.r.t. interpolates
            '''
            inputs = inputs.requires_grad_(True) is needed,

            compute gradients w.r.t inputs and integrate them using coefficients from grad_outputs (of the same shape of outputs), hence 'gradients.shape = inputs.shape';
            
            'create_graph = True' to compute autograd.grad(outputs=gradients,...) after 'gradients = autograd.grad(outputs=outputs,...)', noting 'retain_graph = True' is also needed.

            Moreover, if only_inputs is True, the function will only return a list of gradients w.r.t the specified inputs. If it’s False, then gradient w.r.t. all remaining leaves will still be computed, and will be accumulated into their .grad attribute.
            '''
            gradients.append(autograd.grad(
                outputs=d_interpolates[i],
                inputs=interpolates,
                grad_outputs=fake[i],
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0])

            gradients[i] = gradients[i].view(gradients[i].size(0), -1)
            
            gradient_penalty = ((gradients[i].norm(2, dim=1) - 1) ** 2).mean() if gradient_penalty is None else gradient_penalty + ((gradients[i].norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty/(i+1)       