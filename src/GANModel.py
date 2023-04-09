import torch.nn as nn
import torch
from torch import optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_requires_grad(model, requires_grad=True):
    for p in model.parameters():
        p.requires_grad = requires_grad

"""
A class for combining the generator and discriminator models and train them alternately.

    Attributes
    ----------
    gen : <class nn.Module>
        The GAN's generator
    disc: <class nn.Module>
        The GAN's discriminator
    opt_G: <class optim.Adam>
        The optimizer for the Generator
    opt_D: <class optim.Adam>
        The optimizer for the Discriminator
    Methods
    -------
    optimize():
        Given the input and target data, performs one round of training on the generator and discriminator models
"""
class ColorizationGAN(nn.Module):

    """
    Class initialization method. Initialize generator, discriminator and optimizers.

            Parameters
            ----------
            lr_g : <class 'float'>
                learning rate for the generator parameters
            lr_d : <class 'float'>
                learning rate for the discriminator parameters
            beta1 : <class 'float'>
                first beta parameter for the Adam optimizer
            beta2 : <class 'float'>
                second beta parameter for the Adam optimizer
            lambda_l1: <class 'float'>
                the factor by which the L1 loss is scaled before adding it to the GAN's loss
    """
    def __init__(self, lr_g = 0.0002, lr_d = 0.0002, beta1=0.5, beta2=0.999, lambda_l1=100):
        super().__init__()

        self.lambda_l1=lambda_l1

        #This will be replaced with the generator model
        self.gen = nn.Module
        self.gen = self.gen.to(device)
        #This will be replaced with the discriminator model
        self.disc = nn.Module
        self.disc = self.disc.to(device)

        #This will be replaced with the implementation of the gan loss
        self.GANloss = nn.Module
        self.L1Loss = nn.Module

        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_g, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_d, betas=(beta1, beta2))

    """
        Performs one round of training on a batch of images in the Lab colorspace

                Parameters
                ----------
                L : <class 'Tensor'>
                    The L channel of the image
                ab : <class 'Tensor'>
                    The real ab channels of the image
                
                Returns:
                --------
                loss_D: The loss for the discriminator (fake + real)
                loss_G: The loss for the generator (GANloss + L1)
                

    """
    def optimize(self, L, ab):

        L = L.to(device)
        ab = ab.to(device)

        # Get color channels from generator
        fake_ab = self.net_G(L)
        self.disc.train()
        set_requires_grad(self.disc, True)
        self.disc.zero_grad()

        # Compose fake images and pass them to the discriminator
        fake_image = torch.cat([L, fake_ab], dim=1)
        fake_preds = self.disc(fake_image.detach())
        loss_D_fake = self.GANloss(fake_preds, False)

        # Pass the real images to the discriminator
        real_image = torch.cat([L, ab], dim=1)
        real_preds = self.disc(real_image)
        loss_D_real = self.GANloss(real_preds, True)

        # Combine losses and calculate the gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()

        # Update the discriminator parameters
        self.opt_D.step()

        self.gen.train()
        self.set_requires_grad(self.disc, False)
        self.opt_G.zero_grad()

        # Combine the "reward signal" from the discriminator with L1 loss
        fake_preds = self.net_D(fake_image)
        loss_G_GAN = self.GANloss(fake_preds, True)
        loss_G_L1 = self.L1Loss(fake_ab, ab) * self.lambda_L1
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()

        # Update the parameters of the generator
        self.opt_G.step()

        return loss_D.item(), loss_G.item()