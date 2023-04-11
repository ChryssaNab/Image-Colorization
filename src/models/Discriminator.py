import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchDiscriminator(nn.Module):
    """ Defines the Patch Discriminator network for GAN.

    Arguments
    ----------
    in_channels : <class 'int'>
        the number of input channels
    out_channels : <class 'int'>
        the number of output channels
    kernel_size : <class 'int'>
        the convolution kernel size
    stride : <class 'int'>
        the stride to be used for the convolution
    padding : <class 'int'>
        the padding to be used for the convolution
    """

    def __init__(self, in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1):
        super(PatchDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2_bn = nn.BatchNorm2d(out_channels * 2)
        self.conv3 = nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv3_bn = nn.BatchNorm2d(out_channels * 4)
        self.conv4 = nn.Conv2d(out_channels * 4, out_channels * 8, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.conv4_bn = nn.BatchNorm2d(out_channels * 8)
        self.conv5 = nn.Conv2d(out_channels * 8, 1, kernel_size=kernel_size, stride=1, padding=padding)

    # Initialize weights from a Gaussian distribution with mean 0 and std 0.02
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, image):
        x = F.leaky_relu(self.conv1(image), 0.2)  # [64, 128, 128]
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)  # [128, 64, 64]
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)  # [256, 32, 32]
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)  # [512, 31, 31]
        x = torch.sigmoid(self.conv5(x))  # [1, 30, 30]

        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()

