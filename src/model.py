import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """ Defines the Generator network for GAN.

    Arguments
    ----------
    training_mode : <class 'bool'>
        boolean to control whether to generate a pre-trained ResNet encoder or a U-Net encoder from scratch

    Attributes
    ----------
    encoder_net : PyTorch model object of encoder
        the encoder network of the generator
    decoder_net : PyTorch model object of decoder
        the decoder network of the generator
    """

    def __init__(self, training_mode=True):
        super().__init__()
        self.encoder_features = None
        self.decoder_features = None

        if training_mode:
            self.encoder_net = ResNetEncoder()
        else:
            self.encoder_net = UNetEncoder()

        self.decoder_net = UNetDecoder(self.encoder_net)

    def forward(self, x):
        self.encoder_features = self.encoder_net(x)
        self.decoder_features = self.decoder_net(self.encoder_features)
        return self.decoder_features


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
        the stride to be used for convolution
    padding : <class 'int'>
        the padding to be used for convolution
    """

    def __init__(self, in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1):
        super(PatchDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2_bn = nn.BatchNorm2d(out_channels * 2)
        self.conv3 = nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3_bn = nn.BatchNorm2d(out_channels * 4)
        self.conv4 = nn.Conv2d(out_channels * 4, out_channels * 8, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv4_bn = nn.BatchNorm2d(out_channels * 8)
        self.conv5 = nn.Conv2d(out_channels * 8, 1, kernel_size=kernel_size, stride=1, padding=padding)

    # Initialize weights from a Gaussian distribution with mean 0 and std 0.02
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # Forward method
    def forward(self, input_img, output_img):
        x = torch.cat([input_img, output_img], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


discriminator = PatchDiscriminator()
print(discriminator)
discriminator.weight_init(mean=0.0, std=0.02)