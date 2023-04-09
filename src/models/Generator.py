import torch.nn as nn

from ResNet_Generator import ResNetEncoder
from UNet_Generator import UNetEncoder, UNetDecoder


class Generator(nn.Module):
    """ Defines the Generator network for GAN.

    Arguments
    ----------
    pretrained : <class 'bool'> (default=False)
        boolean to control whether to generate a pretrained ResNet encoder or a U-Net encoder from scratch

    Attributes
    ----------
    encoder_net : PyTorch model object of encoder
        the encoder network of the Generator
    decoder_net : PyTorch model object of decoder
        the decoder network of the Generator
    """

    def __init__(self, pretrained=False):
        super().__init__()

        if pretrained:
            self.encoder_net = ResNetEncoder()
        else:
            self.encoder_net = UNetEncoder()

        self.decoder_net = UNetDecoder(self.encoder_net)

    def forward(self, image):
        encoded_features = self.encoder_net(image)
        decoded_features = self.decoder_net(encoded_features)
        return decoded_features


print(Generator())