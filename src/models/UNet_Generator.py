import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetEncoder(nn.Module):
    """ Defines the U-Net-style encoder for the Generator. """

    def __init__(self, input_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1):
        super().__init__()

        self.conv_layer = nn.Conv2d(input_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.conv_block1 = self.conv_block(out_channels, out_channels * 2, kernel_size, stride, padding)
        self.conv_block2 = self.conv_block(out_channels * 2, out_channels * 4, kernel_size, stride, padding)
        self.conv_block3 = self.conv_block(out_channels * 4, out_channels * 8, kernel_size, stride, padding)
        self.conv_block4 = self.conv_block(out_channels * 8, out_channels * 8, kernel_size, stride, padding)
        self.conv_block5 = self.conv_block(out_channels * 8, out_channels * 8, kernel_size, stride, padding)
        self.conv_block6 = self.conv_block(out_channels * 8, out_channels * 8, kernel_size, stride, padding)

        self.conv_block7 = nn.Conv2d(out_channels * 8, out_channels * 8, kernel_size, stride, padding)

    def forward(self, x):
        self.block1 = self.conv_layer(x)  # [64, H/2, W/2]
        self.block2 = self.conv_block1(self.block1)  # [128, H/4, W/4]
        self.block3 = self.conv_block2(self.block2)  # [256, H/8, W/8]
        self.block4 = self.conv_block3(self.block3)  # [512, H/16, W/16]
        self.block5 = self.conv_block4(self.block4)  # [512, H/32, W/32]
        self.block6 = self.conv_block5(self.block5)  # [512, H/64, W/64]
        self.block7 = self.conv_block6(self.block6)  # [512, H/128, W/128]
        self.block8 = self.conv_block7(F.leaky_relu(self.block7, 0.2))  # [512, H/256, W/256]

        return self.block8

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size, stride, padding):
        """ Builds a convolutional block.

        Arguments
        ---------
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

        Returns
        -------
        A sequential block depending on the input arguments
        """

        block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        return block

    # Initialize encoder weights from a Gaussian distribution with mean 0 and std 0.02
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class UNetDecoder(nn.Module):
    """ Defines the decoder network for the Generator.

    Arguments
    ----------
    encoder_net : PyTorch model object of encoder
        the encoder network of the Generator
    """

    def __init__(self, encoder_net, out_channels=512, kernel_size=4, stride=2, padding=1, use_dropout=False):
        super().__init__()

        self.encoder_net = encoder_net

        self.up_block1 = self.transp_conv_block(512, out_channels, kernel_size, stride, padding, use_dropout=True)
        self.up_block2 = self.transp_conv_block(out_channels * 2, out_channels, kernel_size, stride, padding, use_dropout=True)
        self.up_block3 = self.transp_conv_block(out_channels * 2, out_channels, kernel_size, stride, padding, use_dropout=True)
        self.up_block4 = self.transp_conv_block(out_channels * 2, out_channels, kernel_size, stride, padding, use_dropout)
        self.up_block5 = self.transp_conv_block(out_channels * 2, out_channels // 2, kernel_size, stride, padding, use_dropout)
        self.up_block6 = self.transp_conv_block(out_channels, out_channels // 4, kernel_size, stride, padding, use_dropout)
        self.up_block7 = self.transp_conv_block(out_channels // 2, out_channels // 8, kernel_size, stride, padding, use_dropout)

        self.up_block8 = nn.ConvTranspose2d(out_channels // 4, 2, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        up_1 = self.up_block1(x)  # [512, H/128, W/128]
        up_1 = torch.cat([self.encoder_net.block7, up_1], dim=1)  # [1024, H/128, W/128]

        up_2 = self.up_block2(up_1)  # [512, H/64, W/64]
        up_2 = torch.cat([self.encoder_net.block6, up_2], dim=1)   # [1024, H/64, W/64]

        up_3 = self.up_block3(up_2)  # [512, H/32, W/32]
        up_3 = torch.cat([self.encoder_net.block5, up_3], dim=1)  # [1024, H/32, W/32]

        up_4 = self.up_block4(up_3)  # [512, H/16, W/16]
        up_4 = torch.cat([self.encoder_net.block4, up_4], dim=1)  # [1024, H/16, W/16]

        up_5 = self.up_block5(up_4)  # [256, H/8, W/8]
        up_5 = torch.cat([self.encoder_net.block3, up_5], dim=1)  # [512, H/8, W/8]

        up_6 = self.up_block6(up_5)  # [128, H/4, W/4]
        up_6 = torch.cat([self.encoder_net.block2, up_6], dim=1)  # [256, H/4, W/4]

        up_7 = self.up_block7(up_6)  # [64, H/2, W/2]
        up_7 = torch.cat([self.encoder_net.block1, up_7], dim=1)  # [128, H/2, W/2]

        up_8 = self.up_block8(F.relu(up_7))  # [2, H, W]
        output_image = F.tanh(up_8)

        return output_image

    @staticmethod
    def transp_conv_block(in_channels, out_channels, kernel_size, stride, padding, use_dropout):
        """ Builds a transposed convolutional block.

        Arguments
        ---------
        in_channels : <class 'int'>
            the number of input channels
        out_channels : <class 'int'>
            the number of output channels
        kernel_size : <class 'int'>
            the convolution kernel size
        stride : <class 'int'>
            the stride to be used for the transposed convolution
        padding : <class 'int'>
            the padding to be used for the transposed convolution
        use_dropout : bool (default=False)
            boolean to control whether to use dropout or not

        Returns
        -------
        A sequential block depending on the input arguments
        """

        if use_dropout:
            block = nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.5),
            )
        else:
            block = nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        return block

    # Initialize decoder weights from a Gaussian distribution with mean 0 and std 0.02
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
