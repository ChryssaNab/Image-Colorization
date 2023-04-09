import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class ResNetEncoder(nn.Module):
    """ Defines the ResNet-18 encoder for the Generator. """

    def __init__(self):
        super().__init__()

        self.resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    def forward(self, x):
        self.block1 = self.resnet18.conv1(x)
        self.block1 = self.resnet18.bn1(self.block1)
        self.block1 = self.resnet18.relu(self.block1)  # [64, H/2, W/2]

        self.block2 = self.resnet18.maxpool(self.block1)
        self.block2 = self.resnet18.layer1(self.block2)  # [64, H/4, W/4]
        self.block3 = self.resnet18.layer2(self.block2)  # [128, H/8, W/8]
        self.block4 = self.resnet18.layer3(self.block3)  # [256, H/16, W/16]
        self.block5 = self.resnet18.layer4(self.block4)  # [512, H/32, W/32]

        return self.block5
