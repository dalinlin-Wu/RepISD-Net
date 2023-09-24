""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from model.ECB import ECB_block

class DoubleConv_ori(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.double_conv(x)
        return out###

###depoly=False时，用这个模块
class DoubleConv_down(nn.Module):  ###stride要注意设置，因为涉及到了maxpooling
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, pdcs,in_channels, out_channels, mid_channels=None, stride=1, convert=False,deploy=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            ECB_block(pdcs,in_channels, mid_channels,kernel_size=3, padding=1,stride=stride,convert=convert,deploy=deploy),
            ECB_block(pdcs,mid_channels, out_channels, kernel_size=3, padding=1,stride=1,convert=convert,deploy=deploy),

        )


    def forward(self, x):
        return self.double_conv(x)



class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self,pdcs, in_channels, out_channels, mid_channels,stride,convert, deploy):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_down(pdcs, in_channels, out_channels, mid_channels=mid_channels, convert=convert,deploy=deploy)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class DoubleConv_up(nn.Module):  ###stride要注意设置，因为涉及到了maxpooling
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, pdcs,in_channels, out_channels, mid_channels=None, convert=False,deploy=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            ECB_block(pdcs,in_channels, mid_channels,kernel_size=3, padding=1,stride=1,convert=convert,deploy=deploy),
            ECB_block(pdcs,mid_channels, out_channels,kernel_size=3, padding=1,stride=1,convert=convert,deploy=deploy),
        )


    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self,pdcs,in_channels, out_channels,convert, deploy):
        super().__init__()

        #  use the normal convolutions to reduce the number of channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv_up(pdcs,in_channels, out_channels, in_channels // 2,convert, deploy)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self,pdcs, n_channels, n_classes,convert=False,deploy=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.deploy = deploy
        self.convert = convert

        self.inc = DoubleConv_ori(n_channels, 32)

        self.down1 = Down(pdcs,32, 64,None, stride=2, convert=self.convert,deploy =self. deploy)
        factor = 2
        self.down2 = Down(pdcs,64, 128// factor,None,2, convert=self.convert,deploy =self. deploy)

        self.up3 = Up(pdcs,128, 64 // factor, convert=self.convert,deploy =self. deploy)
        self.up4 = Up(pdcs,64, 32, convert=self.convert,deploy =self. deploy)

    def forward(self, x):


        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up3(x3, x2)
        x = self.up4(x, x1)


        return x




