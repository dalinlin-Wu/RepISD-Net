import torch
from torch import nn
import torch.nn.functional as F

from model.Unet import UNet

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []

        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):

        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        out = torch.cat(out, 1)  ###
        return out


class RepISD(nn.Module):
    def __init__(self,pdcs, bins=(1, 2, 3, 6), dropout=0.1, classes=2,convert=False,deploy=False):
        super(RepISD, self).__init__()

        assert classes > 1
        self.deploy = deploy
        self.convert = convert
        self.backbone = UNet(pdcs,n_channels=3, n_classes=classes,convert=self.convert, deploy=self.deploy)

        fea_dim = 32

        self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
        fea_dim *= 2

        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(32, classes, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):

        feats = self.backbone(x)
        ppm_fea = self.ppm(feats)
        x = self.cls(ppm_fea)

        return x


