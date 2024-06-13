# from https://github.com/ssktotoro/neuro/blob/master/training/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

"""
Synthseg spec:

n_levels = 5           # number of resolution levels
nb_conv_per_level = 2  # number of convolution per level
conv_size = 3          # size of the convolution kernel (e.g. 3x3x3)
unet_feat_count = 24   # number of feature maps after the first convolution
activation = 'elu'     # activation for all convolution layers except the last, which will use softmax regardless
feat_multiplier = 2    # if feat_multiplier is set to 1, we will keep the number of feature maps constant throughout the

"""

class DoubleConv(nn.Module):
    """(convolution => [BN] => ELU) * 2""" # changed to match synthseg spec

    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        Docs.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ELU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        """Docs."""
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        """
        Docs.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """Docs."""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self, in_channels, out_channels, mid_channels=None, bilinear=True
    ):
        """
        Docs.
        """
        super().__init__()

        # if bilinear,
        # use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="trilinear", align_corners=True
            )
            if mid_channels:
                self.conv = DoubleConv(
                    in_channels, out_channels, mid_channels=mid_channels,
                )
            else:
                self.conv = DoubleConv(
                    in_channels, out_channels // 2, in_channels // 2
                )
        else:
            self.up = nn.ConvTranspose3d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Docs.
        """
        x1 = self.up(x1)
        # input is CHW
        diffZ = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffY = torch.tensor([x2.size()[3] - x1.size()[3]])
        diffX = torch.tensor([x2.size()[4] - x1.size()[3]])

        x1 = F.pad(
            x1,
            [
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2,
                diffZ // 2,
                diffY - diffZ // 2,
            ],
        )
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    Docs.
    """

    def __init__(self, in_channels, out_channels):
        """
        Docs.
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Docs.
        """
        return self.conv(x)


class UNet(nn.Module):
    """Docs."""

    def __init__(self, n_channels, n_classes, bilinear=True):
        """Docs."""
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear


        dim0 = feat_cont
        dim1 = dim0 * 2
        dim2 = dim1 * 2
        dim3 = dim2 * 2
        dim4 = dim3 * 2
        dim5 = dim4 * 2
        dim6 = dim5 * 2

        self.inc =   DoubleConv(n_channels, feat_cont)
        self.down1 = Down(dim0, dim1)
        self.down2 = Down(dim1, dim2)
        self.down3 = Down(dim2, dim3)
        self.down4 = Down(dim3, dim4)
        self.down5 = Down(dim4, dim5)

        self.up1   = Up(dim5, dim4, bilinear=bilinear, mid_channels=256)
        self.up2   = Up(dim4, dim3, bilinear=bilinear, mid_channels=256)
        self.up3   = Up(dim3, dim2, bilinear=bilinear, mid_channels=256)
        self.up4   = Up(dim2, dim1, bilinear=bilinear, mid_channels=256)
        self.up5   = Up(dim1, dim0, bilinear=bilinear, mid_channels=256)
        self.outc =  OutConv(dim0, n_classes)

    def forward(self, x):
        """Docs."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits = self.outc(x)
        return logits

def compound_l1_loss(y_hat, y):
    # compound l1 loss for segmentation based on https://github.com/by-liu/SegLossBias
    temp = 10
    logits_softmax = F.log_softmax(y_hat * 10, dim=1).exp()
    cardinality = y.shape[1] * y.shape[2] * y.shape[3]
    EPS = 0.00000000001

    #gt proportion
    gt_region_proportion = (torch.einsum("bcwhd->bc", one_hot_targets) +
                            EPS) / (cardinality + EPS)
    pred_region_proportion = (torch.einsum("bcwhd->bc", logits_softmax) +
                              EPS) / (cardinality + EPS)
    loss_reg = (pred_region_proportion - gt_region_proportion).abs().mean()
    alpha = 1
    loss = ce_loss + alpha * loss_reg
    return loss
