'''
This is a PyTorch implementation of the ICCV 2021 paper:
"Adaptive Unfolding Total Variation Network for Low-Light Image Enhancement": https://arxiv.org/abs/2110.00984

Please cite the paper if you use this code

@InProceedings{Zheng_2021_ICCV,
    author    = {Zheng, Chuanjun and Shi, Daming and Shi, Wentian},
    title     = {Adaptive Unfolding Total Variation Network for Low-Light Image Enhancement},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {4439-4448}
}

Tested with Pytorch 1.7.1, Python 3.6

Author: Chuanjun Zheng (chuanjunzhengcs@gmail.com)

'''
import torch.nn as nn
import torch
import torch.nn.functional as F


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = x2 + x1
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        # self.than=nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        # x=self.than(x)
        return x


class globalFeature(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, inSize, outSize):
        super(globalFeature, self).__init__()
        self.global_feature = nn.Sequential(
            nn.Linear(inSize, outSize),
            nn.LeakyReLU(0.2, inplace=True),

        )
        self.global_feature_1 = nn.Sequential(
            nn.Linear(outSize, outSize),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, y2, x):
        y = torch.mean(x, dim=(2, 3))
        y1 = self.global_feature(y)

        y = self.global_feature_1(y1)
        y1 = torch.unsqueeze(y1, dim=2)
        y1 = torch.unsqueeze(y1, dim=3)

        y = torch.unsqueeze(y, dim=2)
        y = torch.unsqueeze(y, dim=3)
        # z=torch.zeros(size.size()).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        glob = y2 * y1 + y
        # glob=torch.cat((glob,size),dim=1)
        return glob


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.inc = nn.Sequential(
            single_conv(27, 32),
            single_conv(32, 32),
            single_conv(32, 32)
        )

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(32, 64),
            single_conv(64, 64),
        )
        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
        )

        self.down3 = nn.AvgPool2d(2)
        self.conv3 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
        )

        self.down4 = nn.AvgPool2d(2)
        self.conv4 = nn.Sequential(
            single_conv(256, 256),
            single_conv(256, 256),
        )

        self.glo = globalFeature(256, 256)
        self.convglo = nn.Sequential(
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),

        )
        self.convglo1 = nn.Sequential(
            single_conv(256, 256),
            single_conv(256, 256),
        )
        self.glo1 = globalFeature(256, 256)

        self.up1 = up(256, 256)
        self.convup1 = nn.Sequential(
            single_conv(256, 256),
            single_conv(256, 256),
        )

        self.up2 = up(256, 128)
        self.convup2 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
        )

        self.up3 = up(128, 64)
        self.convup3 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )
        self.up4 = up(64, 32)
        self.convup4 = nn.Sequential(
            single_conv(32, 32),
            single_conv(32, 32)
        )

        self.outc = outconv(32, 3)

    def forward(self, x, level):
        img = torch.cat((level, x), 1)
        inx = self.inc(img)

        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        down3 = self.down3(conv2)
        conv3 = self.conv3(down3)

        down4 = self.down4(conv3)
        conv4 = self.conv4(down4)

        glo = self.glo(down4, conv4)
        convglo = self.convglo(glo)

        convglo1 = self.convglo1(convglo)
        glo1 = self.glo1(convglo, convglo1)

        up1 = self.up1(glo1, conv3)
        convup1 = self.convup1(up1)

        up2 = self.up2(convup1, conv2)
        convup2 = self.convup2(up2)

        up3 = self.up3(convup2, conv1)
        convup3 = self.convup3(up3)

        up4 = self.up4(convup3, inx)
        convup4 = self.convup4(up4)

        out = self.outc(convup4)
        return out
