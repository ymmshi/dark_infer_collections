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
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from models import basicblock as B


def sum(x, device):
    pi = torch.tensor(math.pi)
    w = x.size()[-2]
    h = x.size()[-1]
    eh = 6.0 * (w - 2.0) * (h - 2.0)
    r = noise_esti(x[:, 0, :, :].unsqueeze(0), device)
    g = noise_esti(x[:, 1, :, :].unsqueeze(0), device)
    b = noise_esti(x[:, 2, :, :].unsqueeze(0), device)
    sr = torch.sum(torch.abs(r), (2, 3))[0]
    sg = torch.sum(torch.abs(g), (2, 3))[0]
    sb = torch.sum(torch.abs(b), (2, 3))[0]
    sumr = 2 * (torch.sqrt(pi / 2.0) * (1.0 / eh)) * (sr)
    sumg = 2 * (torch.sqrt(pi / 2.0) * (1.0 / eh)) * (sg)
    sumb = 2 * (torch.sqrt(pi / 2.0) * (1.0 / eh)) * (sb)
    return sumr, sumg, sumb


def noise_esti(x, device):
    a = [1, -2, 1, -2, -4, -2, 1, -2, 1]
    kernel = torch.tensor(a).reshape(1, 1, 3, 3).float().to(device)
    b = F.conv2d(input=x, weight=kernel, stride=3, padding=1)
    return b


class IRCNN(nn.Module):
    def __init__(self, in_nc=3, out_nc=24, nc=32):
        super(IRCNN, self).__init__()
        self.model = B.IRCNN(in_nc, out_nc, nc)
        self.device = torch.device('cuda')

    def forward(self, x):
        row, col = x.size()[-2], x.size()[-1]
        lam, lam1, lam2 = sum(x, self.device)
        am = torch.zeros(1, row, col).to(self.device) + lam
        am1 = torch.zeros(1, row, col).to(self.device) + lam1
        am2 = torch.zeros(1, row, col).to(self.device) + lam2
        n = self.model(x)
        l1 = torch.where((n[0:, 0:8:, :] + lam) > 0, (n[0:, 0:8:, :] + lam), am)
        l2 = torch.where((n[0:, 8:16:, :] + lam1) > 0, (n[0:, 8:16:, :] + lam1), am1)
        l3 = torch.where((n[0:, 16:24:, :] + lam2) > 0, (n[0:, 16:24:, :] + lam2), am2)
        level = torch.cat((l1, l2, l3), 1)
        return l1, l2, l3, level
