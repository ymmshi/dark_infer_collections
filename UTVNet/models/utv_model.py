# -*- coding: utf-8 -*-
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

import torch.fft
import torch
import torch.nn as nn
from models import basicblock as B


class ADMM(nn.Module):
    def __init__(self, inc, k, cha):
        super(ADMM, self).__init__()
        self.device = torch.device('cuda')
        self.hyp = B.HyPaNet(inc, k, cha)
        self.ouc = k
        self.mlp = nn.Sequential()

    def fftn(self, t, row, col, dim):
        y = torch.fft.fftn(t, col, dim=dim)
        y = y.expand(col, row)
        return y

    def fftnt(self, t, row, col, dim):
        y = torch.fft.fftn(t, col, dim=dim)
        y = y.expand(row, col)
        return y

    def ForwardDiff(self, x):
        x_diff = x[:, 1:] - x[:, :-1]
        x_e = (x[:, 0] - x[:, -1]).unsqueeze(1)
        x_diff = torch.cat((x_diff, x_e), 1)
        y_diff = x[1:, :] - x[:-1, :]
        y_e = (x[0, :] - x[-1, :]).unsqueeze(0)
        y_diff = torch.cat((y_diff, y_e), 0)
        return x_diff, y_diff

    def Dive(self, x, y):
        x_diff = x[:, :-1] - x[:, 1:]
        x_e = (x[:, -1] - x[:, 0]).unsqueeze(1)
        x_diff = torch.cat((x_e, x_diff), 1)
        y_diff = y[:-1, :] - y[1:, :]
        y_e = (y[-1, :] - y[0, :]).unsqueeze(0)
        y_diff = torch.cat((y_e, y_diff), 0)
        return y_diff + x_diff

    def shrink(self, x, r, m):
        z = torch.sign(x) * torch.max(torch.abs(x) - r, m)
        return z

    def forward(self, yo, lam):
        s = torch.tensor(2.0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device)
        rho = self.hyp(s)
        row, col = yo.size()[1], yo.size()[2]
        y = yo[0, :, :].squeeze(0)
        x = y
        v1 = torch.zeros(row, col).to(self.device)
        v2 = torch.zeros(row, col).to(self.device)
        m = torch.zeros(row, col).to(self.device)
        y1 = torch.zeros(row, col).to(self.device)
        y2 = torch.zeros(row, col).to(self.device)
        x1 = ([1.0], [-1.0])
        x2 = ([1.0, -1.0])
        Dx = torch.tensor(x1).to(self.device)
        x3 = torch.tensor(x2).to(self.device)
        eigDtD = torch.pow(torch.abs(self.fftn(Dx, col, row, 0)), 2) + torch.pow(torch.abs(self.fftnt(x3, row, col, 0)),
                                                                                 2).to(self.device)

        for k in range(0, self.ouc):
            rhs = y - rho[0, k, 0, 0] * self.Dive(torch.div(y1, rho[0, k, 0, 0]) + v1,
                                                  torch.div(y2, rho[0, k, 0, 0]) + v2)
            lhs = 1 + rho[0, k, 0, 0] * (eigDtD)
            x = torch.div(torch.fft.fftn(rhs), lhs)
            x = torch.real(torch.fft.ifftn(x))
            Dx1, Dx2 = self.ForwardDiff(x)
            u1 = Dx1 + torch.div(y1, rho[0, k, 0, 0])
            u2 = Dx2 + torch.div(y2, rho[0, k, 0, 0])
            v1 = self.shrink(u1, torch.div(lam[k, :, :], rho[0, k, 0, 0]), m)
            v2 = self.shrink(u2, torch.div(lam[k, :, :], rho[0, k, 0, 0]), m)
            y1 = y1 - rho[0, k, 0, 0] * (v1 - Dx1)
            y2 = y2 - rho[0, k, 0, 0] * (v2 - Dx2)
        return x
