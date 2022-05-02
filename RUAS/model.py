import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
import genotypes


class SearchBlock(nn.Module):
    def __init__(self, channel, genotype):
        super(SearchBlock, self).__init__()
        self.channel = channel
        op_names, _ = zip(*genotype.normal)
        self.c1_d = OPS[op_names[0]](self.channel, self.channel)
        self.c1_r = OPS[op_names[1]](self.channel, self.channel)
        self.c2_d = OPS[op_names[2]](self.channel, self.channel)
        self.c2_r = OPS[op_names[3]](self.channel, self.channel)
        self.c3_d = OPS[op_names[4]](self.channel, self.channel)
        self.c3_r = OPS[op_names[5]](self.channel, self.channel)
        self.c4 = OPS[op_names[6]](self.channel, self.channel)
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=False)
        self.c5 = nn.Conv2d(self.channel * 4, self.channel, 1)

    def forward(self, x):
        distilled_c1 = self.act(self.c1_d(x))
        r_c1 = self.act(self.c1_r(x) + x)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.act(self.c2_r(r_c1) + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.act(self.c3_r(r_c2) + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.c5(out)

        return out_fused


class IEM(nn.Module):
    def __init__(self, channel, genetype):
        super(IEM, self).__init__()
        self.channel = channel
        self.genetype = genetype
        self.cell = SearchBlock(self.channel, self.genetype)

    def forward(self, x, u):
        u = F.pad(u, (0, 1, 0, 1), "constant", 0)
        u = F.max_pool2d(u, 2, 1, 0)
        t = 0.5 * (u + x)
        t = self.cell(t)
        t = torch.sigmoid(t)
        t = torch.clamp(t, 0.001, 1.0)
        u = torch.clamp(x / t, 0.0, 1.0)
        return u


class EnhanceNetwork(nn.Module):
    def __init__(self, iteratioin, channel, genotype):
        super(EnhanceNetwork, self).__init__()
        self.iem_nums = iteratioin

        self.iems = nn.ModuleList()
        for i in range(self.iem_nums):
            self.iems.append(IEM(channel, genotype))

    def forward(self, x):
        o = x
        for i in range(self.iem_nums):
            o = self.iems[i](x, o)
        return o


class DenoiseNetwork(nn.Module):
    def __init__(self, layers, channel, genotype):
        super(DenoiseNetwork, self).__init__()
        self.layers = layers
        self.stem = nn.Conv2d(3, channel, 3, 1, 1)
        self.nrms = nn.ModuleList()
        for _ in range(layers):
            self.nrms.append(SearchBlock(channel, genotype))
        self.activate = nn.Sequential(nn.Conv2d(channel, 3, 3, 1, 1))

    def forward(self, x):
        feat = self.stem(x)
        for i in range(self.layers):
            feat = self.nrms[i](feat)
        noise = self.activate(feat)
        output = x - noise
        return output


class Model(nn.Module):
    def __init__(self, with_denoise=True):
        super(Model, self).__init__()
        self.with_denoise = with_denoise
        self.enhance_net = EnhanceNetwork(iteratioin=3, channel=3, genotype=genotypes.IEM)
        self.denoise_net = DenoiseNetwork(layers=3, channel=6, genotype=genotypes.NRM)

    def forward(self, x):
        x = self.enhance_net(x)
        if self.with_denoise:
            x = self.denoise_net(x)
        return x

    def load_weight(self, weight_path):
        self.load_state_dict(torch.load(weight_path))


if __name__ == '__main__':
    model = Model()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
