import torch
import torch.nn as nn
import torch.nn.functional as F


class RDB_Conv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(RDB_Conv, self).__init__()
        self.conv = nn.Sequential(*[
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, init_dim, grow_dim, nConvLayers):
        super(RDB, self).__init__()
        convs = []
        for n in range(nConvLayers):
            convs.append(RDB_Conv(init_dim + n * grow_dim, grow_dim))

        self.convs = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(init_dim + nConvLayers * grow_dim, init_dim, 1, 1, 0)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class DRBN_BU(nn.Module):
    def __init__(self, init_dim, grow_dim, nConvLayers):
        super(DRBN_BU, self).__init__()
        self.SFENet1 = nn.Conv2d(3*2, init_dim, 3, 1, 1)
        self.SFENet2 = nn.Conv2d(init_dim, init_dim, 3, 1, 1)

        self.RDBs = nn.ModuleList()
        self.RDBs.append(RDB(init_dim,   grow_dim,   nConvLayers))
        self.RDBs.append(RDB(init_dim,   grow_dim,   nConvLayers))
        self.RDBs.append(RDB(2*init_dim, 2*grow_dim, nConvLayers))
        self.RDBs.append(RDB(2*init_dim, 2*grow_dim, nConvLayers))
        self.RDBs.append(RDB(init_dim,   grow_dim,   nConvLayers))
        self.RDBs.append(RDB(init_dim,   grow_dim,   nConvLayers))

        self.UPNet = nn.Sequential(*[
                nn.Conv2d(init_dim, init_dim, 3, 1, 1),
                nn.Conv2d(init_dim, 3, 3, 1, 1)
            ])

        self.UPNet2 = nn.Sequential(*[
                nn.Conv2d(init_dim, init_dim, 3, 1, 1),
                nn.Conv2d(init_dim, 3, 3, 1, 1)
            ])

        self.UPNet4 = nn.Sequential(*[
                nn.Conv2d(init_dim*2, init_dim, 3, 1, 1),
                nn.Conv2d(init_dim, 3, 3, 1, 1)
            ])

        self.Down1 = nn.Conv2d(init_dim, init_dim, 3, 2, 1)
        self.Down2 = nn.Conv2d(init_dim, init_dim*2, 3, 2, 1)

        self.Up1 = nn.ConvTranspose2d(init_dim, init_dim, 4, 2, 1)
        self.Up2 = nn.ConvTranspose2d(init_dim*2, init_dim, 4, 2, 1) 

        self.Img_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 

    def forward(self, x):
        input_x = x[0]
        prev_s1, prev_s2, prev_s4 = x[1:4]
        prev_feat_s1, prev_feat_s2, prev_feat_s4 = x[4:7]

        f_first = F.relu(self.SFENet1(input_x))
        f_s1  = F.relu(self.SFENet2(f_first))
        f_s2 = self.Down1(self.RDBs[0](f_s1)) 
        f_s4 = self.Down2(self.RDBs[1](f_s2))

        f_s4 = f_s4 + self.RDBs[3](self.RDBs[2](f_s4)) + prev_feat_s4
        f_s2 = f_s2 + self.RDBs[4](self.Up2(f_s4)) + prev_feat_s2
        f_s1 = f_s1 + self.RDBs[5](self.Up1(f_s2))+ f_first + prev_feat_s1

        s4 = self.UPNet4(f_s4)
        s2 = self.UPNet2(f_s2) + self.Img_up(s4)
        s1 = self.UPNet(f_s1) + self.Img_up(s2)

        return [s1, s2, s4, f_s1, f_s2, f_s4]


class Model(nn.Module):
    def __init__(self, init_dim=16, grow_dim=8, nConvLayers=4, is_stage2_model=False):
        super(Model, self).__init__()
        self.is_stage2_model = is_stage2_model
        self.recur1 = DRBN_BU(init_dim, grow_dim, nConvLayers)
        self.recur2 = DRBN_BU(init_dim, grow_dim, nConvLayers)
        self.recur3 = DRBN_BU(init_dim, grow_dim, nConvLayers)
        self.recur4 = DRBN_BU(init_dim, grow_dim, nConvLayers)
        self.load_weight('weights/DRBN/weight_s1.pth')
        if self.is_stage2_model:
            self.load_weight('weights/DRBN/weight_s2.pth')
            self.recom = Recompose(init_dim, grow_dim, nConvLayers)
            

    def forward(self, x):
        res = self.recur1([torch.cat((x, x), 1), 0, 0, 0, 0, 0, 0])
        res = self.recur2([torch.cat((res[0], x), 1)] + res)
        res = self.recur3([torch.cat((res[0], x), 1)] + res)
        res = self.recur4([torch.cat((res[0], x), 1)] + res)
        if self.is_stage2_model:
            res = [self.recom([x, res[2], res[1], res[0]])]
        return res[0]

    def load_weight(self, weight_path):
        weight = torch.load(weight_path)
        self.load_state_dict(weight)


class Recompose(nn.Module):
    def __init__(self, init_dim=16, grow_dim=8, nConvLayers=4):
        super(Recompose, self).__init__()
        self.SFENet1 = nn.Conv2d(3 * 4, init_dim, 3, 1, 1)
        self.SFENet2 = nn.Conv2d(init_dim, init_dim, 3, 1, 1)
        self.RDBs = nn.Sequential(*[RDB(init_dim, grow_dim, nConvLayers)])
        self.GFF = nn.Sequential(*[
            nn.Conv2d(init_dim, init_dim, 1, 1, 0),
            nn.Conv2d(init_dim, init_dim, 3, 1, 1)
        ])

        self.UPNet = nn.Conv2d(init_dim, init_dim, 3, 1, 1)
        self.UPNet2 = nn.Conv2d(init_dim, 9, 3, 1, 1)
        self.load_weight('weights/DRBN/weight_r_s2.pth')
    
    def load_weight(self, weight_path):
        weight = torch.load(weight_path)
        self.load_state_dict(weight)

    def forward(self, x):
        x[1] = F.interpolate(x[1], scale_factor=4, mode='bilinear', align_corners=True)
        x[2] = F.interpolate(x[2], scale_factor=2, mode='bilinear', align_corners=True)
        lr, s0, s1, s2 = x[0], x[1], x[2]-x[1], x[3]-x[2]

        x = torch.cat((lr, s0, s1, s2), 1)
        f_1 = self.SFENet1(x)
        x = self.GFF(self.RDBs[0](self.SFENet2(f_1)))
        weight = self.UPNet2(self.UPNet(f_1 + x))
        weight = weight * 0.8 + 0.6 # ??? I don't understand here
        w0, w1, w2 = torch.split(weight, 3, 1)
        output = s0 * w0 + s1 * w1 + s2 * w2

        return output


if __name__ == '__main__':
    model = Model()
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    print(y.shape)
