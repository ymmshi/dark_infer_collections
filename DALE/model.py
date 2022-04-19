import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, dim, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Residual_Block(nn.Module):
    def __init__(self, in_num, out_num, dilation_factor):
        super(Residual_Block, self).__init__()
        self.conv1 = (nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=dilation_factor, dilation=dilation_factor, groups= 1, bias=False))
        self.in1 = nn.BatchNorm2d(out_num)

        self.conv2 = (nn.Conv2d(out_num, out_num, kernel_size=3, stride=1, padding=dilation_factor, dilation=dilation_factor, groups= 1, bias=False))
        self.in2 = nn.BatchNorm2d(out_num)
        self.se = SELayer(dim=out_num)

    def forward(self, x):
        identity_data = x
        output = F.relu((self.conv1(x)))
        output = self.conv2(output)
        output = self.se(output)
        output = output + identity_data
        return output


class DownSample(nn.Module):
    def __init__(self, in_dim, kernel_size=3, stride=2):
        super(DownSample, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim, kernel_size, stride=stride, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv2d(in_dim, in_dim, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

        self.avg_pool = nn.AvgPool2d(2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out


class UpSample(nn.Module):
    def __init__(self, in_dim, kernel_size=3, stride=2):
        super(UpSample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_dim, in_dim, kernel_size, stride=stride, padding=1)
        self.conv = nn.Conv2d(in_dim, in_dim, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x, output_size):
        out = F.relu(self.deconv(x, output_size=output_size))
        out = F.relu(self.conv(out))
        return out


class EnhancementNet(nn.Module):
    def __init__(self):
        super(EnhancementNet, self).__init__()
        self.res_input_conv = nn.Sequential(nn.Conv2d(6, 64, 3, 1, 1))
        self.residual_group1 = Residual_Block(64, 64, 3)
        self.residual_group2 = Residual_Block(64, 64, 2)
        self.residual_group3 = Residual_Block(64, 64, 1)
        self.se = SELayer(192)
        self.conv_block = nn.Sequential(
            nn.Conv2d(192, 64, 3, 1, 1),
            nn.Conv2d(64, 3, 3, 1, 1)
        )

    def load_weight(self, weight_file):
        weight = torch.load(weight_file)
        self.load_state_dict(weight)

    def forward(self, x, attention):
        res_input = self.res_input_conv(torch.cat([x,attention],1))
        res1 = self.residual_group1(res_input)
        res2 = self.residual_group2(res1)
        res3 = self.residual_group3(res2)
        group_cat = self.se(torch.cat([res1,res2,res3],1))
        output = self.conv_block(group_cat) + attention
        return output


class VisualAttentionNetwork(nn.Module):
    def __init__(self):
        super(VisualAttentionNetwork, self).__init__()
        self.res_input_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1)
        )

        self.res_encoder1 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            Residual_Block(64, 64, 3),
        )

        self.down1 = DownSample(64)

        self.res_encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 1),
            Residual_Block(128, 128, 2),
        )

        self.down2 = DownSample(128)

        self.res_encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, 1),
            Residual_Block(256, 256, 1),
        )

        self.res_decoder3 = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            Residual_Block(256, 256, 1),
        )
        self.up2 = UpSample(256)

        self.res_decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            Residual_Block(128, 128, 2),
        )
        self.up1 = UpSample(128)

        self.res_decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            Residual_Block(64, 64, 3),
        )

        self.res_final = nn.Conv2d(64, 3, 3, 1, 1)

    def load_weight(self, weight_file):
        weight = torch.load(weight_file)
        self.load_state_dict(weight)

    def forward(self, x):
        res_input = self.res_input_conv(x)

        encoder1 = self.res_encoder1(res_input)
        encoder1_down = self.down1(encoder1)

        encoder2 = self.res_encoder2(encoder1_down)
        encoder2_down = self.down2(encoder2)

        encoder3 = self.res_encoder3(encoder2_down)

        decoder3 = self.res_decoder3(encoder3) + encoder3
        decoder3 = self.up2(decoder3, output_size=encoder2.size())

        decoder2 = self.res_decoder2(decoder3) + encoder2
        decoder2 = self.up1(decoder2, output_size=encoder1.size())

        decoder1 = self.res_decoder1(decoder2) + encoder1

        output = self.res_final(decoder1)

        return output


class Model(nn.Module):
    def __init__(self,):
        super(Model, self).__init__()
        self.van = VisualAttentionNetwork()
        self.en = EnhancementNet()
    
    def load_weight(self, weight_van, weight_en):
        self.van.load_weight(weight_van)
        self.en.load_weight(weight_en)

    def forward(self, x):
        attn_map = self.van(x)
        res = self.en(x, attn_map)
        return res
    

if __name__ == '__main__':
    model = Model()
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    print(y.shape)