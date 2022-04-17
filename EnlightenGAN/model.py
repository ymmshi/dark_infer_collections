import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(EncoderBlock, self).__init__()
        self.en1 = nn.Sequential(*[
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_dim),
        ])
        self.en2 = nn.Sequential(*[
            nn.Conv2d(out_dim, out_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_dim),
        ])
        

    def forward(self, x, mask=None):
        if mask is None:
            return self.en2(self.en1(x))
        else:
            x = self.en1(x)
            x = x * mask
            x = self.en2(x)
            return x 

class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, last_dec=False):
        super(DecoderBlock, self).__init__()
        self.deconv = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        if last_dec:
            self.de = nn.Sequential(*[
                nn.Conv2d(in_dim, out_dim, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(out_dim),
                nn.Conv2d(out_dim, out_dim, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        else:
            self.de = nn.Sequential(*[
                nn.Conv2d(in_dim, out_dim, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(out_dim),
                nn.Conv2d(out_dim, out_dim, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(out_dim),
            ])
        

    def forward(self, dec_feat, enc_feat):
        dec_feat = F.interpolate(dec_feat, 
                                 scale_factor=2, 
                                 mode='bilinear', 
                                 align_corners=True)
        feat = torch.cat([self.deconv(dec_feat), enc_feat], 1)
        return self.de(feat)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.max_pool = nn.MaxPool2d(2)

        self.encoder1 = EncoderBlock(4, 32)
        self.encoder2 = EncoderBlock(32, 64)
        self.encoder3 = EncoderBlock(64, 128)
        self.encoder4 = EncoderBlock(128, 256)
        self.encoder5 = EncoderBlock(256, 512)

        self.decoder6 = DecoderBlock(512, 256)
        self.decoder7 = DecoderBlock(256, 128)
        self.decoder8 = DecoderBlock(128, 64)
        self.decoder9 = DecoderBlock(64, 32, True)
        self.decoder10 = nn.Conv2d(32, 3, 1)
    
    def to_gray(self, img):
        gray = 1. - (0.299 * img[:, 0, :, :] + 
                     0.587 * img[:, 1, :, :] + 
                     0.114 * img[:, 2, :, :] + 1.) / 2.
        gray = gray.unsqueeze(1)
        grays = [gray]
        for _ in range(4):
            gray = self.max_pool(gray)
            grays.append(gray)
        return grays

    def forward(self, img):
        grays = self.to_gray(img)

        conv1 = self.encoder1(torch.cat((img, grays[0]), 1))
        conv2 = self.encoder2(self.max_pool(conv1))
        conv3 = self.encoder3(self.max_pool(conv2))  
        conv4 = self.encoder4(self.max_pool(conv3))
        x = self.encoder5(self.max_pool(conv4), grays[4])

        x = self.decoder6(x, conv4 * grays[3])
        x = self.decoder7(x, conv3 * grays[2])
        x = self.decoder8(x, conv2 * grays[1])
        x = self.decoder9(x, conv1 * grays[0])
        x = self.decoder10(x)

        output = x * grays[0] + img
        return output

    def load_weight(self, weight_path):
        weight = torch.load(weight_path)
        self.load_state_dict(weight)


if __name__ == '__main__':
    model = Model()
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    print(y.shape)
