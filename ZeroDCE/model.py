import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=32):
        super(Model, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, dim, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(dim*2, dim, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(dim*2, dim, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(dim*2, 24, 3, 1, 1, bias=True)
        self.load_weight()

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        for r in torch.split(x_r, 3, dim=1):
            x = x + r * (torch.pow(x, 2) - x)
        return x

    def load_weight(self, weight_path='ZeroDCE/weight.pth'):
        weight = torch.load(weight_path)
        self.load_state_dict(weight)



if __name__ == '__main__':
    model = Model()
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    print(y.shape)
