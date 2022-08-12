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
from models import basicblock as B

class LIRCNN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=16):
        super(LIRCNN, self).__init__()
        self.model = B.IRCNN(in_nc, out_nc, nc)

    def forward(self, x):
        n = self.model(x)
        return x*n
