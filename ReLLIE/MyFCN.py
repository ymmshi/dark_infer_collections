import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
import numpy as np
import math
import cv2
from chainer.links.caffe import CaffeFunction
import chainerrl
from chainerrl.agents import a3c


class DilatedConvBlock(chainer.Chain):

    def __init__(self, d_factor):
        super(DilatedConvBlock, self).__init__(
            diconv=L.DilatedConvolution2D(in_channels=64, out_channels=64, ksize=3, stride=1, pad=d_factor, dilate=d_factor, nobias=False),
        )

        self.train = True

    def __call__(self, x):
        h = F.relu(self.diconv(x))
        return h


class MyFcn(chainer.Chain, a3c.A3CModel):

    def __init__(self, n_actions):
        w = chainer.initializers.HeNormal()
        super(MyFcn, self).__init__(
            conv1=L.Convolution2D(3, 64, 3, stride=1, pad=1, nobias=False),
            diconv2=DilatedConvBlock(2),
            diconv3=DilatedConvBlock(3),
            diconv4=DilatedConvBlock(4),
            diconv5_pi=DilatedConvBlock(3),
            diconv6_pi=DilatedConvBlock(2),
            conv7_r_pi=chainerrl.policies.SoftmaxPolicy(
                L.Convolution2D(64, n_actions, 3, stride=1, pad=1, nobias=False)),
            conv7_g_pi=chainerrl.policies.SoftmaxPolicy(
                L.Convolution2D(64, n_actions, 3, stride=1, pad=1, nobias=False)),
            conv7_b_pi=chainerrl.policies.SoftmaxPolicy(
                L.Convolution2D(64, n_actions, 3, stride=1, pad=1, nobias=False)),
            diconv5_V=DilatedConvBlock(3),
            diconv6_V=DilatedConvBlock(2),
            conv7_V=L.Convolution2D(64, 1, 3, stride=1, pad=1, nobias=False),
        )
        self.train = True

    def pi_and_v(self, x):
        h = F.relu(self.conv1(x))
        h = self.diconv2(h)
        h = self.diconv3(h)
        h = self.diconv4(h)
        h_pi = self.diconv5_pi(h)
        h_pi = self.diconv6_pi(h_pi)
        pout_r = self.conv7_r_pi(h_pi)
        pout_g = self.conv7_g_pi(h_pi)
        pout_b = self.conv7_b_pi(h_pi)
        h_V = self.diconv5_V(h)
        h_V = self.diconv6_V(h_V)
        vout = self.conv7_V(h_V)

        return pout_r, pout_g, pout_b, vout


class MyFcn_denoise(chainer.Chain, a3c.A3CModel):
 
    def __init__(self, n_actions):
        w = chainer.initializers.HeNormal()
        #net = CaffeFunction('../initial_weight/zhang_cvpr17_denoise_15_gray.caffemodel')
        super(MyFcn_denoise, self).__init__(
            conv1=L.Convolution2D(3, 64, 3, stride=1, pad=1, nobias=False),
            diconv2=DilatedConvBlock(2),
            diconv3=DilatedConvBlock(3),
            diconv4=DilatedConvBlock(4),
            diconv5_pi=DilatedConvBlock(3),
            diconv6_pi=DilatedConvBlock(2),
            conv7_pi=chainerrl.policies.SoftmaxPolicy(L.Convolution2D(64, n_actions, 3, stride=1, pad=1, nobias=False)),
            diconv5_V=DilatedConvBlock(3),
            diconv6_V=DilatedConvBlock(2),
            conv7_V=L.Convolution2D(64, 1, 3, stride=1, pad=1, nobias=False),
        )
        self.train = True
 
    def pi_and_v(self, x):
        h = F.relu(self.conv1(x))
        h = self.diconv2(h)
        h = self.diconv3(h)
        h = self.diconv4(h)
        h_pi = self.diconv5_pi(h)
        h_pi = self.diconv6_pi(h_pi)
        de = self.conv7_pi(h_pi)
        #pout = np.concatenate((pout_r,pout_g,pout_b), axis=1)
        h_V = self.diconv5_V(h)
        h_V = self.diconv6_V(h_V)
        vout = self.conv7_V(h_V)
       
        return de, vout