import torch
import torchvision
from torchvision import transforms
from glob import glob
import os
from PIL import Image
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.util import *

import MyFCN
import pixelwise_a3c_de
import pixelwise_a3c_el

model_name = 'ReLLIE'
in_path = './input/test.png'
out_path = './output/' + model_name

EPISODE_LEN = 3 # need to be modified by yourself
LEARNING_RATE = 0.0005
GAMMA = 1.05  # discount factor
N_ACTIONS = 27
MOVE_RANGE = 27  # number of actions that move the pixel values. e.g., when MOVE_RANGE=3, there are three actions: pixel_value+=1, +=0, -=1.

def load_model():
    from model import Model
    model = Model()
    model.load_weight('weights/ReLLIE/weight.pth')
    model = model.eval().cuda()

    model_el = MyFCN.MyFcn(N_ACTIONS)
    optimizer_el = pixelwise_a3c_el.chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer_el.setup(model_el)
    agent_el = pixelwise_a3c_el.PixelWiseA3C(model_el, optimizer_el, EPISODE_LEN, GAMMA)
    pixelwise_a3c_el.chainer.serializers.load_npz('./weights/ReLLIE/pretrained/model.npz', agent_el.model)
    agent_el.act_deterministically = True
    agent_el.model.to_gpu()

    model_de = MyFCN.MyFcn_denoise(2)
    optimizer_de = pixelwise_a3c_de.chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer_de.setup(model_de)
    agent_de = pixelwise_a3c_de.PixelWiseA3C(model_de, optimizer_de, EPISODE_LEN, GAMMA)
    pixelwise_a3c_de.chainer.serializers.load_npz('./weights/ReLLIE/pretrained/init_denoising.npz', agent_de.model)
    agent_de.act_deterministically = True
    agent_de.model.to_gpu()

    return model.eval().cuda(), agent_el, agent_de


def load_data_paths():
    global in_path, out_path
    if os.path.isfile(in_path):
        input_paths = [in_path]
        in_path = os.path.dirname(in_path)
    elif os.path.isdir(in_path):
        input_paths = []
        for root, dirs, files in os.walk(in_path):
            for name in files:
                for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
                    if name.lower().endswith(ext):
                        input_paths.append(os.path.join(root, name))
    return input_paths


class State_de():
    def __init__(self, move_range, model):
        self.cur_img = None
        self.move_range = move_range
        self.net = model
    
    def reset(self, x):
        self.low_img = x
        self.cur_img = x
        torch.clamp_(self.low_img, 3 / 255, 1)

    def step_el(self, act):
        move = torch.tensor(act, dtype=self.cur_img.dtype, device=self.cur_img.device)
        moves = (move - 6) / 20
        moved_image = self.cur_img + (0.1 * moves + 0.9 * moves[:,0:1,:,:]) * self.cur_img * (1 - self.cur_img)
        self.cur_img = 0.8 * moved_image + 0.2 * self.cur_img

    def step_de(self, act_b):
        # noise level map
        nsigma = (self.cur_img - self.low_img) / self.low_img
        nsigma = nsigma.max() * 2 * (nsigma - nsigma.min()) / (nsigma.max() - nsigma.min())
        torch.clamp_(nsigma, 0)
        nsigma = nsigma / 255.
        nsigma = nsigma[:, :, ::2, ::2]

        # Estimate noise and subtract it to the input image
        estim_noise = self.net(self.cur_img, nsigma)
        self.cur_img = torch.clamp(self.cur_img - estim_noise, 0., 1.)


def test(raw_x, agent_el, agent_de, model):
    current_state = State_de(MOVE_RANGE, model)
    current_state.reset(raw_x)

    for t in range(EPISODE_LEN):
        action_el = agent_el.act(current_state.cur_img.cpu().detach().numpy())
        current_state.step_el(action_el)
        if t > 4:
            action_de = agent_de.act(current_state.cur_img.cpu().detach().numpy())
            current_state.step_de(action_de)

    agent_de.stop_episode()

    return current_state.cur_img

def inference(model, input_paths):
    global in_path, out_path
    total_time = 0
    ts = transforms.ToTensor()
    model, agent_el, agent_de = model

    with torch.no_grad():
        for input_path in input_paths:
            output_path = input_path.replace(in_path, out_path)
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))

            img = Image.open(input_path)
            img = ts(img).unsqueeze(0).cuda()
            
            img, h, w = padding(img, 8)
            tic = time.time()
            output = test(img, agent_el, agent_de, model)
            toc = time.time()
            output = unpadding(output, h, w)
            total_time += toc - tic

            torchvision.utils.save_image(output, output_path)
    print('{} Total time: {:.4f}s Speed: {:.4f}s/img'.format(model_name, total_time, total_time / len(input_paths)))


if __name__ == '__main__':
    model = load_model()
    input_paths = load_data_paths()
    inference(model, input_paths)
    
