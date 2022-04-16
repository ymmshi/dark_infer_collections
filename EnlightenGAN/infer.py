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

in_path = './input/test.png'
out_path = './output/EnlightenGAN'


def load_model():
    from model import Model
    model = Model()
    return model.eval().cuda()


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


def inference(model, input_paths):
    global in_path, out_path
    total_time = 0
    ts = transforms.Compose([transforms.ToTensor(), 
                             transforms.Normalize(0.5, 0.5)])
    with torch.no_grad():
        for input_path in input_paths:
            output_path = input_path.replace(in_path, out_path)
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))

            img = Image.open(input_path)
            img = ts(img).unsqueeze(0).cuda()
            
            img, h, w = padding(img, 16)
            tic = time.time()
            output = model(img)
            toc = time.time()
            output = unpadding(output, h, w)
            total_time += toc - tic

            torchvision.utils.save_image(output, output_path, normalize=True, range=(-1, 1))
    print('EnlightenGAN Total time: {:.4f}s Speed: {:.4f}s/img'.format(total_time, total_time / len(input_paths)))


if __name__ == '__main__':
    model = load_model()
    input_paths = load_data_paths()
    inference(model, input_paths)
    
