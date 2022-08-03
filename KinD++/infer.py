from glob import glob
import os
import numpy as np
from PIL import Image
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tqdm import tqdm

model_name = 'KinD++'
in_path = './input/test.png'
out_path = './output/' + model_name

def load_model():
    from model import Model
    model = Model()
    model.load_weight()
    return model


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
    for input_path in tqdm(input_paths):
        output_path = input_path.replace(in_path, out_path)
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        img = Image.open(input_path)
        img = np.array(img, dtype="float32") / 255.0
        img_max = np.max(img)
        img_min = np.min(img)
        img = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))

        tic = time.time()
        output = model(img)
        toc = time.time()
        total_time += toc - tic

        output = Image.fromarray(np.clip(np.squeeze(output) * 255.0, 0, 255.0).astype('uint8'))
        output.save(output_path)
    print('{} Total time: {:.4f}s Speed: {:.4f}s/img'.format(model_name, total_time, total_time / len(input_paths)))



if __name__ == '__main__':
    model = load_model()
    input_paths = load_data_paths()
    inference(model, input_paths)
    
