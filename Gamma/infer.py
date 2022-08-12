import os
from PIL import Image
import numpy as np
import time
from tqdm import tqdm

model_name = 'Gamma'
in_path = './input/test.png'
out_path = './output/' + model_name
enlight_ratio = 1.8 # the larger, the brighter

def load_model():
    model = Gamma
    return model

def Gamma(img):
    bright_arr = np.array(img) * enlight_ratio
    bright_arr = bright_arr.astype('uint8')
    image_bright = Image.fromarray(bright_arr)
    return image_bright

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

        tic = time.time()
        output = model(img)
        toc = time.time()
        total_time += toc - tic

        output.save(output_path)
    print('{} Total time: {:.4f}s Speed: {:.4f}s/img'.format(model_name, total_time, total_time / len(input_paths)))


if __name__ == '__main__':
    model = load_model()
    input_paths = load_data_paths()
    input_paths = input_paths[:]
    inference(model, input_paths)
    

