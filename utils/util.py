import torch.nn.functional as F
def padding(image, divide_size=4):
    n, c, h, w = image.shape
    padding_h = divide_size - h % divide_size
    padding_w = divide_size - w % divide_size
    image = F.pad(image, (0, padding_w, 0, padding_h), "reflect")
    return image, h, w

def unpadding(image, h, w):
    return image[:, :, :h, :w]