import torch

if __name__ == '__main__':
    weight = torch.load('pre_weights/ReLLIE/net_rgb.pth')
    new_weight = {}
    for k, v in weight.items():
        new_weight[k[7:]] = v
    torch.save(new_weight, 'weights/ReLLIE/weight.pth')
