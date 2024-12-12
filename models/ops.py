import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def euclidean_distance(x, y):
    out = -2*torch.matmul(x, y)
    out += (x**2).sum(dim=-1, keepdim=True)
    out += (y**2).sum(dim=-2, keepdim=True)
    return out


def calc_padding(x, patchsize, stride, padding=None):
    if padding is None:
        xdim = x.shape[2:]
        padvert = -(xdim[0] - patchsize) % stride
        padhorz = -(xdim[1] - patchsize) % stride

        padtop = int(np.floor(padvert / 2.0))
        padbottom = int(np.ceil(padvert / 2.0))
        padleft = int(np.floor(padhorz / 2.0))
        padright = int(np.ceil(padhorz / 2.0))
    else:
        padtop = padbottom = padleft = padright = padding

    return padtop, padbottom, padleft, padright
