from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect
import re
import numpy as np
import os
import collections


import pathlib

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8,is_norm=False):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if is_norm :
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    else:
        mean = np.array([0,0,0])
        std = np.array([1,1,1])

    if image_numpy.shape[0] == 1:

        # posemap
        image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    else:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * std + mean) * 255.0
    return np.clip( image_numpy.astype(imtype) ,0,255 )


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    try:
        image_pil = Image.fromarray(image_numpy)
        image_pil.save(image_path)
    except:
        import pdb; pdb.set_trace()


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
	pathlib.Path(path).mkdir(parents=True, exist_ok=True)

#    if not os.path.exists(path):
#        os.makedirs(path)
