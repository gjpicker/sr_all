import torch.utils.data as data
# from torchvision.transforms import *
# from os import listdir
# from os.path import join

import  torchvision.transforms as tff

import os 
# from PIL import Image
import random
import numpy as np 

import torch 
import h5py

# from . import dataset as old_dt 

from .utils import common  

def random_crop(hr, lr, size, scale):
    h, w = lr.shape[:-1]
    x = random.randint(0, w-size)
    y = random.randint(0, h-size)

    hsize = size*scale
    hx, hy = x*scale, y*scale

    crop_lr = lr[y:y+size, x:x+size].copy()
    crop_hr = hr[hy:hy+hsize, hx:hx+hsize].copy()

    return crop_hr, crop_lr


def random_flip_and_rotate(im1, im2):
    if random.random() < 0.5:
        im1 = np.flipud(im1)
        im2 = np.flipud(im2)

    if random.random() < 0.5:
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)

    angle = random.choice([0, 1, 2, 3])
    im1 = np.rot90(im1, angle)
    im2 = np.rot90(im2, angle)

    # have to copy before be called by transform function
    return im1.copy(), im2.copy()



class TrainDatasetFromFolder(data.Dataset):
    def __init__(self, image_index_path, is_gray=False, random_scale=True, crop_size=296, rotate=True, fliplr=True,
#     def __init__(self, image_dirs, is_gray=False, random_scale=True, crop_size=296, rotate=True, fliplr=True,
                 fliptb=True, scale_factor=4,is_debug_size=0,is_normlize=True):
                     
                            
        super(TrainDatasetFromFolder, self).__init__()

        scale = scale_factor 
        self.size = crop_size//scale_factor
        

        h5f = h5py.File(image_index_path, "r")

        self.hr = [v[:] for v in h5f["HR"].values()]
        # perform multi-scale training
        if scale == 0:
            self.scale = [2, 3, 4]
            self.lr = [[v[:] for v in h5f["X{}".format(i)].values()] for i in self.scale]
        else:
            self.scale = [scale]
            self.lr = [[v[:] for v in h5f["X{}".format(scale)].values()]]

        h5f.close()

        self.transform = tff.Compose([
            tff.ToTensor()
        ])

    def __getitem__(self, index):
        size = self.size

        item = [(self.hr[index], self.lr[i][index]) for i, _ in enumerate(self.lr)]

        item = [random_crop(hr, lr, size, self.scale[i]) for i, (hr, lr) in enumerate(item)]
        item = [random_flip_and_rotate(hr, lr) for hr, lr in item]

        return [(self.transform(hr), self.transform(lr)) for hr, lr in item]

    def __len__(self):
        return len(self.hr)




class TestDatasetFromFolder(data.Dataset):
    def __init__(self, image_root, scale_factor=4,is_normlize=True):
        self.is_normlize = is_normlize
        img_list = []
        for item in common.BENCHMARK:
            x_dir = os.path.abspath(os.path.join(image_root,item))
            if not  os.path.isdir(x_dir):
                continue
            tmp =  common.get_image_paths(data_type="img",dataroot=x_dir)
            tmp = [( item ,os.path.join(x_dir,x) ) for x in  tmp if "HR" in x]
            
            img_list+= tmp
        self. img_list = img_list 
        self. scale = scale_factor
        self.transform = tff.Compose([
            tff.ToTensor()
        ])

    def __getitem__(self, index):
        image_info  = self.img_list[index] 
        dt , hr_path = image_info
        lr_path = hr_path.replace("HR","LR_bicubic/X4")
        lr_path = lr_path.replace (".png","x4.png")
        lr = common.read_img(lr_path, "img")
        hr = common.read_img(hr_path, "img")
    
        lr_tensor, hr_tensor = common.np2Tensor([lr, hr], 1.)
        if lr_tensor.shape[0]<3:
            lr_tensor=torch.cat([lr_tensor,lr_tensor,lr_tensor],dim=0)

        if hr_tensor.shape[0]<3:
            hr_tensor=torch.cat([hr_tensor,hr_tensor,hr_tensor,],dim=0)
        return {'LR': lr_tensor, 'HR': hr_tensor, 'LR_path': lr_path, 'HR_path': hr_path}


        return lr, hr, lr_path, hr_path

    def __len__(self):
        return len(self.img_list) 


