import torch.utils.data as data
from torchvision.transforms import *
from os import listdir
from os.path import join
import os 
from PIL import Image
import random
import numpy as np 

import torch 


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


def calculate_valid_crop_size(crop_size, scale_factor):
    return crop_size - (crop_size % scale_factor)


class TrainDatasetFromFolder(data.Dataset):
    def __init__(self, image_index_path, is_gray=False, random_scale=True, crop_size=296, rotate=True, fliplr=True,
#     def __init__(self, image_dirs, is_gray=False, random_scale=True, crop_size=296, rotate=True, fliplr=True,
                 fliptb=True, scale_factor=4,is_debug_size=0):
                     
        super(TrainDatasetFromFolder, self).__init__()

        self.is_debug_size =is_debug_size;
        self.image_filenames = []
        assert os.path.isfile(image_index_path) ,"expect a exist path, but %s"%(image_index_path)
        with open(image_index_path) as f :
            self.image_filenames_tmp =f.readlines()
            self.image_filenames_tmp = [x.strip() for x in self.image_filenames_tmp]
            
            if "imagenet" in image_index_path.lower():

                self.image_filenames = [x for x in self.image_filenames_tmp if "/train/" in x]
                self.image_filenames_val = [x for x in self.image_filenames_tmp if "/test/" in x or "/val/"  in x ]

            else:
                self.image_filenames = [x for x in self.image_filenames_tmp if "_train_" in x]
                self.image_filenames_val = [x for x in self.image_filenames_tmp if "_test_" in x or "_val_"  in x  or "_valid_" in x]
        #for image_dir in image_dirs:
        #    self.image_filenames.extend(join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x))
        self.is_gray = is_gray
        self.random_scale = random_scale
        self.crop_size = crop_size
        self.rotate = rotate
        self.fliplr = fliplr
        self.fliptb = fliptb
        self.scale_factor = scale_factor

    def __getitem__(self, index):
        # load image
        ## random choice 
        img = load_img(self.image_filenames[index])
        #img_id = np.random.choice(self.image_filenames)
        #img = load_img(img_id)

        # determine valid HR image size with scale factor
        self.crop_size = calculate_valid_crop_size(self.crop_size, self.scale_factor)
        hr_img_w = self.crop_size
        hr_img_h = self.crop_size

        # determine LR image size
        lr_img_w = hr_img_w // self.scale_factor
        lr_img_h = hr_img_h // self.scale_factor

        # random scaling between [0.5, 1.0]
        if self.random_scale:
            eps = 1e-3
            ratio = random.randint(5, 10) * 0.1
            if hr_img_w * ratio < self.crop_size:
                ratio = self.crop_size / hr_img_w + eps
            if hr_img_h * ratio < self.crop_size:
                ratio = self.crop_size / hr_img_h + eps

            scale_w = int(hr_img_w * ratio)
            scale_h = int(hr_img_h * ratio)
            transform = Resize((scale_w, scale_h), interpolation=Image.BICUBIC)
            img = transform(img)

        # random crop
        transform = RandomCrop(self.crop_size)
        img = transform(img)

        # random rotation between [90, 180, 270] degrees
        if self.rotate:
            rv = random.randint(1, 3)
            img = img.rotate(90 * rv, expand=True)

        # random horizontal flip
        if self.fliplr:
            transform = RandomHorizontalFlip()
            img = transform(img)

        # random vertical flip
        if self.fliptb:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # only Y-channel is super-resolved
        if self.is_gray:
            img = img.convert('YCbCr')
            # img, _, _ = img.split()

        nm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # hr_img HR image
        hr_transform_common = Compose([Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor() , nm,])
        hr_img = hr_transform_common(img)

        # lr_img LR image
        lr_transform = Compose([Resize((lr_img_w, lr_img_h), interpolation=Image.BICUBIC),  ToTensor() , nm, ])
        lr_img = lr_transform(img)

        lr_transform = Compose([Resize((hr_img_w // 3, hr_img_h // 3), interpolation=Image.BICUBIC),  ToTensor() , nm, ])
        lr_img_3 = lr_transform(img)

        lr_transform = Compose([Resize((hr_img_w // 2, hr_img_h // 2), interpolation=Image.BICUBIC),  ToTensor() , nm, ])
        lr_img_2 = lr_transform(img)



        # Bicubic interpolated image
        bc_transform = Compose([Resize((lr_img_w, lr_img_h), interpolation=Image.BICUBIC), \
            Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor() ,nm])
        bc_img = bc_transform(img)



        return lr_img, hr_img, bc_img , lr_img_3, lr_img_2

    def __len__(self):
        if self.is_debug_size>0:
            return self.is_debug_size
        #return 64
        return len(self.image_filenames)


class TestDatasetFromFolder(TrainDatasetFromFolder):
    def __init__(self, image_index_path, is_gray=False, random_scale=True, crop_size=296, rotate=True, fliplr=True,
                 fliptb=True, scale_factor=4,is_debug_size=0):

        #super(TestDatasetFromFolder, self).__init__()
        rotate=False
        fliplr=False
        fliptb=False

        super().__init__(image_index_path,is_gray,random_scale,crop_size,rotate,fliplr,fliptb,scale_factor,is_debug_size)
        self.image_filenames = self.image_filenames_val




