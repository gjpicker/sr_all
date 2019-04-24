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
            self.image_filenames =f.readlines()
            self. image_filenames = [x.strip() for x in self.image_filenames]
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
#         img = load_img(self.image_filenames[index])
        img_id = np.random.choice(self.image_filenames)
        img = load_img(img_id)

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
#         hr_transform_common = Compose([Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor() , nm,])
        hr_transform_common = [Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor() ]
        hr_transform_vgg = Compose(hr_transform_common)
        hr_transform_norm = Compose(hr_transform_common+[nm])
        hr_img = hr_transform_norm(img)
        hr_img_vgg = hr_transform_vgg(img)

        # lr_img LR image
#         lr_transform = Compose([Resize((lr_img_w, lr_img_h), interpolation=Image.BICUBIC),  ToTensor() , nm, ])
#         lr_img = lr_transform(img)

        lr_transform_common = [Resize((lr_img_w, lr_img_h), interpolation=Image.BICUBIC), ToTensor() ]
        lr_transform_vgg = Compose(lr_transform_common)
        lr_transform_norm = Compose(lr_transform_common+[nm])
        lr_img = lr_transform_norm(img)
        lr_img_vgg = lr_transform_vgg(img)


        # Bicubic interpolated image
        bc_transform = Compose([Resize((lr_img_w, lr_img_h), interpolation=Image.BICUBIC), \
            Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor() ,nm])
        bc_img = bc_transform(img)



        return lr_img, hr_img, bc_img , lr_img_vgg, hr_img_vgg

    def __len__(self):
        if self.is_debug_size>0:
            return self.is_debug_size
        #return 64
        return len(self.image_filenames)


class TestDatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, is_gray=False, scale_factor=4):
        super(TestDatasetFromFolder, self).__init__()

        self.image_filenames = [join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x)]
        self.is_gray = is_gray
        self.scale_factor = scale_factor

    def __getitem__(self, index):
        # load image
        img = load_img(self.image_filenames[index])

        # original HR image size
        w = img.size[0]
        h = img.size[1]

        # determine valid HR image size with scale factor
        hr_img_w = calculate_valid_crop_size(w, self.scale_factor)
        hr_img_h = calculate_valid_crop_size(h, self.scale_factor)

        # determine lr_img LR image size
        lr_img_w = hr_img_w // self.scale_factor
        lr_img_h = hr_img_h // self.scale_factor

        # only Y-channel is super-resolved
        if self.is_gray:
            img = img.convert('YCbCr')
            # img, _, _ = lr_img.split()

        # hr_img HR image
        hr_transform = Compose([Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        hr_img = hr_transform(img)

        # lr_img LR image
        lr_transform = Compose([Resize((lr_img_w, lr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        lr_img = lr_transform(img)

        # Bicubic interpolated image
        bc_transform = Compose([Resize((lr_img_w, lr_img_h), interpolation=Image.BICUBIC),\
            Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        bc_img = bc_transform(lr_img)

        return lr_img, hr_img, bc_img

    def __len__(self):
        return len(self.image_filenames)
