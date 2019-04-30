'''
Created on Apr 30, 2019

@author: a11
'''
import torch 
from utils import quality as metric

import torch.utils.data 
import torchvision.transforms as tff 
import numpy as np 
import PIL.Image 

import os 

def np2Tensor(l, rgb_range=255):
    '''
    copy from SRFBN_CVPR19
    '''
    def _np2Tensor(img):
        # if img.shape[2] == 3: # for opencv imread
        #     img = img[:, :, [2, 1, 0]]
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255.)

        return tensor

    if type(l) ==list :
        return [_np2Tensor(_l) for _l in l]
    else :
        return _np2Tensor(l)

def quantize(img, rgb_range=255):
    pixel_range = 255. / rgb_range
    # return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
    return img.mul(pixel_range).clamp(0, 255).round()

def Tensor2np(tensor_list, rgb_range=255):

    def _Tensor2numpy(tensor, rgb_range=255):
        array = np.transpose(quantize(tensor, rgb_range).numpy(), (1, 2, 0)).astype(np.uint8)
        return array
    if type(tensor_list) ==list :
        return [_Tensor2numpy(tensor, rgb_range) for tensor in tensor_list]
    else :
        return _Tensor2numpy(tensor_list)




class read_dt (torch.utils.data .Dataset):
    def __init__(self, lr_index_log,hr_index_log=None,is_normlize=False):

        def read_fn(fpath) :
            if type(fpath)==list :
                return fpath
            
            with open(fpath) as f :
                image_filenames_tmp =f.readlines()
                image_filenames_tmp = [x.strip() for x in image_filenames_tmp]
            return image_filenames_tmp
            
        lr_list = sorted(read_fn(lr_index_log))
        hr_list = [x.replace("LR.png","HR.png") for x in lr_list]
        assert len(lr_list)==len(hr_list)
        
        self.transform = []
        if is_normlize :
            self.transform.append(\
                                   tff.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) )
        else  :
            self.transform.append(np2Tensor)
        self.transform = tff.Compose(self.transform)
        
        self.image_list = list(zip(lr_list,hr_list))

    def __getitem__(self, index):
        lr_p,hr_p = self.image_list[index]
        assert os.path.isfile(lr_p)
        assert os.path.isfile(hr_p)
        lr= np.array(PIL.Image.open(lr_p),dtype=np.float32)
        hr= np.array(PIL.Image.open(hr_p),dtype=np.float32)
        
        return {"lr":self.transform(lr).float(),"hr":self.transform(hr).float()}
    def __len__(self):
        return len(self.image_list)
        




def go_eval(netG,dataset_dt ,dt_kw =[],  g_kw={}, crop_size=74):


	scale=None
	def _overlap_crop_forward(x, shave=10, min_size=100000, bic=None):
		"""
		chop for less memory consumption during test
		"""
		n_GPUs = 2
		#scale = self.scale
		b, c, h, w = x.size()
		h_half, w_half = h // 2, w // 2
		h_size, w_size = h_half + shave, w_half + shave
		lr_list = [
			x[:, :, 0:h_size, 0:w_size],
			x[:, :, 0:h_size, (w - w_size):w],
			x[:, :, (h - h_size):h, 0:w_size],
			x[:, :, (h - h_size):h, (w - w_size):w]]

		if bic is not None:
			bic_h_size = h_size*scale
			bic_w_size = w_size*scale
			bic_h = h*scale
			bic_w = w*scale
			
			bic_list = [
				bic[:, :, 0:bic_h_size, 0:bic_w_size],
				bic[:, :, 0:bic_h_size, (bic_w - bic_w_size):bic_w],
				bic[:, :, (bic_h - bic_h_size):bic_h, 0:bic_w_size],
				bic[:, :, (bic_h - bic_h_size):bic_h, (bic_w - bic_w_size):bic_w]]

		if w_size * h_size < min_size:
			sr_list = []
			for i in range(0, 4, n_GPUs):
				lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
				if bic is not None:
					bic_batch = torch.cat(bic_list[i:(i + n_GPUs)], dim=0)

				sr_batch_temp = netG(lr_batch,scale=4)

				if isinstance(sr_batch_temp, list):
					sr_batch = sr_batch_temp[-1]
				else:
					sr_batch = sr_batch_temp

				sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
		else:
			sr_list = [
				_overlap_crop_forward(patch, shave=shave, min_size=min_size) \
				for patch in lr_list
				]

		h, w = scale * h, scale * w
		h_half, w_half = scale * h_half, scale * w_half
		h_size, w_size = scale * h_size, scale * w_size
		shave *= scale

		output = x.new(b, c, h, w)
		output[:, :, 0:h_half, 0:w_half] \
			= sr_list[0][:, :, 0:h_half, 0:w_half]
		output[:, :, 0:h_half, w_half:w] \
			= sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
		output[:, :, h_half:h, 0:w_half] \
			= sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
		output[:, :, h_half:h, w_half:w] \
			= sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

		return output

	dl =torch.utils.data.DataLoader(dataset_dt,batch_size=1)
	psnr_list = [] 
	ssim_list = [] 
	for i ,data in  enumerate(dl):
		if dt_kw is None :
			lr ,hr = data
		else :
			lr ,hr = data.get(dt_kw[0]),data.get(dt_kw[1])
		
		scale =  g_kw.get("scale") if "scale" in g_kw else 4 
		o_hr = _overlap_crop_forward(lr)
		assert (o_hr.shape == hr.shape)
		o_hr = Tensor2np(o_hr[0].cpu())
		hr = Tensor2np(hr[0].cpu())

		psnr ,ssim = metric.calc_metrics(o_hr,hr,crop_border=scale)
		
		psnr_list.append(psnr)
		ssim_list.append(ssim)

	print ("psnr:",np.mean(psnr_list), "ssim:",np.mean(ssim_list) )




if __name__=="__main__":
	#index_file = {"Set14":[],"Set5":[],"BSD100_SR":[],"Urban100_SR":[]}
	index_file = {"Set5":[]}
	with open("images/val.txt") as f :
		origin_f=[x.strip() for x in f.readlines()]
		for line in origin_f:
			for k in index_file.keys():
				if k in line:
					index_file[k].append(line)
	print ("====="*8)      
	print ("dataset amount")          
	for k in index_file.keys():
		print (k,len(index_file[k]),"...")
		
		
	import models.nets.pcarn.pcarn as pc 
     
	with torch.no_grad():
		netG =  pc.Net()
		for k in index_file.keys():
			
			dt = read_dt(lr_index_log=index_file[k])
			go_eval(netG,dt ,dt_kw =["lr","hr"],g_kw={"scale":4})
		

