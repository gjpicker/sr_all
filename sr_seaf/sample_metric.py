'''
Created on Apr 29, 2019

@author: gjpicker
'''
import torch 
import numpy as np 

import skimage
import cv2

import os 

import dataset as offset_dt 
import model2 as md 


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False


class read_dt (offset_dt.TestDatasetFromFolder):
    def __init__(self, image_index_path, is_gray=False, random_scale=True, crop_size=296, rotate=True, fliplr=True,
                 fliptb=True, scale_factor=4,is_debug_size=0,is_normlize=False):

        #super(TestDatasetFromFolder, self).__init__()
        super().__init__(image_index_path,is_gray,random_scale,crop_size,rotate,fliplr,fliptb,scale_factor,is_debug_size,is_normlize=is_normlize)
        with open(image_index_path) as f :
            self.image_filenames_tmp =f.readlines()
            self.image_filenames_tmp = [x.strip() for x in self.image_filenames_tmp]

            self.image_filenames = self.image_filenames_tmp
    

    pass 
    
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_cuda = torch.cuda.is_available()


def calculate_metrics(hr_y_list, sr_y_list, bnd=2):
    class BaseMetric:
        def __init__(self):
            self.name = 'base'

        def image_preprocess(self, image):
            image_copy = image.copy()
            image_copy[image_copy < 0] = 0
            image_copy[image_copy > 255] = 255
            image_copy = np.around(image_copy).astype(np.double)
            return image_copy

        def evaluate(self, gt, pr):
            pass

        def evaluate_list(self, gtlst, prlst):
            resultlist = list(map(lambda gt, pr: self.evaluate(gt, pr), gtlst, prlst))
            return sum(resultlist) / len(resultlist)


    class PSNRMetric(BaseMetric):
        def __init__(self):
            self.name = 'psnr'

        def evaluate(self, gt, pr):
            gt = self.image_preprocess(gt)
            pr = self.image_preprocess(pr)
            return skimage.measure.compare_psnr(gt, pr, data_range=255)


    class SSIMMetric(BaseMetric):
        def __init__(self):
            self.name = 'ssim'

        def evaluate(self, gt, pr):
            def ssim(img1, img2):
                C1 = (0.01 * 255) ** 2
                C2 = (0.03 * 255) ** 2

                img1 = img1.astype(np.float64)
                img2 = img2.astype(np.float64)
                kernel = cv2.getGaussianKernel(11, 1.5)
                window = np.outer(kernel, kernel.transpose())

                mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
                mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
                mu1_sq = mu1 ** 2
                mu2_sq = mu2 ** 2
                mu1_mu2 = mu1 * mu2
                sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
                sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
                sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

                ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
                return ssim_map.mean()

            def esrgan_ssim(img1, img2):
                if not img1.shape == img2.shape:
                    raise ValueError('Input images must have the same dimensions.')
                if img1.ndim == 2:
                    return ssim(img1, img2)
                elif img1.ndim == 3:
                    if img1.shape[2] == 3:
                        ssims = []
                        for i in range(3):
                            ssims.append(ssim(img1, img2))
                        return np.array(ssims).mean()
                    elif img1.shape[2] == 1:
                        return ssim(np.squeeze(img1), np.squeeze(img2))
                else:
                    raise ValueError('Wrong input image dimensions.')

            gt = self.image_preprocess(gt)
            pr = self.image_preprocess(pr)
            return esrgan_ssim(gt[..., 0], pr[..., 0])


    y_mean_psnr = 0
    y_mean_ssim = 0
    assert len(hr_y_list) == len(sr_y_list)
    for i in range(len(hr_y_list)):
        hr_y, sr_y = hr_y_list[i], sr_y_list[i]
        hr_y = hr_y[bnd:-bnd, bnd:-bnd, :]
        sr_y = sr_y[bnd:-bnd, bnd:-bnd, :]
        y_mean_psnr += PSNRMetric().evaluate(sr_y, hr_y) / len(sr_y_list)
        y_mean_ssim += SSIMMetric().evaluate(sr_y, hr_y) / len(sr_y_list)
    return y_mean_psnr, y_mean_ssim


def define_G(ge_net_str , g_path):
    netG = None 
    print (ge_net_str,"----"*4)
    if ge_net_str is None or ge_net_str=="":
        raise Exception("unknown G")
    if ge_net_str =="srfeat":
        raise Exception("not support multi-scale")
    elif ge_net_str =="carn":
        netG= md.G1()
    elif ge_net_str =="carnm":
        netG= md.G2()
    elif "carn_gan" in  ge_net_str:
        import sys 
        sys.path.append("/home/wangjian7/workspace/src/code_NNNNIPS/SR/wj_srfeat/sr_seaf/xx_model")

        if ge_net_str=="carn_gan":
            import m as gan_m 
            netG = gan_m.get_G(ge_net_str)
        elif ge_net_str=="carn_ganm":
            import m as gan_m 
            netG = gan_m.get_G(ge_net_str)
    else :
        raise Exception("unknow ")
    
    if os.path.isfile(g_path):
        if not is_cuda:
            netG.load_state_dict(\
                torch.load(g_path, map_location=lambda storage, loc: storage) )
        else :
            netG.load_state_dict(\
                torch.load(g_path))
    else :
        print ("===="*8,"fail load pretrained")
        
    return netG
        

def define_metric(v_list ,target_list ,ii,scale=4 ):
    '''
    convert to  pil.Image(numpy )
    then calc 
    '''
    import PIL.Image
    class to_numpy(object) :
        def __init__(self):
            self.fn = lambda grid :grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        def __call__(self,x):
            x = torchvision.utils.make_grid(x,padding=0,normalize=True)
            return self.fn(x)
    import torchvision .transforms as ttf 
    assert type(v_list[0])==torch.Tensor and v_list[0].type() == "torch.FloatTensor"
    tans = ttf.Compose([ to_numpy()])
    v_list= [tans(x.squeeze_(0)) for x in  v_list]

    target_list= [tans(x.squeeze_(0)) for x in  target_list]
    
    #os.makedirs("save",exist_ok=True)
    #data = np.concatenate([np.vstack(v_list),np.vstack(target_list) ],axis=1 ) 
    #cv2.imwrite("save/%s.jpg"%(str(ii).zfill(4) ) ,data ) 
        
    return calculate_metrics(target_list,v_list,scale)




if __name__=="__main__":
    device = torch.device("cpu")

    dt_list_dict ={
        "set14": "/home/wangjian7/download/fds_data/data/data/validate_scale4_Set14.log",\
        "set5": "/home/wangjian7/download/fds_data/data/data/validate_scale4_Set5.log",\
        "Urban100_SR": "/home/wangjian7/download/fds_data/data/data/validate_scale4_Urban100_SR.log",\
        "BSD100_SR": "/home/wangjian7/download/fds_data/data/data/validate_scale4_BSD100_SR.log",\
        
        }
    is_normlize=True
    #netG= define_G("carnm", "/home/wangjian7/workspace/src/code_NNNNIPS/SR/wj_srfeat/sr_seaf/save_ck/dl90_carn_nowarm_trainer2_yaml/best_net_G.pth")
    netG= define_G("carn", "/home/wangjian7/workspace/src/code_NNNNIPS/SR/wj_srfeat/sr_seaf/save_ck/dl49_carn_nowarm_yaml/best_net_G.pth")
    #netG= define_G("carn_gan","")
    #is_normlize=False
    #netG= define_G("carn_ganm","")
    #is_normlize=False
    #netG= define_G("carnm","")
    #is_normlize=False
    print (netG)
    
    print ("create G")
    netG= netG.to(device)
    print ("load G")
    import tqdm 
    import torchvision


    #px = "/home/wangjian7/download/fds_data/data/data/SR_data_index.log"
    #dl = torch.utils.data.DataLoader(read_dt(image_index_path=px))
    kw ={"pin_memory":True , "num_workers":8 } if torch.cuda.is_available() else {}
    dl_list = [(x,torch.utils.data.DataLoader(\
            read_dt(image_index_path=y,is_normlize=is_normlize),batch_size=16, **kw)) for x,y in dt_list_dict.items()]
    

    with torch.no_grad():
        for key,dl in dl_list:
            print ("dataset size ",len(dl),"-----"*4,key)
            s1_list = []
            s2_list = []
            for i,data in enumerate(dl) :
                assert len(data)==5
                lr4_,hr,cubic_, lr2, lr3 = data 
                #print (lr4_.mean(),lr4_.std() )
                out_hr = netG(lr4_.to(device))
                #torchvision.utils.save_image(out_hr,"a_%d.jpeg"%(i),normalize=True)
                #torchvision.utils.save_image(hr,"b_%d.jpeg"%(i),normalize=True)
                out_hr= out_hr.cpu()
                hr= hr .cpu()
                for one_batch in range( out_hr.size(0) ):
                    v1 = out_hr[one_batch].unsqueeze(0)
                    v2 = hr[one_batch].unsqueeze(0)
                    #s1,s2 = define_metric([out_hr.cpu()], [hr.cpu()] , i)
                    s1,s2 = define_metric([v1], [v2] , i)
                    s1_list .append(s1)
                    s2_list .append(s2)
                if i %20 ==0 and i!=0 :
                    print ("\npsnr:",np.mean(s1_list),"ssim",np.mean(s2_list))


            print ("===="*8,"\n",key,"psnr:",np.mean(s1_list),"ssim",np.mean(s2_list))






