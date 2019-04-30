'''
Created on Apr 30, 2019

@author: gjpicker
'''
import torch 
import numpy as np 

import skimage
import cv2

import os 

from wj_SRFeat.new_sr import data as offset_dt
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



# import sys
# sys.path.append("/home/wangjian7/workspace/src/code_NNNNIPS/SR/wj_srfeat/SRFBN_CVPR19")
# import utils.cvpr_util as util

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
        



def define_metric(v_list ,target_list ,model_name ,data_name,ii):
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
            x = self.fn(x)
            #x = rgb2y(x)
            return x
    import torchvision .transforms as ttf 
    assert type(v_list[0])==torch.Tensor and v_list[0].type() == "torch.FloatTensor"
    tans = ttf.Compose([ to_numpy()])
    v_list= [tans(x.squeeze_(0)) for x in  v_list]

    target_list= [tans(x.squeeze_(0)) for x in  target_list]
    
    os.makedirs(model_name,exist_ok=True)
    os.makedirs(os.path.join(model_name,"gen_"+data_name) ,exist_ok=True)
    os.makedirs(os.path.join(model_name,"gt_"+data_name) ,exist_ok=True)
    for iiii ,(x,y)  in enumerate(zip(v_list,target_list) ): 
        #data = np.concatenate([np.vstack(v_list),np.vstack(target_list) ],axis=1 ) 
        cv2.imwrite("%s/gen_%s/%s_%s.jpg"%(model_name,data_name,str(ii).zfill(4), str(iiii).zfill(4) ) , x[:,:,::-1] ) 
        cv2.imwrite("%s/gt_%s/%s_%s.jpg"%(model_name,data_name,str(ii).zfill(4), str(iiii).zfill(4) ) , y[:,:,::-1] ) 
        
    #return calculate_metrics(target_list,v_list,bnd=4)
        x1,x2 = util.calc_metrics(x,y,4)
        #print ("psnr:",x1,"ssim:",x2)

        return x1,x2



if __name__=="__main__":
    import sys
    device = torch.device("cpu")

    dt_list_dict ={
        "set14": "/home/wangjian7/download/fds_data/data/data/validate_scale4_Set14.log",\
        "set5": "/home/wangjian7/download/fds_data/data/data/validate_scale4_Set5.log",\
        "Urban100_SR": "/home/wangjian7/download/fds_data/data/data/validate_scale4_Urban100_SR.log",\
        "BSD100_SR": "/home/wangjian7/download/fds_data/data/data/validate_scale4_BSD100_SR.log",\
        
        }
    ch = sys.argv[1]
    if ch =="carnm":
        model_name ="carnm" 
        px= "/home/wangjian7/workspace/src/code_NNNNIPS/SR/wj_srfeat/sr_seaf/save_ck/dl90_carn_nowarm_trainer2_yaml/best_net_G.pth"
        is_normlize=True
    elif ch =="carn":
        model_name ="carn"
        px="/home/wangjian7/workspace/src/code_NNNNIPS/SR/wj_srfeat/sr_seaf/save_ck/dl49_carn_nowarm_yaml/best_net_G.pth"
        is_normlize=True

    elif ch =="carn_gan":
        model_name ="carn_gan"
        px=""
        is_normlize=False

    elif ch=="carn_ganm":
        model_name ="carn_ganm"
        px=""
        is_normlize=False

    #netG= define_G(model_name, "/home/wangjian7/workspace/src/code_NNNNIPS/SR/wj_srfeat/sr_seaf/save_ck/dl90_carn_nowarm_trainer2_yaml/best_net_G.pth")
    #netG= define_G(model_name, "/home/wangjian7/workspace/src/code_NNNNIPS/SR/wj_srfeat/sr_seaf/save_ck/dl49_carn_nowarm_yaml/best_net_G.pth")

    netG= define_G(model_name,px)
    #netG= define_G(model_name,"")
    #netG= define_G(model_name,"")
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
                    s1,s2 = define_metric([v1], [v2] ,  data_name=key,model_name = model_name ,ii=i)
                    s1_list .append(s1)
                    s2_list .append(s2)
                if i %20 ==0 and i!=0 :
                    print ("\npsnr:",np.mean(s1_list),"ssim",np.mean(s2_list))


            print ("===="*8,"\n",key,"psnr:",np.mean(s1_list),"ssim",np.mean(s2_list))






