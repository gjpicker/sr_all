'''
Created on Apr 18, 2019

@author: gjpicker
'''
import torch 
import torch.nn  as nn
import torchvision

from torch.nn import init


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>



class  ResidualBlock(nn.Module):
    def __init__(self,in_c_and_out_c=64,group=1 ):
        super(ResidualBlock, self).__init__()

        self.res=nn.Sequential(*[
        
            nn.Conv2d(in_c_and_out_c,in_c_and_out_c,\
            kernel_size=3,stride=1,padding=1,groups=group),
            
            nn.BatchNorm2d(in_c_and_out_c),
            
            nn.LeakyReLU(),
            
            nn.Conv2d(in_c_and_out_c,in_c_and_out_c,\
            kernel_size=3,stride=1,padding=1,groups=group),
            
            nn.BatchNorm2d(in_c_and_out_c),
        
        ])
        
    
    def forward(self,x):
        x_new = self.res(x)
        assert x_new.shape ==x.shape ,"error design"
        return x_new+x 
         
        
        
class G(nn.Module):
    def __init__(self,in_dim=3,res_channel=64,group=1,\
        block_size=16):
            
        super(G, self).__init__()
            
        self.first_in =nn.Sequential(*[
            nn.Conv2d(in_dim,res_channel,\
                kernel_size=9,stride=1,padding=4,groups=group),
        ])
        
        block_list =[ResidualBlock() for i in range(block_size)] 
        cn_1x1_list =[nn.Conv2d(res_channel,res_channel,kernel_size=1) for i in range(block_size)] 
        self.block_size =block_size
        
        for i,block in enumerate(block_list,):
            self.add_module('denseblock%d' % (i + 1), block)
        
        for i,cn11 in enumerate(cn_1x1_list,):
            self.add_module('denseblock_out%d' % (i + 1), cn11)
        
        
        self.upsample= nn.Sequential(*[
            nn.Conv2d(res_channel,res_channel*4,3,1,1),
            nn.PixelShuffle(2),
            nn.Conv2d(res_channel,res_channel*4,3,1,1),
            nn.PixelShuffle(2),
            nn.Conv2d(res_channel,3,3,1,1),

        ])
         
        
    def forward(self,x):
        o =x 
        o= self.first_in(o)
        
        tmp_list =[]
        
        for i  in  range(self.block_size):
            o_pre=o 
            o= getattr(self,"denseblock%d"% (i + 1) )(o)
            o1= getattr(self,"denseblock_out%d"% (i + 1) )(o)
            tmp_list.append(o1)
#             print ("pre:",o_pre.shape , "after:",o.shape, "tmp:",o1.shape)
        
        for o_1x1 in tmp_list:
            o=o+o_1x1
#         print ("element wise ",o.shape )
        out = self.upsample(o)
        
        return out 
    
# class D_vgg(nn.Module):
#     def __init__(self,input_width=296):
#         super(D, self).__init__()
#          
#         self.first_in=nn.Sequential(*[
#          nn.Conv2d(3,64,3,1,1),
#          nn.LeakyReLU(),
#         ])
#          
#          
#         m_list=[]
#         block_list = [64,128,128,256,256,512,512]
#         c_pre = 64 
#         padding = 1 
#         stride = 2
#         kernel_size=3 
#         for c_last in  block_list:
#             m_list.extend([
#                 nn.Conv2d(c_pre,c_last,kernel_size,1,1),
#                 nn.LeakyReLU(),
#                 nn.BatchNorm2d(c_last),
#  
#                 nn.Conv2d(c_last,c_last,kernel_size,stride,padding),
#                 nn.LeakyReLU(),
#                 nn.BatchNorm2d(c_last),
#             ])
#             c_pre=c_last 
#          
#         self.body = nn.Sequential(*m_list)
#          
#         #padding =1 ,
#         #k_size =3 
#         #strid =2 
#         downsize_fn =lambda x: int(x+2*padding -(kernel_size-1) -1 )/stride +1 
#         for _ in range(len(block_list)):
#             input_width=downsize_fn(input_width)
#          
#         self.last_fc_channel=int(input_width)*block_list[-1]*int(input_width)
#         self.out = nn.Sequential(*[
#             nn.Linear(self.last_fc_channel,1024),
#             nn.LeakyReLU(),
#             nn.Linear(1024,1),
#             nn.Sigmoid(),
#         ])
#  
#     def forward(self,x):
#         x1= self.first_in(x)
#         x2= self.body(x1)
#          
#         x2= x2.view(x2.shape[0] , -1 )
#         out =self.out(x2)
#         return out 
#         
class D(nn.Module):
    def __init__(self,input_c = 3 , input_width=296):
        super(D, self).__init__()
        
        self.first_in=nn.Sequential(*[
         nn.Conv2d(input_c,64,3,1,1),
         nn.LeakyReLU(),
        ])
        
        
        m_list=[]
#         block_list = [64,128,128,256,256,512,512]
        block_list = [64,128,256,512]

        c_pre = 64 
        padding = 1 
        stride = 2
        kernel_size=3 
        for c_last in  block_list:
            m_list+=[
                nn.Conv2d(c_pre,c_last,kernel_size,1,1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(c_last),
                
                nn.Conv2d(c_last,c_last,kernel_size,stride,padding),
                nn.LeakyReLU(),
                nn.BatchNorm2d(c_last),
                
            ]
            c_pre=c_last 
        self.body = nn.Sequential(*m_list)
        
        #padding =1 ,
        #k_size =3 
        #strid =2 
        downsize_fn =lambda x: int(x+2*padding -(kernel_size-1) -1 )/stride +1 
#         downsize_fn =lambda x: int(x+2*padding -(kernel_size-1) -1 )/stride +1 
        for _ in range(len(block_list)):
            input_width=downsize_fn(input_width)
        
        self.last_fc_channel=int(input_width)*block_list[-1]*int(input_width)
        self.out = nn.Sequential(*[
            nn.Linear(self.last_fc_channel,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,1),
            nn.Sigmoid(),
        ])

    def forward(self,x):
        x1= self.first_in(x)
        x2= self.body(x1)
        
        x2= x2.view(x2.shape[0] , -1 )
        out =self.out(x2)
        return out 
    
    
class vgg19_withoutbn_customefinetune(nn.Module):
    def __init__(self,pre_trained=True):
        super(vgg19_withoutbn_customefinetune, self).__init__()

#         vgg19_office = torchvision.models.vgg.vgg19(pretrained=pre_trained)
        vgg19_office = torchvision.models.vgg.vgg19(pretrained=pre_trained)
        o_features = vgg19_office.features 
        self.features=    nn.Sequential(*list(o_features.children())[:-1]) 
        
    def forward(self,x):
        out = self.features(x)
        return out 
        
if __name__=="__main__":
    x=torch.randn(1,3,296,296)
    cd = D(input_c=3,input_width=296)
    out_c = cd(x)
    
    print (out_c.shape ,"d_out ","in:",x.shape)
    exit()
    
    
    m = vgg19_withoutbn_customefinetune()
    out = m(x)
    print (m,out.shape,"vgg")
#     out = torch.randn(1, 512, 18, 18)
    d=D(input_c=512,input_width=18)
    print (d)
    print (d,"D 512...")
    
    d_out = d(out )
    print (d_out.shape, "d_out ")
#     model = G()
#     print (model) 
#     x= torch.randn(1,3,74,74)
#     x1= model(x)
#     print (x1.shape ,"<---",x.shape)
#         
        
        
        
        
#     x= torch.randn(1,3,296,296)
#     x= torch.randn(1,512,9,9)
#     model = D(input_c=512)
#     print (model) 
#     x1= model(x)
#     print (x1.shape ,"<---",x.shape)
        
