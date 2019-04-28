'''
Created on Apr 18, 2019

@author: gjpicker
'''
import torch 
import torch.nn  as nn
import torchvision

from carn import carn as carn_1
from carn import carn_m as carn_2

import functools


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
         
class G1(carn_1.Net):
    def __init__(self,in_dim=3,res_channel=64,group=1,\
        block_size=16):
            
        super(G1, self).__init__(scale=0,multi_scale=True,group=1)
         
class G2(carn_2.Net):
    def __init__(self,in_dim=3,res_channel=64,group=1,\
        block_size=16):
            
        super(G2, self).__init__(scale=0,multi_scale=True,group=4)
    
    
    
   
        
class G(nn.Module):
    def __init__(self,in_dim=3,res_channel=64,group=1,\
        block_size=16):
            
        super(G, self).__init__()
            
        self.first_in =nn.Sequential(*[
            nn.Conv2d(in_dim,res_channel,\
                kernel_size=9,stride=1,padding=4,groups=group),
        ])
        
        block_list =[ResidualBlock(in_c_and_out_c=res_channel) for i in range(block_size)] 
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
        
        
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

if __name__=="__main__":

    from networks import Vgg16 as tmp_Vgg16

    import unittest 
    import numpy as np 
    
    class T(unittest.TestCase):
#         def test_G(self):
#             model1 = G1()
#             model2 = G2()
#             model3 = G(res_channel=128)
#             model4 = G(res_channel=64)
#             x= torch.randn(1,3,74,74)
#             y1= model1(x)
#             y2= model2(x)
#             y3= model3(x)
#             y4= model4(x)
#             self.assertTrue(y1.shape==(1,3,296,296) , y1.shape)
#             self.assertTrue(y2.shape==(1,3,296,296) , y2.shape)
#             self.assertTrue(y3.shape==(1,3,296,296) , y3.shape)
#             self.assertTrue(y4.shape==(1,3,296,296) , y4.shape)
# 
#         def test_D(self):
#             x=torch.randn(1,3,296,296)
#             img_D = D(input_c=3,input_width=296)
#             out_c = img_D(x)
#             self.assertTrue(out_c.shape==(1,1),out_c.shape)
#             #print (out_c.shape ,"d_out ","in:",x.shape)
#             m = vgg19_withoutbn_customefinetune()
#             x_f = m(x)
#             print ("vgg_f " , x_f.shape)
# 
#             d=D(input_c=512,input_width=18)
#             d_out = d(x_f )
#             print (d_out.shape, "d_out ")
#         
#         
#         def test_D1D2(self):
#             d1= NLayerDiscriminator(input_nc=3)
#             print ("layer....",d1)
#             x1= torch.randn(1,3,296,296)
#             y1= d1(x1)
#             self.assertTrue(y1.size(1)==1,"expect channel==1")
#             print (x1.shape,y1.shape)
#         
#             x2= torch.randn(1,64,18,18)
#             d2= PixelDiscriminator(input_nc=64)
#             print (d2)
#             y2= d2(x2)
#             self.assertTrue(y2.size(1)==1,"expect channel==1")
#             print (x2.shape,y2.shape)
#         

        def test_vgg_and_style_classfiy_and_D(self):
            vgg_1 = vgg19_withoutbn_customefinetune()
            vgg_11 = vgg19_withoutbn_customefinetune()
            
            print (vgg_1)
            x1= torch.randn(1,3,64,64)
            with torch.no_grad():
                y1= vgg_1(x1)
                y11= vgg_11(x1)
            
            self.assertTrue(np.allclose(y11.numpy(),y1.numpy() ) , "expect load_state_dict in __init__")
            
            
            vgg_2 = tmp_Vgg16()
            with torch.no_grad():
                y2= vgg_2(x1)
                print (y2.shape)


        
    unittest.main()
        
        
#     x= torch.randn(1,3,296,296)
#     x= torch.randn(1,512,9,9)
#     model = D(input_c=512)
#     print (model) 
#     x1= model(x)
#     print (x1.shape ,"<---",x.shape)
        
