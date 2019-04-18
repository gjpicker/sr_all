'''
Created on Apr 18, 2019

@author: gjpicker
'''
from __future__ import absolute_import 
# from . import model as model  
import model as model  

import torch 
import torch.nn as nn 
import torch.optim as optim

import itertools

import time 
##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class Treainer(object):
    def __init__(self,opt=None,train_dt =None,valid_dt=None):
        self.device =torch.device("cuda")
        print ("ini...")

        tm_list=[]
        start_ = time.time() 
        
        self.G= model.G() 
        print ("create G")
        
        end_ = time.time() 
        tm_list .append(end_-start_)
        print (tm_list,"G")
        start_ = time.time()
        
        self.D_vgg= model. D(input_c=512,input_width=18) 
        end_ = time.time() 
        tm_list .append(end_-start_)
        start_ = time.time()
        print (tm_list,"d1")
        
        self.D = model.D()
        end_ = time.time() 
        tm_list .append(end_-start_)
        start_ = time.time()
        print (tm_list,"d2")
        
        print (self.G, self.D ,self.D_vgg )
        
        self.vgg = model. vgg19_withoutbn_customefinetune()
        end_ = time.time() 
        tm_list .append(end_-start_)
        start_ = time.time()
        print (tm_list,"d3")

        model.init_weights(self.D)
        model.init_weights(self.D_vgg)
        model.init_weights(self.G)
        
        self.vgg= self.vgg.to(self.device)
        self.D= self.D.to(self.device)
        self.D_vgg= self.D_vgg.to(self.device)
        self.G= self.G.to(self.device)

        print (tm_list,"all _create")
        self.optim_G= optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()),\
         lr=opt.lr, betas=opt.betas, weight_decay=0.0)
         
        self.optim_D=optim.SGD( filter(lambda p: p.requires_grad, \
            itertools.chain(self.D_vgg.parameters(),self.D.parameters() ) ),\
            lr=opt.lr,
         )
        print ("create dataset ")
        self.opt = opt 
        
        self.dt_train = train_dt 
        self.dt_valid = valid_dt


        self.critic_pixel = torch.nn.MSELoss().to(self.device)
        self.gan_loss = GANLoss(gan_mode="lsgan").to(self.device)
        print ("init ....")
    
    def run(self):
        
        for epoch in range(self.opt.epoches):
            for input_lr ,input_hr , _  in self.dt_train :
                self. input_lr = input_lr .to(self.device)
                self. input_hr = input_hr .to(self.device)
                
                self.forward()
                self.optim_G .zero_grad ()
                self.g_loss()
                self.optim_G.step()
                
                self.optim_D .zero_grad ()
                self.d_loss()
                self.optim_D.step()
                
                
    def forward(self,):
        self.output_hr = self.G(self.input_lr)
#         self.input_hr 
        pass 
        
        
    
    def g_loss (self,):
        d_fake = self.D(self.output_hr)
        
        self.loss_gan_g = self.gan_loss (d_fake,True )
        
        ## pixel 
        self.loss_p = self.critic_pixel (self.output_hr , self.input_hr )

        self.loss_g = self.loss_gan_g + self.loss_p

        self.loss_g.backward()
        
        print ("loss_g",self.loss_g.item() )
        pass 
        
    def d_loss (self,):
        d_fake = self.D(self.output_hr.detach())
        d_real = self.D(self.input_hr)
        
        with torch.no_grad():
            x_f_fake= self.vgg(self.output_hr.detach()) 
            x_f_real= self.vgg(self.input_hr) 
            
        vgg_d_fake = self.D_vgg(x_f_fake)
        vgg_d_real = self.D_vgg(x_f_real)
        
        
        self.loss_d_fake = self.gan_loss (d_fake,False )
        self.loss_d_real = self.gan_loss (d_real,True )
        
#         self.loss_d_f_fake = self.gan_loss (vgg_d_fake,False )
#         self.loss_d_f_real = self.gan_loss (vgg_d_real,True )
        
        self.loss_d_f_fake = 0
        self.loss_d_f_real = 0
        
        
        total_d_gan = self.loss_d_fake+ self.loss_d_real 
        total_d_f_gan = \
            self.loss_d_f_fake+self.loss_d_f_real
            
        loss_d =  total_d_gan +   self.opt.lambda_r * total_d_f_gan 
        print ("loss_d",loss_d.item() )
        
        loss_d.backward()
        
    
if __name__=="__main__":
    import torch.utils.data  as dt 
    class dt_c(dt.Dataset):
        def __init__(self):
            pass
        def __getitem__(self, index):
            return (
            torch.randn(3,74,74),
            torch.randn(3,296,296)
            )
        def __len__(self):
            return 1000
            
    class config :
        lr =0.0001
        betas =( 0.9 ,0.99) 
        lambda_r= 0.1
        epoches =10
        
    dl_c =dt.DataLoader(dt_c() ,batch_size=1)
    
    tr_l = Treainer(opt=config() ,train_dt = dl_c )
    tr_l.run()
    
