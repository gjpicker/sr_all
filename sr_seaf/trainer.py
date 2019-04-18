'''
Created on Apr 18, 2019

@author: gjpicker
'''
from __future__ import absolute_import 
from collections import OrderedDict
# from . import model as model  
import model as model  

import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler

import itertools

import time 

#=====START: ADDED FOR DISTRIBUTED======
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from torch.utils.data.distributed import DistributedSampler
#=====END:   ADDED FOR DISTRIBUTED======


import torch.utils.data as t_data

from utils.visualizer import Visualizer
import utils.util as util


def get_scheduler(optimizer, opt):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 2 - opt.niter) / float(opt.niter_decay + 1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler

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
    def __init__(self,opt=None,train_dt =None,valid_dt=None,dis_list=[]):
        self.device =torch.device("cuda")

        self.opt = opt 

        self.visualizer = Visualizer(opt)

        num_gpus = torch.cuda.device_count()
        #dis_list[1]
        print (dis_list)
#torch.cuda.device_count()
        self.rank =dis_list[0]
        print (self.rank)

        #=====START: ADDED FOR DISTRIBUTED======
        if num_gpus > 1:
            #init_distributed(rank, num_gpus, group_name, **dist_config)
            dist_config= dis_list[3]
            init_distributed(dis_list[0], dis_list[1], dis_list[2], **dist_config)
        #=====END:   ADDED FOR DISTRIBUTED======


        
        self.G= model.G() 
        
        self.D_vgg= model. D(input_c=512,input_width=18) 
        
        self.D = model.D()
        
        
        self.vgg = model. vgg19_withoutbn_customefinetune()
        for p in self.vgg.parameters():
            p.requires_grad = True

        print (self.G,self.D)

        model.init_weights(self.D)
        model.init_weights(self.D_vgg)
        model.init_weights(self.G)
        
        self.vgg= self.vgg.to(self.device)
        self.D= self.D.to(self.device)
        self.D_vgg= self.D_vgg.to(self.device)
        self.G= self.G.to(self.device)

        #=====START: ADDED FOR DISTRIBUTED======
        if num_gpus > 1:
            #self.vgg = apply_gradient_allreduce(self.vgg)
            self.D_vgg = apply_gradient_allreduce(self.D_vgg)
            self.D = apply_gradient_allreduce(self.D)
            self.G = apply_gradient_allreduce(self.G)
            
        #=====END:   ADDED FOR DISTRIBUTED======



        self.optim_G= optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()),\
         lr=opt.lr, betas=opt.betas, weight_decay=0.0)
         
        self.optim_D=optim.SGD( filter(lambda p: p.requires_grad, \
            itertools.chain(self.D_vgg.parameters(),self.D.parameters() ) ),\
            lr=opt.lr,
         )
        print ("create dataset ")
        self.schedulers = []
        self.optimizers = []
        self.optimizers.append(self.optim_G)
        self.optimizers.append(self.optim_D)
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, self.opt))




        
        # =====START: ADDED FOR DISTRIBUTED======
        train_sampler = DistributedSampler(train_dt) if num_gpus > 1 else None
        # =====END:   ADDED FOR DISTRIBUTED======

        kw ={"pin_memory":True , "num_workers":opt.num_workers } if torch.cuda.is_available() else {}
        dl_c =t_data.DataLoader(train_dt ,batch_size=opt.batch_size, **kw ,\
             sampler=train_sampler , drop_last=True)


        self.dt_train = dl_c
#train_dt 
        #self.dt_valid = valid_dt


        self.critic_pixel = torch.nn.MSELoss().to(self.device)
        self.gan_loss = GANLoss(gan_mode="lsgan").to(self.device)
        print ("init ....")
    
    def run(self):
        
        total_steps=0 
        opt= self.opt 
        dataset_size= len(self.dt_train) * opt.batch_size 

        for epoch in range(self.opt.epoches_warm):
            epoch_start_time = time.time()
            epoch_iter = 0

            for input_lr ,input_hr , cubic_hr  in self.dt_train :
                iter_start_time = time.time()

                self. input_lr = input_lr .to(self.device)
                self. input_hr = input_hr .to(self.device)
                self. input_cubic_hr = cubic_hr

                self.forward()
                self.optim_G .zero_grad ()
                self.warm_loss()
                self.optim_G.step()


                self.visualizer.reset()
                total_steps += opt.batch_size
                epoch_iter += opt.batch_size

                if self.rank !=0 :
                   continue
                if total_steps % opt.display_freq == 0:
                    save_result = total_steps % opt.update_html_freq == 0
                    self.visualizer.display_current_results(self.get_current_visuals(), opt.epoches, save_result)

                if total_steps % opt.print_freq == 0:
                    errors = self.get_current_errors()
                    t = (time.time() - iter_start_time) / opt.batch_size
                    self.visualizer.print_current_errors(epoch , epoch_iter, errors, t)
                    if opt.display_id > 0:
                        self.visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size , opt, errors)







        self.loss_w_g=torch.tensor(0)


        for epoch in range(self.opt.epoches_warm , self.opt.epoches_warm +self.opt.epoches):
            epoch_start_time = time.time()
            epoch_iter = 0

            for input_lr ,input_hr , cubic_hr  in self.dt_train :
                iter_start_time = time.time()

                self. input_lr = input_lr .to(self.device)
                self. input_hr = input_hr .to(self.device)
                self. input_cubic_hr = cubic_hr
                
                self.forward()
                self.optim_G .zero_grad ()
                self.g_loss()
                self.optim_G.step()
                
                self.optim_D .zero_grad ()
                self.d_loss()
                self.optim_D.step()


                self.visualizer.reset()
                total_steps += opt.batch_size
                epoch_iter += opt.batch_size

                if self.rank !=0 :
                   continue
                if total_steps % opt.display_freq == 0:
                    save_result = total_steps % opt.update_html_freq == 0
                    self.visualizer.display_current_results(self.get_current_visuals(), opt.epoches, save_result)

                if total_steps % opt.print_freq == 0:
                    errors = self.get_current_errors()
                    t = (time.time() - iter_start_time) / opt.batch_size
                    self.visualizer.print_current_errors(epoch , epoch_iter, errors, t)
                    if opt.display_id > 0:
                        self.visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size , opt, errors)


                
                
    def forward(self,):
        self.output_hr = self.G(self.input_lr)
#         self.input_hr 
        pass 
        
    def warm_loss(self):

        #x_f_fake= self.vgg(self.output_hr)
        #x_f_real= self.vgg(self.input_hr)

        ## pixel 
        #self.loss_G_p = self.critic_pixel (x_f_fake , x_f_real ) 
        self.loss_w_g = self.critic_pixel(self.output_hr , self.input_hr )

        self.loss_w_g.backward()


        
    
    def g_loss (self,):
        
        x_f_fake= self.vgg(self.output_hr) 

        x_f_real= self.vgg(self.input_hr) 

        
        d_fake = self.D(self.output_hr)
        fd_fake = self.D_vgg(x_f_fake)

        self.loss_G_g = self.gan_loss (d_fake,True )
        self.loss_G_fg = self.gan_loss (fd_fake,True )

        ## pixel 
        #self.loss_G_p = self.critic_pixel (self.output_hr , self.input_hr )
        self.loss_G_p = self.critic_pixel (x_f_fake,x_f_real )

        self.loss_g = self.opt.lambda_r *( self.loss_G_g + self.loss_G_fg) + self.loss_G_p

        self.loss_g.backward()
        
        #print ("loss_g",self.loss_g.item() )
        pass 
        
    def d_loss (self,):
        d_fake = self.D(self.output_hr.detach())
        d_real = self.D(self.input_hr)
        
        with torch.no_grad():
            x_f_fake= self.vgg(self.output_hr.detach()) 
            x_f_real= self.vgg(self.input_hr) 
            
        vgg_d_fake = self.D_vgg(x_f_fake)
        vgg_d_real = self.D_vgg(x_f_real)
        
        
        self.loss_D_f = self.gan_loss (d_fake,False )
        self.loss_D_r = self.gan_loss (d_real,True )
        
        self.loss_Df_f = self.gan_loss (vgg_d_fake,False )
        self.loss_Df_r = self.gan_loss (vgg_d_real,True )
        
        #self.loss_d_f_fake = 0
        #self.loss_d_f_real = 0
        
        
            
        loss_d =self.loss_D_f+ self.loss_D_r +self.loss_Df_f+\
            self.loss_Df_r
        #print ("loss_d",loss_d.item() )
        
        loss_d.backward()
        
    def get_current_errors(self):
        return OrderedDict([('G_p', self.loss_G_p.item() if hasattr(self,"loss_G_p") else 0 ),
                            ('G_fg', self.loss_G_fg.item()  if hasattr(self,"loss_G_fg") else 0 ),
                            ('G_g', self.loss_G_g.item()  if hasattr(self,"loss_G_g") else 0 ),
                            ('D_f_real', self.loss_Df_r.item()  if hasattr(self,"loss_Df_r") else 0 ),
                            ('D_f_fake', self.loss_Df_f.item()  if hasattr(self,"loss_Df_f") else 0 ),
                            ('D_real', self.loss_D_r.item()  if hasattr(self,"loss_D_r") else 0 ) ,
                            ('D_fake', self.loss_D_f.item()  if hasattr(self,"loss_D_f") else 0 ),
                            ('warm_p', self.loss_w_g.item()  if hasattr(self,"loss_w_g") else 0 ),
                            ])

    def get_current_visuals(self):
        input = util.tensor2im(self.input_cubic_hr)
        target = util.tensor2im(self.input_hr)
        fake = util.tensor2im(self.output_hr.detach() )
        return OrderedDict([('input', input),  ('fake', fake), ('target', target)])

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
    
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
    
