'''
Created on Apr 18, 2019

@author: gjpicker
'''
# -*- coding: utf-8 -*- 
from __future__ import absolute_import 
from collections import OrderedDict
# from . import model as model  

import model as model  

import torch 
import torch.nn as nn 
#import torch.optim as optim
import torch.optim
from torch.optim import lr_scheduler

import itertools

import time 
import math 
import tqdm

from networks import *
#=====START: ADDED FOR DISTRIBUTED======
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from torch.utils.data.distributed import DistributedSampler
#=====END:   ADDED FOR DISTRIBUTED======


import torch.utils.data as t_data

from utils.visualizer import Visualizer
import utils.util as util



class Treainer(object):
    def __init__(self,opt=None,train_dt =None,train_dt_warm=None,dis_list=[]):
        self.device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        
        if opt.vgg_type =="style":
            self.vgg = load_vgg16(opt.vgg_model_path + '/models')
        elif opt.vgg_type =="classify" :
            self.vgg = model. vgg19_withoutbn_customefinetune()
            
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

#         for p in self.vgg.parameters():
#             p.requires_grad = False


        init_weights(self.D,init_type=opt.init)
        init_weights(self.D_vgg,init_type=opt.init)
        init_weights(self.G,init_type=opt.init)
        
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

        print (opt)

        self.optim_G_warm= torch. optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()),\
         lr=opt.warm_opt.lr, betas=opt.warm_opt.betas, weight_decay=0.0)
         
        self.optim_G = self.optim_G_warm
#        self.optim_G= torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()),\
#         lr=opt.gen.lr, betas=opt.gen.betas, weight_decay=0.0)
         
         
        if opt.dis.optim =="sgd":
            self.optim_D= torch.optim.SGD( filter(lambda p: p.requires_grad, \
                itertools.chain(self.D_vgg.parameters(),self.D.parameters() ) ),\
                lr=opt.dis.lr,
             )
        elif opt.dis.optim =="adam":
            self.optim_D= torch.optim.Adam( filter(lambda p: p.requires_grad, \
                itertools.chain(self.D_vgg.parameters(),self.D.parameters() ) ),\
                lr=opt.dis.lr,betas=opt.dis.betas, weight_decay=0.0
             )
        else:
            raise Exception("unknown")
             
        
          
        print ("create schedule ")
        
        lr_sc_warm = get_scheduler(self.optim_G_warm,opt.warm_opt )
        lr_sc_G = get_scheduler(self.optim_G,opt.gen )
        lr_sc_D = get_scheduler(self.optim_D,opt.dis )
        
        
        self.schedulers = []
        self.schedulers_warm = []
        
        self.schedulers_warm.append(lr_sc_warm)
        self.schedulers.append(lr_sc_G)
        self.schedulers.append(lr_sc_D)
        
        
        # =====START: ADDED FOR DISTRIBUTED======
        train_sampler = DistributedSampler(train_dt) if num_gpus > 1 else None
        train_sampler_warm = DistributedSampler(train_dt_warm) if num_gpus > 1 else None
        # =====END:   ADDED FOR DISTRIBUTED======

        kw ={"pin_memory":True , "num_workers":8 } if torch.cuda.is_available() else {}
        dl_c =t_data.DataLoader(train_dt ,batch_size=opt.batch_size,\
             sampler=train_sampler , drop_last=True, **kw )
             
        dl_c_warm =t_data.DataLoader(train_dt_warm ,batch_size=opt.batch_size if not hasattr(opt,"batch_size_warm" ) else opt.batch_size_warm,  
             sampler=train_sampler_warm , drop_last=True  ,**kw)


        self.dt_train = dl_c
        self.dt_train_warm = dl_c_warm


        if opt.warm_opt.loss_fn=="mse":
            self.critic_pixel = torch.nn.MSELoss()
        elif opt.warm_opt.loss_fn=="l1":
            self.critic_pixel = torch.nn.L1Loss()
        elif opt.warm_opt.loss_fn=="smooth_l1":
            self.critic_pixel = torch.nn.SmoothL1Loss()
        else:
            raise Exception("unknown")

        self.critic_pixel=self.critic_pixel.to(self.device)
        
        self.gan_loss = GANLoss(gan_mode=opt.gan_loss_fn).to(self.device)
        print ("init ....")
    
    def run(self):
        
        total_steps=0 
        opt= self.opt 
        dataset_size= len(self.dt_train_warm) * opt.batch_size 

        for epoch in range(self.opt.epoches_warm):
#             epoch_start_time = time.time()
            epoch_iter = 0

            print ("warm start...",len(self.dt_train_warm))
            for ii,data   in  enumerate(self.dt_train_warm) :
                if len(data)>3:
                    input_lr ,input_hr , cubic_hr,_,_ =data 
                else :
                    input_lr ,input_hr , cubic_hr =data 
                
                iter_start_time = time.time()

                self. input_lr = input_lr .to(self.device)
                self. input_hr = input_hr .to(self.device)
                self. input_cubic_hr = cubic_hr

                self.forward()
                self.optim_G_warm .zero_grad ()
                self.warm_loss()
                self.optim_G_warm.step()


                self.visualizer.reset()
                total_steps += opt.batch_size
                epoch_iter += opt.batch_size

                if self.rank !=0 :
                    continue
                if total_steps % opt.display_freq == 0:
                    save_result = total_steps % opt.update_html_freq == 0
                    self.visualizer.display_current_results(self.get_current_visuals(), opt.epoches_warm, save_result)

                if total_steps % opt.print_freq == 0:
                    errors = self.get_current_errors()
                    t = (time.time() - iter_start_time) / opt.batch_size
                    self.visualizer.print_current_errors(epoch , epoch_iter, errors, t)
                    if opt.display_id > 0:
                        self.visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size , opt, errors)

            
            lr_warm,_=self.update_learning_rate(is_warm=True)
            self.visualizer.plot_current_lrs(epoch,0,opt=None,\
                errors=OrderedDict([('lr_warm_g',lr_warm),("lr_g",0),("lr_d",0)]) , loss_name="lr_warm" ,display_id_offset=1 )




        self.loss_w_g=torch.tensor(0)
        dataset_size= len(self.dt_train) * opt.batch_size 

        for epoch in range(self.opt.epoches_warm , self.opt.epoches_warm +self.opt.epoches):
#             epoch_start_time = time.time()
            epoch_iter = 0

            for data in self.dt_train :
                if len(data)>3:
                    input_lr ,input_hr , cubic_hr,_,_ =data 
                else :
                    input_lr ,input_hr , cubic_hr =data 
                
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
                    self.visualizer.display_current_results(self.get_current_visuals(), epoch, save_result)

                if total_steps % opt.print_freq == 0:
                    errors = self.get_current_errors()
                    t = (time.time() - iter_start_time) / opt.batch_size
                    self.visualizer.print_current_errors(epoch , epoch_iter, errors, t)
                    if opt.display_id > 0:
                        self.visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size , opt, errors)

            lr_g,lr_d=self.update_learning_rate(is_warm=False)
            self.visualizer.plot_current_lrs(epoch,0,opt=None,\
                errors=OrderedDict([ ('lr_warm_g',0),("lr_g",lr_g),("lr_d",lr_d) ]) ,  loss_name="lr_warm"  ,display_id_offset=1)

                
                
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
        #print (self.opt.gen,type(self.opt.gen),self.opt.gen.keys())
        vgg_r = self.opt.gen.lambda_vgg_input
        #g feature f  
        x_f_fake= self.vgg(vgg_r * self.output_hr) 
        
        #g .. f 
        d_fake = self.D(self.output_hr)
        self.loss_G_g = self.gan_loss (d_fake,True )

        fd_fake = self.D_vgg(x_f_fake)
        self.loss_G_fg = self.gan_loss (fd_fake,True )

        ## perception 
        x_f_real= self.vgg(vgg_r * self.input_hr) 
        self.loss_G_p = self.critic_pixel (x_f_fake,x_f_real )

        self.loss_g = self.opt.gen.lambda_vgg_loss *( self.loss_G_g + self.loss_G_fg) + self.loss_G_p

        self.loss_g.backward()
        
        
    def d_loss (self,):
        d_fake = self.D(self.output_hr.detach())
        d_real = self.D(self.input_hr)
        
        vgg_r = self.opt.gen .lambda_vgg_input
        x_f_fake= self.vgg(vgg_r* self.output_hr.detach()) 
        x_f_real= self.vgg(vgg_r* self.input_hr) 
        
        vgg_d_fake = self.D_vgg(x_f_fake)
        vgg_d_real = self.D_vgg(x_f_real)
        
        
        self.loss_D_f = self.gan_loss (d_fake,False )
        self.loss_D_r = self.gan_loss (d_real,True )
        
        self.loss_Df_f = self.gan_loss (vgg_d_fake,False )
        self.loss_Df_r = self.gan_loss (vgg_d_real,True )
        
        #self.loss_d_f_fake = 0
        #self.loss_d_f_real = 0
        if self.opt.gan_loss_fn =="wgangp":
            # train with gradient penalty
            gradient_penalty_vgg,_ = cal_gradient_penalty(netD=self.D_vgg, real_data=x_f_real.data,\
                fake_data=x_f_fake.data,device=self.device)
            gradient_penalty_vgg.backward()
            
            gradient_penalty,_ = cal_gradient_penalty(netD=self.D, real_data=self.input_hr.data, \
                fake_data = self.output_hr.data,  device=self.device)
            gradient_penalty.backward()

        
            
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

    def update_learning_rate(self,is_warm =True ):
        if is_warm :
            for scheduler in self.schedulers_warm:
                scheduler.step(self.loss_w_g)
            
            lr = self.optim_G_warm.param_groups[0]['lr']
            return (lr,0)
        else:
            for scheduler in self.schedulers:
                scheduler.step()

            lr_g = self.optim_G.param_groups[0]['lr']
            lr_d = self.optim_G.param_groups[0]['lr']
            return (lr_g,lr_d)
        
    
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
    
