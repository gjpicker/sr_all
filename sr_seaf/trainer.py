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

import numpy as np 

from networks import *
#=====START: ADDED FOR DISTRIBUTED======
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from torch.utils.data.distributed import DistributedSampler
#=====END:   ADDED FOR DISTRIBUTED======


import torch.utils.data as t_data

from utils.visualizer import Visualizer
import utils.util as util
from utils import image_quality



class Treainer(object):
    def __init__(self,opt=None,train_dt =None,train_dt_warm=None,dis_list=[] , val_dt_warm=None):
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


        if opt.ge_net =="srfeat":
            self.netG= model.G() 
        elif opt.ge_net =="carn":
            self.netG= model.G1()
        elif opt.ge_net =="carnm":
            self.netG= model.G2()
        else :
            raise Exception("unknow ")
                
        
        self.netD_vgg= model. D(input_c=512,input_width=18) 
        
        self.netD = model.D()
        
        
        if opt.vgg_type =="style":
            self.vgg = load_vgg16(opt.vgg_model_path + '/models')
        elif opt.vgg_type =="classify" :
            self.vgg = model. vgg19_withoutbn_customefinetune()
            
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

#         for p in self.vgg.parameters():
#             p.requires_grad = False


        init_weights(self.netD,init_type=opt.init)
        init_weights(self.netD_vgg,init_type=opt.init)
        init_weights(self.netG,init_type=opt.init)
        
        self.vgg= self.vgg.to(self.device)
        self.netD= self.netD.to(self.device)
        self.netD_vgg= self.netD_vgg.to(self.device)
        self.netG= self.netG.to(self.device)

        #=====START: ADDED FOR DISTRIBUTED======
        if num_gpus > 1:
            #self.vgg = apply_gradient_allreduce(self.vgg)
            self.netD_vgg = apply_gradient_allreduce(self.netD_vgg)
            self.netD = apply_gradient_allreduce(self.netD)
            self.netG = apply_gradient_allreduce(self.netG)
            
        #=====END:   ADDED FOR DISTRIBUTED======

        print (opt)

        self.optim_G_warm= torch. optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()),\
         lr=opt.warm_opt.lr, betas=opt.warm_opt.betas, weight_decay=0.0)
         
        self.optim_G = self.optim_G_warm
#        self.optim_G= torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()),\
#         lr=opt.gen.lr, betas=opt.gen.betas, weight_decay=0.0)
         
         
        if opt.dis.optim =="sgd":
            self.optim_D= torch.optim.SGD( filter(lambda p: p.requires_grad, \
                itertools.chain(self.netD_vgg.parameters(),self.netD.parameters() ) ),\
                lr=opt.dis.lr,
             )
        elif opt.dis.optim =="adam":
            self.optim_D= torch.optim.Adam( filter(lambda p: p.requires_grad, \
                itertools.chain(self.netD_vgg.parameters(),self.netD.parameters() ) ),\
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
        val_sampler_warm = DistributedSampler(val_dt_warm) if num_gpus > 1 else None
        # =====END:   ADDED FOR DISTRIBUTED======

        kw ={"pin_memory":True , "num_workers":8 } if torch.cuda.is_available() else {}
        dl_c =t_data.DataLoader(train_dt ,batch_size=opt.batch_size,\
             sampler=train_sampler , drop_last=True, **kw )
        dl_c_warm =t_data.DataLoader(train_dt_warm ,batch_size=opt.batch_size if not hasattr(opt,"batch_size_warm" ) else opt.batch_size_warm,  
             sampler=train_sampler_warm , drop_last=True  ,**kw)
             

        dl_val_warm =t_data.DataLoader(val_dt_warm ,batch_size=opt.batch_size if not hasattr(opt,"batch_size_warm" ) else opt.batch_size_warm,  
             sampler=val_sampler_warm , drop_last=True  ,**kw)



        self.dt_train = dl_c
        self.dt_train_warm = dl_c_warm
        self.dt_val_warm =  dl_val_warm


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
    

        self.save_dir = os.path.dirname( self.visualizer. log_name )

        
    def run(self):

        self._run_warm()

    def _validate_(self):
        with torch.no_grad():
            print ("val ,"*8,"warm start...",len(self.dt_val_warm))
            iter_start_time = time.time()
            ssim = []
            batch_loss = []
            psnr = []

            cub_ssim = []
            cub_batch_loss = []
            cub_psnr = []


            save_image_list_1 = []

            for ii,data   in  tqdm.tqdm( enumerate(self.dt_val_warm) ):
                if len(data)>3:
                    input_lr ,input_hr , cubic_hr,_,_ =data
                else :
                    input_lr ,input_hr , cubic_hr =data

                self. input_lr = input_lr .to(self.device)
                self. input_hr = input_hr .to(self.device)
                self. input_cubic_hr = cubic_hr .to(self.device)

                self.forward()

                save_image_list_1.append(torch.cat( [self.input_cubic_hr ,\
                 self.output_hr ,\
                 self.input_hr ],dim=3)  )

                loss = self.critic_pixel(self.output_hr, self.input_hr)
                batch_loss.append(loss.item())
                ssim.append(image_quality.msssim(self.output_hr, self.input_hr).item())
                psnr.append(image_quality.psnr( self.output_hr, self.input_hr ).item())

                cub_loss = self.critic_pixel( self.input_cubic_hr , self.input_hr)
                cub_batch_loss.append(cub_loss.item())
                cub_ssim.append(image_quality.msssim(self.input_cubic_hr, self.input_hr).item())
                cub_psnr.append(image_quality.psnr( self.input_cubic_hr, self.input_hr ).item())

            np.random.shuffle(save_image_list_1)
            save_image_list= save_image_list_1[:8]
            save_image_list = util.tensor2im( torch.cat(save_image_list,dim=2) )
            save_image_list = OrderedDict([ ("cub_out_gt", save_image_list )] ) 
            self.visualizer.display_current_results(save_image_list,self.epoch, save_result=True, offset=20,title="val_imag")

            val_info = ( np.mean(batch_loss),np.mean(ssim),np.mean(psnr) ,np.mean(cub_batch_loss) , np.mean(cub_ssim) ,np.mean(cub_psnr))
            errors = dict(zip ( ("loss","ssim","psnr","cub_loss","cub_ssim","cub_psnr") , val_info ) ) 
            t = (time.time() - iter_start_time) 
            self.visualizer.print_current_errors(self.epoch , self.epoch, errors, t,log_name="loss_log_val.txt")
            self.visualizer.plot_current_errors(self.epoch , self.epoch,opt=None,errors=errors,display_id_offset=3,loss_name="val")

            return val_info



    def _run_warm(self):
        total_steps=0 
        opt= self.opt 
        dataset_size= len(self.dt_train_warm) * opt.batch_size 

        self.model_names = ["G"]

        best_loss= 10e5
        for epoch in range(self.opt.epoches_warm):
            self.epoch = epoch
#             epoch_start_time = time.time()
            val_loss = self._validate_()
            val_loss = val_loss[0]
            if best_loss  > val_loss:
                best_loss= val_loss
                self.save_networks("best")
            self.save_networks(epoch)

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

                if total_steps % opt.display_freq == 0:
                    save_result = total_steps % opt.update_html_freq == 0
                    self.visualizer.display_current_results(self.get_current_visuals(), opt.epoches_warm, save_result)

                if total_steps % opt.print_freq == 0:
                    errors = self.get_current_errors()
                    t = (time.time() - iter_start_time) / opt.batch_size
                    self.visualizer.print_current_errors(epoch , epoch_iter, errors, t)
                    if opt.display_id > 0:
                        self.visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size , opt, errors)

            
                if self.rank !=0 :
                    continue
            lr_warm,_=self.update_learning_rate(is_warm=True)
            self.visualizer.plot_current_lrs(epoch,0,opt=None,\
                errors=OrderedDict([('lr_warm_g',lr_warm),("lr_g",0),("lr_d",0)]) , loss_name="lr_warm" ,display_id_offset=1 )




    def _run_train(self):
        total_steps=0
        opt= self.opt

        self.model_names = ["G","D","D_vgg"]

        self.loss_w_g=torch.tensor(0)
        dataset_size= len(self.dt_train) * opt.batch_size 
        best_loss = 10e5

        for epoch in range(self.opt.epoches_warm , self.opt.epoches_warm +self.opt.epoches):
            self.epoch = epoch
#             epoch_start_time = time.time()
            epoch_iter = 0

            val_loss = self._validate_()
            val_loss = val_loss[0]
            if best_loss  > val_loss:
                best_loss= val_loss
                self.save_networks("best")
            self.save_networks(epoch)


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

                if total_steps % opt.display_freq == 0:
                    save_result = total_steps % opt.update_html_freq == 0
                    self.visualizer.display_current_results(self.get_current_visuals(), epoch, save_result)

                if total_steps % opt.print_freq == 0:
                    errors = self.get_current_errors()
                    t = (time.time() - iter_start_time) / opt.batch_size
                    self.visualizer.print_current_errors(epoch , epoch_iter, errors, t)
                    if opt.display_id > 0:
                        self.visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size , opt, errors)


                if self.rank !=0 :
                    continue
            lr_g,lr_d=self.update_learning_rate(is_warm=False)
            self.visualizer.plot_current_lrs(epoch,0,opt=None,\
                errors=OrderedDict([ ('lr_warm_g',0),("lr_g",lr_g),("lr_d",lr_d) ]) ,  loss_name="lr_warm"  ,display_id_offset=1)

                
                
    def forward(self,):
        self.output_hr = self.netG(self.input_lr)
#         self.input_hr 
        pass 
        
    def warm_loss(self):

        #x_f_fake= self.vgg(self.output_hr)
        #x_f_real= self.vgg(self.input_hr)

        ## pixel 
        #self.loss_G_p = self.critic_pixel (x_f_fake , x_f_real ) 
        self.loss_w_g = self.critic_pixel(self.output_hr , self.input_hr )

        self.loss_w_g.backward()

        if hasattr(self.opt.warm_opt, "clip"):
            nn.utils.clip_grad_norm(self.netG.parameters(), self.opt.warm_opt.clip)

        
    
    def g_loss (self,):
        #print (self.opt.gen,type(self.opt.gen),self.opt.gen.keys())
        vgg_r = self.opt.gen.lambda_vgg_input
        #g feature f  
        x_f_fake= self.vgg(vgg_r * self.output_hr) 
        
        #g .. f 
        d_fake = self.netD(self.output_hr)
        self.loss_G_g = self.gan_loss (d_fake,True )

        fd_fake = self.netD_vgg(x_f_fake)
        self.loss_G_fg = self.gan_loss (fd_fake,True )

        ## perception 
        x_f_real= self.vgg(vgg_r * self.input_hr) 
        self.loss_G_p = self.critic_pixel (x_f_fake,x_f_real )

        self.loss_g = self.opt.gen.lambda_vgg_loss *( self.loss_G_g + self.loss_G_fg) + self.loss_G_p

        self.loss_g.backward()
        
        
    def d_loss (self,):
        d_fake = self.netD(self.output_hr.detach())
        d_real = self.netD(self.input_hr)
        
        vgg_r = self.opt.gen .lambda_vgg_input
        x_f_fake= self.vgg(vgg_r* self.output_hr.detach()) 
        x_f_real= self.vgg(vgg_r* self.input_hr) 
        
        vgg_d_fake = self.netD_vgg(x_f_fake)
        vgg_d_real = self.netD_vgg(x_f_real)
        
        
        self.loss_D_f = self.gan_loss (d_fake,False )
        self.loss_D_r = self.gan_loss (d_real,True )
        
        self.loss_Df_f = self.gan_loss (vgg_d_fake,False )
        self.loss_Df_r = self.gan_loss (vgg_d_real,True )
        
        #self.loss_d_f_fake = 0
        #self.loss_d_f_real = 0
        if self.opt.gan_loss_fn =="wgangp":
            # train with gradient penalty
            gradient_penalty_vgg,_ = cal_gradient_penalty(netD=self.netD_vgg, real_data=x_f_real.data,\
                fake_data=x_f_fake.data,device=self.device)
            gradient_penalty_vgg.backward()
            
            gradient_penalty,_ = cal_gradient_penalty(netD=self.netD, real_data=self.input_hr.data, \
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
        
    def save_networks(self, epoch):
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if "parallel" in str(type(net))  and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

                net.to(self.device)



    def load_networks(self, epoch):
        """Load all the networks from the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        #for name in self.model_names:
        for name in ["G","D","D_vgg"]:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                if not os.path.isfile(load_path):
                    print ("***","fail find%s"(load_path))
                    continue
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    
