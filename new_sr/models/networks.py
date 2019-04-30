# import torch
# -*- coding: utf-8 -*-
# import torch.nn as nn
from __future__ import absolute_import 
from torch.nn import init
import functools
from torch.optim import lr_scheduler


import os 
import torch 
import torch.nn.functional as F 
import torchvision
import torch.nn as nn 

###############################################################################

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
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


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


def define_G(ge_net_str, g_path="",opt={}):
    '''
    ge_net_str==opt.gen.net_type ,
    g_path chpt 
    '''
#     assert ge_net_str in ["carnm" ,"carn" ,"carn_gan" ,"carn_ganm"] , "but expect "+ge_net_str



    #ge_net_str= opt.gen.net_type
    netG = None 
    print (ge_net_str,"----"*4)
    if ge_net_str is None or ge_net_str=="":
        raise Exception("unknown G")
    if ge_net_str =="srfeat":
        raise Exception("not support multi-scale")
    elif ge_net_str =="carn":
        kwargs = {
        "group": 1,
        "multi_scale": True,
        "scale": 0,
        }
        from .nets.carn.carn import Net as g1_net 
        netG= g1_net(**kwargs)
    elif ge_net_str =="carnm":
        from .nets.carn.carn_m import Net as g2_net 
        kwargs = {
        "group": 4,
        "multi_scale": True,
        "scale": 0,
        }
        netG= g2_net(**kwargs)
        print (netG)
    elif "carn_gan" in  ge_net_str:
        from .nets.pcarn.pcarn import Net as g1_net 
        if ge_net_str=="carn_gan":
            kwargs = {
                "num_channels": 64,
                "groups": 1,
                "mobile": False,
                "scale": 0,
            }
            netG = g1_net(**kwargs)
        elif ge_net_str=="carn_ganm":
            kwargs = {
            "num_channels": 64,
            "groups": 4,
            "mobile": True,
            "scale": 0,
            }
            netG = g1_net(**kwargs)
    elif ge_net_str == 'DBPN':
        from .nets.srfbn_cvpr19.dbpn_arch import DBPN
        netG = DBPN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                         num_features=opt['num_features'], bp_stages=opt['num_blocks'],
                         upscale_factor=opt['scale'])

    elif ge_net_str == 'D-DBPN':
        from .nets.srfbn_cvpr19.dbpn_arch import D_DBPN
        netG = D_DBPN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                           num_features=opt['num_features'], bp_stages=opt['num_blocks'],
                           upscale_factor=opt['scale'])

    elif ge_net_str.find('SRFBN') >= 0:
        from .nets.srfbn_cvpr19.srfbn_arch import SRFBN
        netG = SRFBN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                                  num_features=opt['num_features'], num_steps=opt['num_steps'], num_groups=opt['num_groups'],
                                  upscale_factor=opt['scale'])

    elif ge_net_str.find('RDN') >= 0:
        from .nets.srfbn_cvpr19.rdn_arch import RDN
        netG = RDN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                             num_features=opt['num_features'], num_blocks = opt['num_blocks'], num_layers = opt['num_layers'],
                             upscale_factor=opt['scale'])

    elif ge_net_str.find('EDSR') >= 0:
        from .nets.srfbn_cvpr19.edsr_arch import EDSR
        netG = EDSR(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                             num_features=opt['num_features'], num_blocks = opt['num_blocks'], res_scale=opt['res_scale'],
                             upscale_factor=opt['scale'])

    else:
        raise NotImplementedError("Network [%s] is not recognized." % ge_net_str)

    
    
    
    if os.path.isfile(g_path):
        if not torch.cuda.is_available():
            netG.load_state_dict(\
                torch.load(g_path, map_location=lambda storage, loc: storage) )
        else :
            netG.load_state_dict(\
                torch.load(g_path))
    else :
        print ("===="*8,"fail load pretrained")
        
    return netG
    
    
def define_D_vgg(opt,input_nc):
    return define_D(opt,input_nc)
    pass 
def define_D(d_net_type_str="", input_nc=3,num_d=0):
    '''
    dis : 
        net_type 
        num_d 
    '''
    if d_net_type_str =="nlayer":
        return NLayerDiscriminator(input_nc=input_nc)
    elif d_net_type_str =="pixel":
        return PixelDiscriminator(input_nc=input_nc)
    elif d_net_type_str =="multi":
        assert num_d >=1 
        return [
            Discriminator_multi(nc) for nc in range(num_d) 
        ]
    else :
        raise Exception("unlon")
    pass 


class Discriminator_multi(nn.Module):
    def __init__(self, downsample=1):
        super().__init__()

        def conv_bn_lrelu(in_channels, out_channels, ksize, stride, pad):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )

        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.LeakyReLU(),
            conv_bn_lrelu(64, 64, 4, 2, 1),
            conv_bn_lrelu(64, 128, 3, 1, 1),
            conv_bn_lrelu(128, 128, 4, 2, 1),
            conv_bn_lrelu(128, 256, 3, 1, 1),
            conv_bn_lrelu(256, 256, 4, 2, 1),
            conv_bn_lrelu(256, 512, 3, 1, 1),
            conv_bn_lrelu(512, 512, 3, 1, 1),
            nn.Conv2d(512, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        if downsample > 1:
            self.avg_pool = nn.AvgPool2d(downsample)

        self.downsample = downsample

    def forward(self, x):
        if self.downsample > 1:
            x = self.avg_pool(x)

        out = self.body(x)
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


def define_vgg(opt ):
    if opt.vgg =="classify":
        return  vgg19_torchvision(True)
    elif opt.vgg == "transfer":
        return load_vgg16(opt.model_dir)
    else:
        raise Exception("errror:"+opt.vgg) 

    
class vgg19_torchvision(nn.Module):
    def __init__(self,pre_trained=True):
        super(vgg19_torchvision, self).__init__()

#         vgg19_office = torchvision.models.vgg.vgg19(pretrained=pre_trained)
        vgg19_office = torchvision.models.vgg.vgg19(pretrained=pre_trained)
        o_features = vgg19_office.features 
        self.features=    nn.Sequential(*list(o_features.children())[:-1]) 
        
    def forward(self,x):
        out = self.features(x)
        return out 
      

def load_vgg16(model_dir):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(os.path.join(model_dir, 'vgg16.weight')):
        raise Exception("fail get vgg weight")
#         if not os.path.exists(os.path.join(model_dir, 'vgg16.t7')):
#             os.system('wget https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1 -O ' + os.path.join(model_dir, 'vgg16.t7'))
#         vgglua = load_lua(os.path.join(model_dir, 'vgg16.t7'))
#         vgg = Vgg16()
#         for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
#             dst.data[:] = src
#         torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg16.weight'))
    vgg = Vgg16()
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    return vgg


def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(mean) # subtract mean
    return batch


##################################################################################
# VGG network definition
##################################################################################
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        # relu4_3 = h

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        relu5_3 = h

        return relu5_3
        # return [relu1_2, relu2_2, relu3_3, relu4_3]
    
# if __name__ == "__main__":
#     import  unittest 
#     class TestAll(unittest.TestCase):
#         
#         def setUp(self):
#             class att(object) :
#                 vgg = "transfer" #"transfer"
#                 model_dir  = "./"
#             self.opt = att() 
#             if self.opt . vgg =="transfer":
#                 m = Vgg16 ()
#                 if os.path.isfile(os.path.join(self.opt.model_dir ,"vgg16.weight")):
#                     torch.save(m.state_dict(),"./vgg16.weight")
#             
#             
#         def test_vgg(self):
#             print (self.opt) 
#             ##mock 
#             vgg = define_vgg(self.opt )
#             self.assertEqual(len([x for x in vgg.named_children()]) ,13 )
#             
#             setattr(self.opt ,"vgg" ,"classify")
#             vgg = define_vgg(self.opt )
#             self.assertEqual(len([x for x in vgg.named_children()]) ,1 )
# 
#             
#     
#     unittest.main()
