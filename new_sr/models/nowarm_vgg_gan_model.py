'''
Created on Apr 30, 2019

@author: gjpicker
'''
import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class NowarmVggGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_vgg_A, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G: A -> B; G_vgg_A: B -> A.
        Discriminators: D_A: G(A) vs. B; D_vgg_A: G_vgg_A(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_vgg_A(G(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_vgg_A * ||G(G_vgg_A(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G(B) - B|| * lambda_vgg_A + ||G_vgg_A(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_vgg_A', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'D_pixel', 'D_vgg',"D_GP","D_vgg_GP", 'perception', 'pixel', ]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_hr', "real_cub",'fake_sr' ]
        visual_names_vgg_A = [  ]

        self.visual_names = visual_names_A + visual_names_vgg_A  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G', 'D', 'D_vgg']
        else:  # during test time, only load Gs
            self.model_names = ['G', ]

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G (G), G_vgg_A (F), D_A (D_Y), D_vgg_A (D_X)
        self.netG = networks.define_G()


        if self.isTrain:  # define discriminators
            self.netD_vgg = networks.define_D_vgg()
            self.netD = networks.define_D()
            self.vgg = networks.define_vgg()

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters(), self.netD_vgg.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_hr = input['hr'].to(self.device)
        self.real_lr = input['lr'].to(self.device)
        self.real_cub = input['cubic'].to(self.device)
#         self.real_lr = input['lr2'].to(self.device)
#         self.real_lr = input['lr3'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_sr = self.netG(self.real_lr)  # G(A)
        
        if self.isTrain:
            self.fake_sr_vgg = self.vgg(self.fake_sr)
            self.real_hr_vgg = self.vgg(self.real_hr)
            

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_pixel(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_sr = self.fake_vgg_A_pool.query(self.fake_sr)
        
        loss_GP, _ = networks.cal_gradient_penalty(self.netD, self.real_hr, fake_sr.detach(), \
        self.device, type=self.opt.gp_type_3D, constant=self.opt.gp_norm_3D,
                lambda_gp=self.opt.lambda_gp_3D * self.opt.lambda_GAN_3D)
        if type(loss_GP) != float:
            loss_GP.backward(retain_graph=True)
            self.loss_D_GP= loss_GP
            
        self.loss_D_pixel = self.backward_D_basic(self.netD_A, self.real_hr, fake_sr)

    def backward_D_vgg(self):
        """Calculate GAN loss for discriminator D_vgg_A"""
        fake_sr_vgg = self.fake_A_pool.query(self.fake_sr_vgg)
        
        loss_GP, _ = networks.cal_gradient_penalty(self.netD_vgg, self.real_hr_vgg, fake_sr_vgg.detach(), \
            self.device, type=self.opt.gp_type_3D, constant=self.opt.gp_norm_3D,
                            lambda_gp=self.opt.lambda_gp_3D * self.opt.lambda_GAN_3D)
        if type(loss_GP) != float:
            loss_GP.backward(retain_graph=True)
            self.loss_D_vgg_GP= loss_GP
            
        self.loss_D_vgg = self.backward_D_basic(self.netD_vgg, self.real_hr_vgg, fake_sr_vgg)

    def backward_G(self):
        """Calculate the loss for generators G and G_vgg_A"""
        lambda_vgg_loss = self.opt.lambda_vgg_loss
        lambda_pixel = self.opt.lambda_pixel
        lambda_perception = self.opt.lambda_perception
        
        # Identity loss
        if lambda_vgg_loss > 0:
            self.loss_G_vgg = self.criterionGAN(self.netD_vgg(self.fake_sr_vgg), True)
        else:
            self.loss_G_vgg = 0

        # GAN loss D_A(G(A))
        self.loss_G = self.criterionGAN(self.netD(self.fake_sr), True)

        # Forward perception loss 
        self.loss_perception = self.criterionIdt(self.fake_sr_vgg, self.real_hr_vgg) * lambda_perception
        
        # Forward pixel loss 
        self.loss_pixel = self.criterionIdt(self.fake_sr, self.real_hr) * lambda_pixel

        # combined loss and calculate gradients
        self.loss_G = self.loss_G + self.loss_perception + self.loss_pixel 
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G and G_vgg_A
        self.set_requires_grad([self.netD, self.netD_vgg], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G and G_vgg_A's gradients to zero
        self.backward_G()             # calculate gradients for G and G_vgg_A
        self.optimizer_G.step()       # update G and G_vgg_A's weights
        # D_A and D_vgg_A
        self.set_requires_grad([self.netD, self.netD_vgg], True)
        self.optimizer_D.zero_grad()   # set D_A and D_vgg_A's gradients to zero
        self.backward_D_pixel()      # calculate gradients for D_A
        self.backward_D_vgg()      # calculate graidents for D_vgg_A
        self.optimizer_D.step()  # update D_A and D_vgg_A's weights