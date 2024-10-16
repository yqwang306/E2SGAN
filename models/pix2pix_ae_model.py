import torch
from models.base_model import BaseModel
from models import networks
import os
from layers.normalizer import DataNormalizer
from adabelief_pytorch import AdaBelief
from layers import MutualAttention
from utils.eeg_tools import Configuration
import itertools


class Pix2PixAEModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """

        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_lat', type=float, default=1.0, help='weight for lat loss')
        #parser.add_argument('--loss_freq', type=int, default=50, help='frequency of saving loss plots')

        return parser

    
    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        
        BaseModel.__init__(self, opt)
        self.conf = Configuration()
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_real', 'D_fake', 'Wasserstein', 'D', 'D_grad', 'G', 'G_L1', 'G_GAN', 'adj']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']#, 'latent', 'back_latent']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
            #self.model_names = ['G', 'D', 'D_vgg']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.AtoB = self.opt.direction == 'AtoB'

        self.netG = networks.define_G(opt, opt.input_nc, opt.output_nc, opt.ngf, 'resnet', n_blocks=opt.n_blocks, norm=opt.norm,
                                      init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids, device=self.device)
        
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt, opt.input_nc + opt.output_nc, opt.ndf, opt.netD, 2,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, device=self.device, aux=opt.d_aux)
    
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.g_lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.d_lr, betas=(opt.beta1, 0.999))
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
            self.train_G = False
            self.train_D = True
            self.use_L1 = True
            self.lambda_L1 = opt.lambda_L1

        self.normalizer = None

    
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """

        self.real_A = self.normalizer.normalize(input['A' if self.AtoB else 'B'], 'seeg' if self.AtoB else 'eeg').to(self.device)
        self.real_B = self.normalizer.normalize(input['B' if self.AtoB else 'A'], 'eeg' if self.AtoB else 'seeg').to(self.device)
        self.image_paths = input['A_paths' if self.AtoB else 'B_paths']

    
    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    
    def update_batch_idx(self, batch_idx):
        self.batch_idx = batch_idx

    
    def start_training_G(self):
        self.train_G = True

    
    def flip_training_D(self):
        self.train_D = not self.train_D

    
    def flip_training_G(self):
        self.train_G = not self.train_G

    
    def update_lambda_L1(self, value):
        if value == 0:
            self.use_L1 = False
            print("Stop using L1 loss")
        else:
            self.lambda_L1 = value
            print("Update lambda L1 to {}...".format(self.lambda_L1))

    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake
        # stop backprop to the generator by detaching fake_B !!!!! attach importance!!!!
        # fake.detach() sets fake_grad_fn to none, which enables it to be sent as input as a pure tensor and avoids duplicate grad calculations
        
        # fake_attn_feat = self.netAttn(self.real_A[:, :1, :, :].detach(), self.fake_B[:, :1, :, :].detach())
        # fake_phase_diff = self.fake_B[:, 1:, :, :] - self.real_A[:, 1:, :, :]
        # fake_AB = torch.cat((self.real_A[:, :1, :, :], self.fake_B[:, :1, :, :]), 1)
        # pred_fake = self.netD(fake_AB)

        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        
        # Real
        # real_attn_feat = self.netAttn(self.real_A[:, :1, :, :], self.real_B[:, :1, :, :])
        # real_phase_diff = self.real_B[:, 1:, :, :] - self.real_A[:, 1:, :, :]
        # real_AB = torch.cat((self.real_A[:, :1, :, :], self.real_B[:, :1, :, :]), 1)

        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # WGANGP penalty and loss
        gradient_penalty, _ = networks.cal_gradient_penalty(self.netD, real_AB, fake_AB.detach(), self.device)
        self.loss_D_grad = gradient_penalty
        self.loss_Wasserstein = - (self.loss_D_real + self.loss_D_fake)
        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5 + gradient_penalty

        self.loss_D.backward()

    
    def backward_for_cam(self, real=True, label=True):
        self.forward()

        if real:
            AB_pair = torch.cat((self.real_A, self.real_B), 1)
        else:
            AB_pair = torch.cat((self.real_A, self.fake_B), 1).detach()
        pred = self.netD(AB_pair)

        loss = self.criterionGAN(pred, label)
        loss.backward()

    
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        # First, G(A) should fake the discriminator
        # fake_attn_feat = self.netAttn(self.real_A[:, :1, :, :], self.fake_B[:, :1, :, :])
        # fake_phase_diff = self.fake_B[:, 1:, :, :] - self.real_A[:, 1:, :, :]
        # fake_AB = torch.cat((self.real_A[:, :1, :, :], self.fake_B[:, :1, :, :]), 1)  
        
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        
        # combine loss and calculate gradients
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 #+ self.loss_G_lat_consis
        self.loss_G.backward()

    
    def optimize_parameters(self):

        #self.set_VGG_requires_grad(self.netD_vgg, True)  # enable backprop for D_vgg
        n_alter = 1
        self.forward()  # compute fake images: G(A)

        if self.train_D:
            n_alter = 5
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            # update D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            # with torch.autograd.detect_anomaly():
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights

        # update G
        if self.train_G and self.batch_idx % n_alter == 0:
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            # with torch.autograd.detect_anomaly():
            self.backward_G()                   # calculate graidents for G
            self.optimizer_G.step()             # udpate G's weights
