import torch
from models.base_model import BaseModel
from models import networks
import os
from layers.normalizer import DataNormalizer
from adabelief_pytorch import AdaBelief
from layers.attention import MutualAttention
from utils.eeg_tools import Configuration
import itertools
import numpy as np


class Pix2PixAEPartialModel(BaseModel):
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
        parser.add_argument('--load_path', type=str, default="", help='path to load checkpoint')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_lat', type=float, default=1.0, help='weight for lat loss')
        # parser.add_argument('--loss_freq', type=int, default=50, help='frequency of saving loss plots')

        return parser

    
    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseModel.__init__(self, opt)
        self.conf = Configuration()
        
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_real', 'D_fake', 'Wasserstein', 'D', 'D_grad', 'D_eeg', 'G', 'G_L1', 'G_GAN', 'mag_L1',
                           'phase_L1']
        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']  
        
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
            # self.model_names = ['G', 'D', 'D_vgg']
        else:  # during test time, only load G
            self.model_names = ['G']
        
        # define networks (both generator and discriminator)
        self.AtoB = self.opt.direction == 'AtoB'

        self.netG = networks.define_G(opt, opt.input_nc, opt.output_nc, opt.ngf, 'resnet', n_blocks=opt.n_blocks,
                                      norm=opt.norm,
                                      init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids,
                                      device=self.device)
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt, opt.input_nc + opt.output_nc, opt.ndf, opt.netD, 2,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,
                                          device=self.device, aux=opt.d_aux)
            # self.netAttn = networks.init_net(opt, MutualAttention(self.conf.w, self.conf.h), init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids, device=self.device)
    
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.g_lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.d_lr, betas=(opt.beta1, 0.999))\

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.train_G = False
            self.train_D = True
            self.use_L1 = True
            self.lambda_L1 = opt.lambda_L1
            amplified_matrix = torch.ones((1, 1, 128, 128))
            self.amplified_matrix = amplified_matrix.index_fill(2, torch.LongTensor(range(40, 128)), 0.2).to(
                self.device)

        self.normalizer = None

    
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """

        self.real_A = self.normalizer.normalize(input['A' if self.AtoB else 'B'], 'seeg' if self.AtoB else 'eeg').to(
            self.device)
        self.real_B = self.normalizer.normalize(input['B' if self.AtoB else 'A'], 'eeg' if self.AtoB else 'seeg').to(
            self.device)
        self.image_paths = input['A_paths' if self.AtoB else 'B_paths']

    
    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            if len(opt.load_path) == 0:
                self.load_networks(load_suffix)
            else:
                self.load_partial_model(load_suffix, opt.load_path)
        
        for loss_name in self.loss_names:
            setattr(self, 'loss_' + loss_name, 0)
        self.print_networks(opt.verbose)

    
    def load_partial_model(self, epoch, path=None):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """

        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                if path is None:
                    load_path = os.path.join(self.save_dir, load_filename)
                else:
                    load_path = os.path.join(path, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel) or isinstance(net, torch.nn.parallel.DistributedDataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self._BaseModel__patch_instance_norm_state_dict(state_dict, net, key.split('.'))

                if name == 'D':
                    model_dict = net.state_dict()
                    model_dict['model.0.weight'][:, :4, :, :].data = state_dict['model.0.weight'].data
                    model_dict['model.1.layer_.weight'][:, :4, :, :].data = state_dict['model.1.layer_.weight'].data
                    state_dict = {k: v for k, v in state_dict.items() if
                                  (k in model_dict.keys()) and (k not in ['model.0.weight', 'model.1.layer_.weight'])}
                    model_dict.update(state_dict)
                    net.load_state_dict(model_dict)
                else:
                    net.load_state_dict(state_dict)

    
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

    
    def get_attention(self):
        self.forward()

        from util.eeg_tools import IF_to_eeg, Configuration, phase_operation
        import librosa

        conf = Configuration()
        fake_temp = IF_to_eeg(self.fake_B.squeeze().detach().numpy(), self.normalizer, iseeg=False, is_IF=True)
        fake_s_spec = librosa.stft(fake_temp[0], n_fft=conf.seeg_n_fft, win_length=conf.seeg_win_len,
                                   hop_length=conf.seeg_hop)
        s_mag = np.log(np.abs(fake_s_spec) + conf.epsilon)[: conf.w]
        s_angle = np.angle(fake_s_spec)[: conf.w]
        s_IF = phase_operation.instantaneous_frequency(s_angle, time_axis=1)
        fake_s_spec = np.stack((s_mag, s_IF), axis=0)[None, ...]
        fake_B = self.normalizer.normalize(torch.from_numpy(fake_s_spec).to(torch.float32), 'seeg')

        real_temp = IF_to_eeg(self.real_B.squeeze().detach().numpy(), self.normalizer, iseeg=False, is_IF=True)
        real_s_spec = librosa.stft(real_temp[0], n_fft=conf.seeg_n_fft, win_length=conf.seeg_win_len,
                                   hop_length=conf.seeg_hop)
        real_s_mag = np.log(np.abs(real_s_spec) + conf.epsilon)[: conf.w]
        real_s_angle = np.angle(real_s_spec)[: conf.w]
        real_s_IF = phase_operation.instantaneous_frequency(real_s_angle, time_axis=1)
        real_s_spec = np.stack((real_s_mag, real_s_IF), axis=0)[None, ...]
        real_B = self.normalizer.normalize(torch.from_numpy(real_s_spec).to(torch.float32), 'seeg')

        real_AB = torch.cat((self.real_A, real_B), 1)
        _, real_attn, r_query, r_key = self.netD(real_AB)
        fake_AB = torch.cat((self.real_A, fake_B), 1)
        _, fake_attn, f_query, f_key = self.netD(fake_AB.detach())
        return real_attn.detach(), fake_attn.detach(), r_query.detach(), r_key.detach(), f_query.detach(), f_key.detach()


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        print(self.real_A.shape)
        self.fake_B = self.netG(self.real_A)  # G(A)


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake
        # stop backprop to the generator by detaching fake_B !!!!! attach importance!!!!
        # fake.detach() sets fake_grad_fn to none, which enables it to be sent as input as a pure tensor and avoids duplicate grad calculations
        
        # fake_attn_feat = self.netAttn(self.real_A[:, :1, :, :].detach(), self.fake_B[:, :1, :, :].detach())
        # fake_phase_diff = self.fake_B[:, 1:, :, :] - self.real_A[:, 1:, :, :]
        # fake_AB = torch.cat((self.real_A[:, :1, :, :], self.fake_B[:, :1, :, :]), 1)
        # pred_fake, _ = self.netD(fake_AB.detach())

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
        
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5 + gradient_penalty
        self.loss_D.backward()


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
        # amplified_real_phase = self.real_B[:, 1:, :, :] * self.amplified_matrix
        # amplified_fake_phase = self.fake_B[:, 1:, :, :] * self.amplified_matrix
        # self.loss_mag_L1 = self.criterionL1(self.fake_B[:, :1, :, :], self.real_B[:, :1, :, :]) * self.opt.lambda_L1 * 0.5
        # self.loss_phase_L1 = self.criterionL1(self.fake_B[:, 1:, :, :], self.real_B[:, 1:, :, :]) * self.opt.lambda_L1 * 0.5
        # self.loss_phase_L1 = self.criterionL1(amplified_fake_phase, amplified_real_phase) * self.opt.lambda_L1 / 5
        # self.loss_G_L1 = self.loss_mag_L1 + self.loss_phase_L1

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1  # + self.loss_G_lat_consis
        self.loss_G.backward()


    def optimize_parameters(self):

        # self.set_VGG_requires_grad(self.netD_vgg, True)  # enable backprop for D_vgg
        n_alter = 1
        self.forward()  # compute fake images: G(A)

        if self.train_D:
            n_alter = 5
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            # update D
            self.optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D()  # calculate gradients for D
            self.optimizer_D.step()  # update D's weights

        # update G
        if self.train_G and self.batch_idx % n_alter == 0:
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()  # set G's gradients to zero
            self.backward_G()  # calculate graidents for G
            self.optimizer_G.step()  # udpate G's weights
