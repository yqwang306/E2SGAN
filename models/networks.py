import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torchvision import models
import sys
import os
from models.swin_transformer import SwinTransformer
import numpy as np
import random
from layers.PGGAN import PixelWiseNormLayer, EqualizedLearningRateLayer
from layers.attention import MutualAttention, OneSidedAttention, GCN
import torch.nn.functional as F


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_size_after_conv(inputsize, n_conv, kernal_size, stride, padding_size): # (W − kernel_size + 2*padding ) / stride+1
    w, h = inputsize
    w_ls = []
    h_ls = []
    for i in range(n_conv):
        w = (w - kernal_size + 2 * padding_size) // stride + 1
        h = (h - kernal_size + 2 * padding_size) // stride + 1
        w_ls.append(w)
        h_ls.append(h)

    return w_ls, h_ls


def get_size_after_deconv(inputsize, n_conv, kernal_size, stride, padding_size, output_padding): # (W - 1）* stride + output_padding - 2 * padding + kernel_size
    w, h = inputsize
    w_ls = []
    h_ls = []
    for i in range(n_conv):
        w = (w - 1) * stride + output_padding - 2 * padding_size + kernal_size
        h = (h - 1) * stride + output_padding - 2 * padding_size + kernal_size
        w_ls.append(w)
        h_ls.append(h)

    return w_ls, h_ls


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """

    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions.
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """

    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


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


def init_net(opt, net, init_type='normal', init_gain=0.02, gpu_ids=[], device=None):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(device)
        if opt.isTrain and 'eeggan' not in opt.model and len(gpu_ids) > 0:
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[opt.local_rank],
                                                        output_device=opt.local_rank, find_unused_parameters=True)
        elif 'eeggan' in opt.model:
            net = torch.nn.DataParallel(net, gpu_ids)
        else:
            net = torch.nn.DataParallel(net, [0])  
            # net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(opt, input_nc, output_nc, ngf, netG, convD=2, n_blocks=6, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], device=None):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """

    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, convD=convD)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, convD=convD)
    elif netG == 'resnet':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks,
                              convD=convD)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet':
        net = UnetGenerator(input_nc, output_nc, n_blocks, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(opt, net, init_type, init_gain, gpu_ids, device=device)


def define_D(opt, input_nc, ndf, netD, convD=2, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], device=None, aux=None):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """

    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, convD=convD, aux=aux)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, convD=convD, aux=aux)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'classsifier':
        net = CNNClassifier(input_nc, 7, ndf=ndf, n_layers=n_layers_D, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(opt, net, init_type, init_gain, gpu_ids, device=device)


def cafe_ver_vgg16(path):
    ''' Load the caffe version vgg16 and convert it to the pytorch version. Reference to https://github.com/jcjohnson/pytorch-vgg

    :param path: the path of vgg16
    
    :return: vgg16 model
    '''

    net = models.vgg16()
    pre = torch.load(path)

    pre['classifier.0.weight'] = pre['classifier.1.weight']
    pre['classifier.0.bias'] = pre['classifier.1.bias']
    del pre['classifier.1.weight']
    del pre['classifier.1.bias']

    pre['classifier.3.weight'] = pre['classifier.4.weight']
    pre['classifier.3.bias'] = pre['classifier.4.bias']
    del pre['classifier.4.weight']
    del pre['classifier.4.bias']
    net.load_state_dict(pre)
    return net


##############################################################################
# Classes
##############################################################################
class LayerActivations:
    features = None

    def __init__(self, feature):
        self.hook = feature.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


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
                loss = -prediction.mean()  # tensor
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
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
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


class PublicLayer(nn.Module):

    def __init__(self, size, layer):
        super(PublicLayer, self).__init__()
        assert len(size) == 3
        self.size = size
        self.layer = layer
        self.publicfeat = torch.nn.Parameter(torch.FloatTensor(*self.size))
        init.xavier_normal_(self.publicfeat.data, gain=0.02)

    def forward(self, x):

        for i in range(x.shape[0]):
            x[i] += self.publicfeat
        return x


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, convD=2, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """

        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = (norm_layer.func == nn.InstanceNorm2d or norm_layer.func == nn.InstanceNorm3d)
        else:
            use_bias = (norm_layer.func == nn.InstanceNorm2d or norm_layer.func == nn.InstanceNorm3d)
        use_bias = False

        seeg_chans = 130
        eeg_chans = 17
        model = []

        if convD == 3:
            padding_type = 'replicate'
        self.convD = convD

        if convD == 3:
            oneD_opr1 = [nn.Conv3d(seeg_chans, eeg_chans, kernel_size=1, stride=1, padding=0, bias=use_bias),
                    norm_layer(eeg_chans),
                    nn.ReLU(True)]  
            self.oneD_opr1 = nn.Sequential(*oneD_opr1)

        model += [PixelWiseNormLayer()]

        if convD == 2:
            model += [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)]
        elif convD == 3:
            model += [nn.ReflectionPad3d(3), nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)]
        
        model += [EqualizedLearningRateLayer(model[-1]), nn.LeakyReLU(negative_slope=0.2), PixelWiseNormLayer()]

        n_downsampling = 2  
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if convD == 2:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias)]
                model += [EqualizedLearningRateLayer(model[-1]), nn.LeakyReLU(negative_slope=0.2), PixelWiseNormLayer()]
            elif convD == 3:
                model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]

        publicfeat_size = (1, 128 // (2 ** n_downsampling), 128 // (2 ** n_downsampling))
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias, convD=convD)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if convD == 2:
                model += [nn.Upsample(scale_factor=2, mode='nearest'),  # Upsampling + Conv
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias)]
                model += [EqualizedLearningRateLayer(model[-1]), nn.LeakyReLU(negative_slope=0.2), PixelWiseNormLayer()]
            elif convD == 3:
                model += [nn.Upsample(scale_factor=2, mode='nearest'),  # Upsampling + Conv
                          nn.Conv3d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]

        self.model = nn.Sequential(*model)

        if convD == 3:
            oneD_opr2 = [nn.Conv3d(24, eeg_chans, kernel_size=1, stride=1, padding=0, bias=use_bias),  
                         norm_layer(eeg_chans),
                         nn.ReLU(True)]

            self.oneD_opr2 = nn.Sequential(*oneD_opr2)

        final = []
        if convD == 2:
            final += [nn.ReplicationPad2d(3)]
            final += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0, bias=use_bias)]
            final += [EqualizedLearningRateLayer(final[-1])]
        elif convD == 3:
            final += [nn.ReplicationPad3d(3)]
            final += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0)]

        final += [nn.Tanh()]  # scale to (-1,1)
        self.final = nn.Sequential(*final)

    
    def forward(self, x):
        """Standard forward"""
        if self.convD == 2:
            x = self.model(x)
            x = self.final(x)
        elif self.convD == 3:
            x = self.oneD_opr1(x.transpose(2, 1))
            x = self.model(x.transpose(1, 2))
            x = self.oneD_opr2(x.transpose(2, 1))
            x = self.final(x.transpose(1, 2))
        return x


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, convD):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, convD)

    
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, convD):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """

        conv_block = []
        p = 0

        if convD == 2:
            if padding_type == 'reflect':
                conv_block += [nn.ReflectionPad2d(1)]
            elif padding_type == 'replicate':
                conv_block += [nn.ReplicationPad2d(1)]
            elif padding_type == 'zero':
                p = 1
            else:
                raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        elif convD == 3:
            conv_block += [nn.ReplicationPad3d(1)]

        if convD == 2:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        elif convD == 3:
            conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]

        conv_block += [EqualizedLearningRateLayer(conv_block[-1]), nn.LeakyReLU(negative_slope=0.2),
                       PixelWiseNormLayer()]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if convD == 2:
            if padding_type == 'reflect':
                conv_block += [nn.ReflectionPad2d(1)]
            elif padding_type == 'replicate':
                conv_block += [nn.ReplicationPad2d(1)]
            elif padding_type == 'zero':
                p = 1
            else:
                raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        elif convD == 3:
            conv_block += [nn.ReplicationPad3d(1)]

        if convD == 2:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        elif convD == 3:
            conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        
        conv_block += [EqualizedLearningRateLayer(conv_block[-1])]
        
        self.relu = nn.Sequential(nn.LeakyReLU(negative_slope=0.2), PixelWiseNormLayer())

        return nn.Sequential(*conv_block)

    
    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        out = self.relu(out)  
        return out

    
    def set_requires_grad(self, requires_grad=False):
        for p in self.parameters():
            p.requires_grad = requires_grad


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """

        super(UnetGenerator, self).__init__()
        down_w_ls, _ = get_size_after_conv((224, 224), num_downs, 4, 2, 1)
        up_w_ls, _ = get_size_after_deconv((down_w_ls[-1], down_w_ls[-1]), num_downs, 4, 2, 1, 0)
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, feat_w=up_w_ls[0])  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, feat_w=up_w_ls[i+1])
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, feat_w=up_w_ls[-4])
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, feat_w=up_w_ls[-3])
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, feat_w=up_w_ls[-2])
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, feat_w=up_w_ls[-1])  # add the outermost layer

    
    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, feat_w=-1):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """

        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, upnorm, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            xx = self.model(x)
            return torch.cat([x, xx], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, convD=2, aux=[]):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """

        super(NLayerDiscriminator, self).__init__()
        
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = (norm_layer.func == nn.InstanceNorm2d or norm_layer.func == nn.InstanceNorm3d)
        else:
            use_bias = (norm_layer.func == nn.InstanceNorm2d or norm_layer.func == nn.InstanceNorm3d)

        sequence = []
        self.aux = aux
        use_bias = False

        if 'attn' in aux:
            self.attn_block = MutualAttention(128, 128)
            self.attn_norm = norm_layer(2)
            input_nc += 2
        elif 'corrcoef' in aux:
            self.corrcorf_block = CorrcoefMatrix()
            input_nc += 1
        elif 'onesided' in aux:
            self.attn_block = OneSidedAttention(128, 128)
            self.attn_norm = norm_layer(1)
            input_nc += 1
        elif 'gcn' in aux:
            self.gcn = [GCN(1, 32), norm_layer(1),
                   GCN(32, 64), norm_layer(1)]

            self.adj = []
            self.gcnLayer = nn.Sequential(*self.gcn)
            self.gap = F.adaptive_avg_pool2d
            self.fuse = nn.Linear(14 * 14 + 64, 1)  

        kw = 4
        padw = 1
        if convD == 2:
            sequence += [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw, bias=use_bias)]
            sequence += [EqualizedLearningRateLayer(sequence[-1]), nn.LeakyReLU(0.2, True)]
        elif convD == 3:
            sequence += [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if convD == 2:
                sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)]
            elif convD == 3:
                sequence += [nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)]
            sequence += [
                EqualizedLearningRateLayer(sequence[-1]),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        if convD == 2:
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)]
        elif convD == 3:
            sequence += [nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)]
        sequence += [
            EqualizedLearningRateLayer(sequence[-1]),
            nn.LeakyReLU(0.2, True)
        ]

        if convD == 2:
            if 'linear' in aux:
                sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw, bias=use_bias)]
                self.nonlin = nn.LeakyReLU(0.2, True)
                self.fuse = nn.Linear(14 * 14 + 1, 1)
            else:
                sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw, bias=use_bias)]
        elif convD == 3:
            sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    
    def get_adj(self):
        return self.adj

    
    def forward(self, x):
        """Standard forward."""
        n_batch = x.size(0)

        if 'attn' in self.aux or 'onesided' in self.aux:
            attn_feat = self.attn_block(x[:, :1, :, :], x[:, 2:3, :, :])
            x = torch.cat((x, self.attn_norm(attn_feat)), 1)
            
        elif 'corrcoef' in self.aux:
            corrcoefmat = self.corrcorf_block(x)
            x = torch.cat((x, corrcoefmat), 1)
            
        elif 'gcn' in self.aux:
            gcn_feat = self.gcnLayer(torch.cat((x[:, :1, :, :], x[:, 2:3, :, :]), dim=-2))
            gcn_feat = self.gap(gcn_feat, (1, 1)).reshape(n_batch, -1)
            self.adj = []
            for i in range(0, len(self.gcn), 2):
                self.adj.append(self.gcn[i].get_adj())

        x = self.model(x)

        if 'linear' in self.aux:
            x = x.reshape(n_batch, -1)
            ## weighted patch prediction
            global_view = self.nonlin(x.mean(axis=1, keepdims=True))  # p_global
            x = self.fuse(torch.cat((x, global_view), 1))  # concat p_global and p
            # x = x * self.voting_weights

        if 'gcn' in self.aux:
            x = x.reshape(n_batch, -1)
            x = self.fuse(torch.cat((x, gcn_feat), 1))

        return x


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
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

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


class VGG16_AE(nn.Module):

    def __init__(self, version, pre_path=None, input_nc=2, output_nc=2, ngf=32, norm_layer=nn.InstanceNorm2d, use_bias=True):

        super(VGG16_AE, self).__init__()

        if version == 11:
            net = models.vgg11()
        else:
            net = models.vgg16()  
        if pre_path is not None:
            pre = torch.load(pre_path)
            net.load_state_dict(pre)

        norm_layer = get_norm_layer('instance')
        en_features = [nn.Conv2d(input_nc, 3, kernel_size=1, stride=1, padding=0, bias=False)]  # 1x1 conv
        if version == 11:
            en_features += list(net.features)[: 10]  
        else:
            en_features += list(net.features)[: 23]  

        if version == 11:
            k_size = 256
            n_itr = 2
        else:
            k_size = 512
            n_itr = 3
        de_features = []
        for i in range(n_itr):
            de_features += [
                nn.ConvTranspose2d(k_size, k_size // 2,
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1,
                                   bias=use_bias),
                            norm_layer(k_size // 2), nn.LeakyReLU(0.2, True),
                            nn.Conv2d(k_size // 2, k_size // 2, kernel_size=3, padding=1, bias=False),
                            norm_layer(k_size // 2), nn.LeakyReLU(0.2, True)]
            k_size = k_size // 2

        de_features += [nn.Conv2d(64, output_nc, kernel_size=1, stride=1, padding=0, bias=False), nn.Tanh()]

        self.encoder = nn.Sequential(*en_features)
        self.decoder = nn.Sequential(*de_features)

        self.init_network()

    
    def forward(self, x):
        """Standard forward"""
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    
    def init_network(self):

        m = self.encoder[0]  
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

        for m in self.decoder:
            classname = getattr(getattr(m, '__class__'), '__name__')
            if classname.find('Conv2d') != -1:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    
    def print_parameters(self):
        print(self.encoder)
        print(self.decoder)
        for p in self.parameters():
            print(p.size())


class VGG_generator(nn.Module):

    def __init__(self, vgg_version, encoder, decoder):

        super(VGG_generator, self).__init__()

        norm_layer = get_norm_layer('instance')
        self.encoder = encoder  
        self.vgg_version = vgg_version

        decoder = list(decoder)
        down_0 = []
        down_1 = []
        down_2 = []
        down_3 = []
        if vgg_version == 11:
            for i, m in enumerate(self.encoder):
                if i < 3:
                    down_2.append(m)
                elif i < 6:
                    down_1.append(m)
                else:
                    down_0.append(m)
        else:
            for i, m in enumerate(self.encoder):
                if i < 5:
                    down_3.append(m)
                elif i < 10:
                    down_2.append(m)
                elif i < 17:
                    down_1.append(m)
                else:
                    down_0.append(m)
        self.down_0 = nn.Sequential(*down_0)
        self.down_1 = nn.Sequential(*down_1)
        self.down_2 = nn.Sequential(*down_2)
        if vgg_version != 11:
            self.down_3 = nn.Sequential(*down_3)
        if vgg_version == 11:
            k_size = 256
        else:
            k_size = 512

        self.up_0 = nn.Sequential(*decoder[: 3])
        self.up_1 = nn.Sequential(*[nn.Conv2d(k_size, k_size // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                    norm_layer(k_size), nn.LeakyReLU(0.2, True)] + decoder[3: 9])
        k_size = k_size // 2
        if vgg_version == 11:
            self.up_2 = nn.Sequential(*[nn.Conv2d(k_size, k_size // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                        norm_layer(k_size), nn.LeakyReLU(0.2, True)] + decoder[9:])
        else:
            self.up_2 = nn.Sequential(*[nn.Conv2d(k_size, k_size // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                      norm_layer(k_size), nn.LeakyReLU(0.2, True)] + decoder[9: 15])
            k_size = k_size // 2
            self.up_3 = nn.Sequential(*[nn.Conv2d(k_size, k_size // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_layer(k_size), nn.LeakyReLU(0.2, True)] + decoder[15:])


    def forward(self, x):
        if self.vgg_version == 11:
            down_x2 = self.down_2(x)
        else:
            down_x3 = self.down_3(x)
            down_x2 = self.down_2(down_x3)
        down_x1 = self.down_1(down_x2)
        down_x0 = self.down_0(down_x1)

        self.mid_hook = down_x0

        up_x0 = self.up_0(down_x0)
        up_x1 = self.up_1(torch.cat([down_x1, up_x0], 1))
        up_x2 = self.up_2(torch.cat([down_x2, up_x1], 1))

        if self.vgg_version == 11:
            output = up_x2
        else:
            output = self.up_3(torch.cat([down_x3, up_x2], 1))
        self.out_hook = output

        return output

    
    def return_hook(self):
        return self.mid_hook.grad, self.out_hook.grad


class AE_generator(nn.Module):

    def __init__(self, opt, eeg_chans, seeg_chans):
        super(AE_generator, self).__init__()
        self.encoder = AE_component(opt, 'seeg', 'encoder', eeg_chans, seeg_chans)
        self.decoder = AE_component(opt, 'eeg', 'decoder', eeg_chans, seeg_chans)
        model = [self.encoder, self.decoder]

        self.model = nn.Sequential(*model)

    
    def forward(self, x):
        return self.model(x)


class AE_component(nn.Module):

    def __init__(self, opt, type, component, eeg_chans=None, seeg_chans=None):
        super(AE_component, self).__init__()

        from .ae_model import AEModel
        netAE = AEModel(opt, eeg_chans, seeg_chans)
        netAE.load_networks('latest', path=os.path.join(opt.checkpoints_dir, opt.ae_name))  

        if component == 'encoder':
            if type == 'eeg':
                self.component = netAE.netAE_e.module.encoder
            else:
                self.component = netAE.netAE_s.module.encoder
        else:
            if type == 'eeg':
                self.component = netAE.netAE_e.module.decoder
            else:
                self.component = netAE.netAE_s.module.decoder

    
    def forward(self, x):
        return self.component(x)


class TransposeHelper(nn.Module):

    def __init__(self, d1, d2):
        super(TransposeHelper, self).__init__()
        self.d1 = d1
        self.d2 = d2

    
    def forward(self, x):
        return x.transpose(self.d1, self.d2)


class SwinTransformerDiscriminator(nn.Module):

    def __init__(self, preTrainPath, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):

        super(SwinTransformerDiscriminator, self).__init__()

        self.oneDconv = nn.Sequential(nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0, bias=False),
                    norm_layer((224, 224)), nn.LeakyReLU(0.2, True))
        self.swintransformer = SwinTransformer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                 embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                 window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                 norm_layer=norm_layer, ape=ape, patch_norm=patch_norm,
                 use_checkpoint=use_checkpoint)

        self.swintransformer.load_state_dict(torch.load(preTrainPath, map_location=torch.device('cpu'))['model'])  

        for param in self.swintransformer.parameters():
            param.requires_grad = False

        self.swintransformer.head = nn.Linear(768, 2)
        self.swintransformer.cuda()

    
    def forward(self, x):

        x = self.oneDconv(x)  
        x = self.swintransformer(x)
        return x


class CNNClassifier(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, n_class, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """

        super(CNNClassifier, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = (norm_layer.func == nn.InstanceNorm2d or norm_layer.func == nn.InstanceNorm3d)
        else:
            use_bias = (norm_layer.func == nn.InstanceNorm2d or norm_layer.func == nn.InstanceNorm3d)

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)]
            sequence += [
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)]
        sequence += [
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        self.model = nn.Sequential(*sequence)
        self.ln = nn.Linear(ndf * nf_mult * 27 * 27, n_class)

    
    def forward(self, x):
        """Standard forward."""
        x = self.model(x)
        x = self.ln(x.view(x.size(0), -1))
        return x


class FrequencyNorm(nn.Module):
    def __init__(self, C, W):
        super(FrequencyNorm, self).__init__()
        self.eps = 1e-6
        self.mean = nn.Parameter(torch.Tensor(1, C, W, 1))
        self.std = nn.Parameter(torch.Tensor(1, C, W, 1))
        self.initParams(self.mean.data)
        self.initParams(self.std.data)

    def initParams(self, data):
        init.xavier_normal_(data, gain=0.02)

    def forward(self, input):
        size = input.size()
        N, C, W = size[: 3]
        feat_var = input.var(dim=3) + self.eps
        feat_std = feat_var.sqrt().view(N, C, W, 1)
        feat_mean = input.mean(dim=3).view(N, C, W, 1)

        normalized_feat = (input - feat_mean.expand(
            size)) / feat_std.expand(size)
        return normalized_feat * self.std.expand(size) + self.std.expand(size)


class EEGEncoder(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """

        assert(n_blocks >= 0)
        super(EEGEncoder, self).__init__()

        use_bias = False
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)]
        model += [EqualizedLearningRateLayer(model[-1]), nn.LeakyReLU(negative_slope=0.2), PixelWiseNormLayer()]

        n_downsampling = 3  
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias)]
            model += [EqualizedLearningRateLayer(model[-1]), nn.LeakyReLU(negative_slope=0.2), PixelWiseNormLayer()]
            model += [nn.Conv2d(ngf * mult * 2, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias)]
            model += [EqualizedLearningRateLayer(model[-1]), nn.LeakyReLU(negative_slope=0.2), PixelWiseNormLayer()]
            model += [nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Standard forward"""

        x = self.model(x)
        return x


class CorrcoefMatrix(nn.Module):

    def __init__(self):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        super(CorrcoefMatrix, self).__init__()

    def corrcoef(self, x, y):
        """x.shape(n,m), m is channel number, n is time step, output its correlation coefficient matrix"""

        assert (x.shape == y.shape) and (len(x.shape) == 4)
        n_batch, n_feat, n_freq, n_time = x.size()

        f = (n_time - 1) / n_time  
        x_reducemean = x - torch.mean(x, axis=-1)
        y_reducemean = y - torch.mean(y, axis=-1)
        numerator = torch.matmul(x_reducemean, y_reducemean.transpose(-1, -2)) / n_time
        xvar_ = x.var(axis=-1)
        yvar_ = y.var(axis=-1)
        denominator = torch.sqrt(torch.matmul(xvar_, yvar_.transpose(-1, -2))) * f
        corrcoef = numerator / denominator
        return corrcoef

    def forward(self, input):
        x = input[:, :1, :, :]
        y = input[:, 2:3, :, :]
        return self.corrcoef(x, y)


class TSGAN_G(nn.Module):

    def __init__(self, temporal_len=1016):
        super(TSGAN_G, self).__init__()

        models = []
        self.reshapeOpr = nn.Sequential(nn.Linear(temporal_len, 256), nn.LeakyReLU(0.2, True))

        models += [nn.Upsample(scale_factor=2, mode='nearest'),
                   nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=True), nn.LeakyReLU(0.2, True)]
        models += [nn.Upsample(scale_factor=2, mode='nearest'),
                   nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True), nn.LeakyReLU(0.2, True)]
        models += [nn.Upsample(scale_factor=2, mode='nearest'),
                   nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True), nn.LeakyReLU(0.2, True)]
        models += [nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=True), nn.Tanh()]

        self.models = nn.Sequential(*models)

    def forward(self, x):

        x = self.reshapeOpr(x)
        x = self.models(x.view(-1, 1, 16, 16))
        return x


class TSGAN_Dx(nn.Module):

    def __init__(self, input_dim=2):
        super(TSGAN_Dx, self).__init__()

        models = []

        models += [nn.Conv2d(input_dim, 32, kernel_size=3, stride=2, padding=1, bias=True), nn.LeakyReLU(0.2, True)]
        models += [nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True), nn.LeakyReLU(0.2, True)]
        models += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True), nn.LeakyReLU(0.2, True)]
        models += [nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=True)]  # 14x14

        self.models = nn.Sequential(*models)

    def forward(self, x):
        x = self.models(x)
        return x


class TSGAN_F(nn.Module):

    def __init__(self, input_dim=1, temporal_len=1016):
        super(TSGAN_F, self).__init__()

        downsample = []

        downsample += [nn.Conv2d(input_dim, 32, kernel_size=3, stride=2, padding=1, bias=True), nn.LeakyReLU(0.2, True)]
        downsample += [nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True), nn.LeakyReLU(0.2, True)]
        downsample += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True), nn.LeakyReLU(0.2, True)]  # n_batch, 128, 28, 28
        # downsample += [nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=True)]

        self.downsample = nn.Sequential(*downsample)

        upsample = []

        # upsample += [nn.Linear(28 * 28, temporal_len // 4), nn.LeakyReLU(0.2, True)]
        upsample += [nn.Upsample(scale_factor=2, mode='nearest'),
                     nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1, bias=True), nn.LeakyReLU(0.2, True)]
        upsample += [nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1, bias=True), nn.LeakyReLU(0.2, True)]
        upsample += [nn.Linear(16 * 16 * 2, temporal_len), nn.Tanh()]

        self.upsample = nn.Sequential(*upsample)

    def forward(self, x):
        n_batch = x.size(0)
        x = self.downsample(x)
        x = self.upsample(x.view(n_batch, -1, 16 * 16))
        return x


class TSGAN_Dy(nn.Module):

    def __init__(self, input_dim=2):
        super(TSGAN_Dy, self).__init__()

        models = []

        models += [nn.Conv1d(input_dim, 32, kernel_size=3, stride=2, padding=1, bias=True), nn.LeakyReLU(0.2, True)]
        models += [nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=True), nn.LeakyReLU(0.2, True)]
        models += [nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1, bias=True), nn.LeakyReLU(0.2, True)]
        models += [nn.Conv1d(128, 1, kernel_size=3, stride=1, padding=1, bias=True)]  # 14x14

        self.models = nn.Sequential(*models)

    def forward(self, x):
        x = self.models(x)
        return x
