import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from models.network import PGNet, set_requires_grad, VGGNet

from models import networks_gan
from loss import multiclassLoss, multidiceLoss, multiLabelLoss, multiclassLossAV, CrossEntropyLossWithSmooth, \
    L1LossWithLogits,multiHausdorffDTLoss, vggloss, gradient_penalty, tripletMarginLoss_vggfea, centernessLoss, SmoothL1_weighted
import os, copy
from opt_UKBB import get_patch_trad_5, modelEvalution
from collections import OrderedDict
import numpy as np


class SW_GAN():

    def __init__(self, opt, isTrain=True):
        self.cfg = opt
        self.use_GAN = opt.use_GAN
        self.isTrain = isTrain
        self.use_cuda = opt.use_cuda
        self.use_centerness = opt.use_centerness
        # initilize all the loss names for print in each iteration
        self.get_loss_names(opt)
        self.centerness_map_size = opt.centerness_map_size
        self.use_global_semantic = opt.use_global_semantic
        self.global_warmup_step = opt.global_warmup_step
        # define networks (both generator and discriminator)
        self.netG = PGNet(input_ch=opt.input_nc,
                          resnet=opt.use_network,
                          pretrained=True,
                          use_cuda=opt.use_cuda,
                          num_classes=opt.n_classes,
                          centerness=opt.use_centerness,
                          centerness_map_size=opt.centerness_map_size,
                          use_global_semantic=opt.use_global_semantic)

        if self.use_cuda:
            self.netG = self.netG.cuda()
        print(self.netG)

        if self.isTrain and opt.use_GAN:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            tmp_num_classes_D = opt.num_classes_D

            self.netD = networks_gan.define_D(input_nc=opt.input_nc_D, ndf=opt.ndf,
                                              netD=opt.netD_type, n_layers_D=opt.n_layers_D,
                                              norm=opt.norm, use_sigmoid=opt.use_sigmoid,
                                              init_type=opt.init_type, init_gain=opt.init_gain,
                                              gpu_ids=opt.gpu_ids, num_classes_D=tmp_num_classes_D,
                                              use_noise=opt.use_noise_input_D, use_dropout=opt.use_dropout_D)
            print(self.netD)
            self.netD.train()
            self.netG.train()

        if self.isTrain:

            # define loss functions

            self.criterionCE = nn.BCEWithLogitsLoss()
            self.criterionDICE = multidiceLoss()

            self.criterion = multiclassLoss()
            self.criterionHu = multiHausdorffDTLoss()
            # initialize optimizers and scheduler.
            if opt.use_SGD:
                self.optimizer_G = torch.optim.SGD(filter(lambda p: p.requires_grad, self.netG.parameters()), lr=opt.lr,
                                                   momentum=0.9, weight_decay=5e-4)
            else:
                self.optimizer_G = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.netG.parameters()),
                                                     lr=opt.lr, betas=(0.5, 0.999), weight_decay=5e-4)
            self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=opt.step_size,
                                                               gamma=opt.lr_decay_gamma)
            # self.scheduler_G = GradualWarmupScheduler(self.optimizer_G, multiplier=1, total_epoch=opt.step_size//20,
            # after_scheduler=self.scheduler_G)
            if opt.use_GAN:
                self.optimizer_D = torch.optim.AdamW(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),
                                                     weight_decay=5e-4)
                self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=opt.step_size,
                                                                   gamma=opt.lr_decay_gamma)
                # self.scheduler_D = GradualWarmupScheduler(self.optimizer_D, multiplier=1, total_epoch=opt.step_size//20,after_scheduler=self.scheduler_D)

            if opt.use_topo_loss:
                self.vggnet = VGGNet(opt.vgg_type, opt.vgg_layers, opt.use_cuda)
                self.vggnet.vgg.to(opt.device)
                self.criterionTopo = nn.SmoothL1Loss(reduction='mean')
            # define loss function for centerness score map
            if opt.use_centerness:
                self.criterionSmoothL1 = nn.SmoothL1Loss(reduction='none')
            if opt.use_av_cross:
                self.criterionAV = multiclassLossAV()

            if opt.use_high_semantic:
                self.criterionHighSemantic = multiLabelLoss()
    def setup(self, opt, log_folder):
        # define the directory for logger
        self.log_folder = log_folder
        # mkdir for training result
        self.train_result_folder = os.path.join(self.log_folder, 'training_result')
        if not os.path.exists(self.train_result_folder):
            os.mkdir(self.train_result_folder)
        # load network
        if not self.isTrain or opt.use_pretrained_G:
            model_path = os.path.join(opt.model_path_pretrained_G, 'G_' + str(opt.model_step_pretrained_G) + '.pkl')
            # self.netG.load_state_dict(torch.load(model_path), strict=False)
            pt = torch.load(model_path, map_location=self.cfg.device)
            model_static = self.netG.state_dict()
            pt_ = {k: v for k, v in pt.items() if k in model_static}
            model_static.update(pt_)
            self.netG.load_state_dict(model_static)

            print("Loading pretrained model for Generator from " + model_path)
            if opt.use_GAN and opt.use_pretrained_D:
                model_path_D = os.path.join(opt.model_path_pretrained_G,
                                            'D_' + str(opt.model_step_pretrained_G) + '.pkl')
                self.netD.load_state_dict(torch.load(model_path_D))
                print("Loading pretrained model for Discriminator from " + model_path_D)

    def set_input(self, step, train_data=None, train_label_data=None, train_label_data_centerness=None, train_patch_data=None,
                  train_patch_label=None):

        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap images in domain A and domain B.
        """

        opt = self.cfg

        self.step = step
        if self.use_centerness and not self.use_global_semantic:
            patch_data, patch_label, label_centerness = get_patch_trad_5(opt.batch_size, opt.patch_size,
                                                                         train_patch_data=train_patch_data,
                                                                         train_patch_label=train_patch_data,
                                                                         train_label_data_centerness=train_label_data_centerness)

        if self.use_centerness and self.use_global_semantic:
            patch_data, patch_label, label_centerness, data, label = get_patch_trad_5(opt.batch_size,
                                                                                           opt.patch_size,
                                                                                           train_data=train_data,
                                                                                           train_label_data=train_label_data,
                                                                                           train_patch_data=train_patch_data,
                                                                                           train_patch_label=train_patch_label,
                                                                                           train_label_data_centerness=train_label_data_centerness)
        if not self.use_centerness and not self.use_global_semantic:
            patch_data, patch_label = get_patch_trad_5(opt.batch_size, opt.patch_size, train_patch_data=train_patch_data,
                                                       train_patch_label=train_patch_label,
                                                       )

        if not self.use_centerness and self.use_global_semantic:
            patch_data, patch_label, data, label = get_patch_trad_5(opt.batch_size, opt.patch_size, train_data=train_data,
                                                                    train_label_data=train_label_data,
                                                                    train_patch_data=train_patch_data,
                                                                    train_patch_label=train_patch_label)

        self.patch_data_input = torch.FloatTensor(patch_data)
        self.patch_label_input = torch.FloatTensor(patch_label)

        if self.use_centerness:
            self.label_centerness_map = torch.FloatTensor(label_centerness)
            self.label_input_sm = torch.FloatTensor(copy.deepcopy(patch_label))
        if self.use_global_semantic:
            self.data_input = torch.FloatTensor(data)
            self.label_input = torch.FloatTensor(label)
        if opt.use_cuda:
            self.patch_data_input = self.patch_data_input.cuda()
            self.patch_label_input = self.patch_label_input.cuda()

            if self.use_centerness:
                self.label_centerness_map = self.label_centerness_map.cuda()
                self.label_input_sm = self.label_input_sm.cuda()
            if self.use_global_semantic:
                self.data_input = self.data_input.cuda()
                self.label_input = self.label_input.cuda()

        # downsample the centerness scores maps
        if self.use_centerness and self.centerness_map_size[0] == 128:
            self.label_centerness_map = F.interpolate(self.label_centerness_map, scale_factor=0.5, mode='bilinear',
                                                      align_corners=True)

            self.label_input_sm = F.interpolate(self.label_input_sm, scale_factor=0.5, mode='bilinear',
                                                align_corners=True)

        if self.use_centerness and self.centerness_map_size[0] == 256:
            self.label_centerness_map = self.label_centerness_map
            self.label_input_sm = self.label_input_sm

        self.patch_data_input = autograd.Variable(self.patch_data_input)
        self.patch_label_input = autograd.Variable(self.patch_label_input)

        if self.use_centerness:
            self.label_centerness_map = autograd.Variable(self.label_centerness_map)
            self.label_input_sm = autograd.Variable(self.label_input_sm)

        if self.use_centerness:
            self.label_centerness_map_all = [self.label_centerness_map]

            self.label_centerness_map_all = torch.cat(self.label_centerness_map_all, dim=1)

        if self.use_global_semantic:
            self.data_input = autograd.Variable(self.data_input)
            self.label_input = autograd.Variable(self.label_input)
            self.input = self.data_input

        self.patch_input = self.patch_data_input

    def cosine_schedule(
            self, step, max_steps, start_value, end_value):
        """
        Use cosine decay to gradually modify start_value to reach target end_value during iterations.

        Args:
            step:
                Current step number.
            max_steps:
                Total number of steps.
            start_value:
                Starting value.
            end_value:
                Target value.

        Returns:
            Cosine decay value.

        """
        if step < 0:
            raise ValueError("Current step number can't be negative")
        if max_steps < 1:
            raise ValueError("Total step number must be >= 1")

        if max_steps == 1:
            # Avoid division by zero
            decay = end_value
        elif step == max_steps:
            # Special case for Pytorch Lightning which updates LR scheduler also for epoch
            # after last training epoch.
            decay = end_value
        else:
            decay = (
                    end_value
                    - (end_value - start_value)
                    * (np.cos(np.pi * step / (max_steps - 1)) + 1)
                    / 2
            )
        return decay


    def update_momentum(self, model: nn.Module, sia_model: nn.Module):
        """Updates parameters of `model_ema` with Exponential Moving Average of `model`

        Momentum encoders are a crucial component fo models such as MoCo or BYOL.
        This helper function implements the momentum update of the encoder weights.
        """
        # with torch.no_grad():
        #     for model_ema, model in zip(model_ema.parameters(), model.parameters()):
        #         model_ema.data = model.data
        with torch.no_grad():
            model_state_dict = model.state_dict()
            sia_model_state_dict = sia_model.state_dict()
            for entry in sia_model_state_dict.keys():
                sia_param = sia_model_state_dict[entry].clone().detach()
                param = model_state_dict[entry].clone().detach()
                new_param = param * 1.
                sia_model_state_dict[entry] = new_param
            sia_model.load_state_dict(sia_model_state_dict)


    def update_momentum_ema(self, model: nn.Module, sia_model: nn.Module, m):
        """Updates parameters of `model_ema` with Exponential Moving Average of `model`

        Momentum encoders are a crucial component fo models such as MoCo or BYOL.
        This helper function implements the momentum update of the encoder weights.
        """
        # with torch.no_grad():
        #     for model_ema, model in zip(model_ema.parameters(), model.parameters()):
        #         model_ema.data = model.data
        with torch.no_grad():
            model_state_dict = model.state_dict()
            sia_model_state_dict = sia_model.state_dict()
            for entry in sia_model_state_dict.keys():
                sia_param = sia_model_state_dict[entry].clone().detach()
                param = model_state_dict[entry].clone().detach()
                # new_param = param * 1.
                new_param = (sia_param * m) + (param * (1. - m))
                sia_model_state_dict[entry] = new_param
            sia_model.load_state_dict(sia_model_state_dict)


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        if self.use_global_semantic:
            self.update_momentum(self.netG.sn_unet, self.netG.base_layers_global_momentum)
            # G is main, P is aux
            # self.pre_target,self.centerness_maps = self.netG(self.input,self.patch_input)

            # P is main, G is aux
            self.pre_target, self.centerness_maps = self.netG(self.patch_input, self.input)
        else:
            self.pre_target, _ = self.netG(self.patch_input)
        # sigmoid
        self.pre_target = torch.sigmoid(self.pre_target)


    def forward_ema(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.fake_B = self.netG(self.real_A)  # G(A)

        if self.use_global_semantic:
            if self.step <= self.cfg.global_warmup_step:
                self.update_momentum_ema(self.netG.sn_unet, self.netG.base_layers_global_momentum, 1)
            else:
                self.momentum_val = self.cosine_schedule(self.step, self.cfg.max_step, 0.999, 1)
                self.update_momentum_ema(self.netG.sn_unet, self.netG.base_layers_global_momentum, self.momentum_val)
            # G is main, P is aux
            # self.pre_target,self.centerness_maps = self.netG(self.input,self.patch_input)

            # P is main, G is aux
            self.pre_target, self.centerness_maps = self.netG(self.patch_input, self.input)
        else:
            self.pre_target, self.centerness_maps = self.netG(self.patch_input)
        # sigmoid

        self.pre_target = torch.sigmoid(self.pre_target)


    def save_model(self):
        # save generator
        torch.save(self.netG.state_dict(), os.path.join(self.log_folder, 'G_' + str(self.step) + '.pkl'))
        torch.save(self.netG, os.path.join(self.log_folder, 'G_' + str(self.step) + '.pth'))
        # save discriminator
        if self.cfg.use_GAN:
            torch.save(self.netD.state_dict(), os.path.join(self.log_folder, 'D_' + str(self.step) + '.pkl'))
            torch.save(self.netD, os.path.join(self.log_folder, 'D_' + str(self.step) + '.pth'))
        print("save model to {}".format(self.log_folder))


    def log(self, logger):
        logger.draw_prediction(self.pre_target, self.patch_label_input,self.centerness_maps,self.label_centerness_map, self.step)


    def get_loss_names(self, opt):
        self.loss_names = []
        if opt.use_GAN:
            self.loss_names.append('D_real')
            self.loss_names.append('D_fake')

            self.loss_names.append('D')
            self.loss_names.append('G_GAN')

        self.loss_names.append('G_BCE')

        self.loss_names.append('G_DICE')
        if opt.use_topo_loss:
            self.loss_names.append('G_topo')
        if opt.use_centerness:
            self.loss_names.append('G_centerness')
        if opt.use_av_cross:
            self.loss_names.append('G_av_cross')
        if opt.use_high_semantic:
            self.loss_names.append('G_high_semantic')
        self.loss_names.append('G_Hu')
        self.loss_names.append('G')

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))
        return errors_ret


    def test(self, result_folder):
        print("-----------start to test-----------")
        modelEvalution(self.step, self.netG.state_dict(),
                       result_folder,
                       use_cuda=self.cfg.use_cuda,
                       dataset=self.cfg.dataset,
                       input_ch=self.cfg.input_nc,
                       config=self.cfg,
                       strict_mode=True)
        print("---------------end-----------------")


    def backward_D(self, isBackward=True):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B

        opt = self.cfg

        # define the input of D
        real_input = torch.cat([self.patch_data_input, self.patch_label_input], dim=1)
        fake_input = torch.cat([self.patch_data_input, self.pre_target], dim=1)

        pred_real = self.netD(real_input)
        pred_fake = self.netD(fake_input.detach())  # bs x ch x (HxW)  b,1,(h*w)

        # Compute loss
        self.loss_D = 0

        # for GT
        label_shape = [opt.batch_size, 1, pred_real.shape[2]]
        # 0, 1
        label_fake = torch.zeros(label_shape)
        label_real = torch.ones(label_shape)

        if opt.use_cuda:
            label_real = label_real.cuda()
            label_fake = label_fake.cuda()
        if opt.GAN_type == 'vanilla':
            self.loss_D_real = self.criterionCE(pred_real, label_real)
            self.loss_D_fake = self.criterionCE(pred_fake, label_fake)

            self.loss_D = (self.loss_D_real + self.loss_D_fake)
            self.loss_D = self.loss_D * opt.lambda_GAN_D  # loss_D_fake_shuffle

        # backward
        if isBackward:
            self.loss_D.backward()


    def backward_G(self, isBackward=True):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        opt = self.cfg

        # define input
        fake_input_cpy = torch.cat([self.patch_data_input, self.pre_target], dim=1)
        self.loss_G = 0

        # GAN
        if opt.use_GAN:
            pred_fake = self.netD(fake_input_cpy)
            if opt.GAN_type == 'vanilla':
                ones_tensor = torch.ones([opt.batch_size, 1, pred_fake.shape[2]])
                if opt.use_cuda:
                    ones_tensor = ones_tensor.cuda()
                self.loss_G_GAN = opt.lambda_GAN_G * self.criterionCE(pred_fake, ones_tensor)
            self.loss_G += self.loss_G_GAN

        # BCE
        self.loss_G_BCE = opt.lambda_BCE * self.criterion(self.pre_target, self.patch_label_input)
        self.loss_G += self.loss_G_BCE
        self.loss_G_DICE = opt.lambda_DICE * self.criterionDICE(self.pre_target, self.patch_label_input)
        self.loss_G += self.loss_G_DICE
        if opt.use_av_cross:
            self.loss_G_av_cross = opt.lambda_BCE * self.criterionAV(self.coarse_target, self.patch_label_input)
            self.loss_G += self.loss_G_av_cross

        # topo loss
        if opt.use_topo_loss:
            self.loss_G_topo = vggloss(self.vggnet, self.criterionTopo, self.pre_target, self.patch_label_input,
                                       use_cuda=opt.use_cuda, weight_list=opt.lambda_topo_list)
            self.loss_G += opt.lambda_topo * self.loss_G_topo

        self.loss_G_Hu = 0.005 * self.criterionHu(self.pre_target, self.patch_label_input)
        self.loss_G += self.loss_G_Hu

        # centerness scores maps prediction
        if opt.use_centerness:

            # use centerness loss
            # 1. mask out the background first

            self.centerness_maps = self.centerness_maps * self.label_input_sm  # self.label_nodisk_map

            if opt.center_loss_type == 'centerness':
                # calculate V, the number of pixel
                # bs, ch, h, w = self.label_input.shape
                # v = bs*ch*h*w
                v1 = torch.sum(self.label_input_sm)

                self.loss_G_centerness = 0
                self.loss_G_centerness = centernessLoss(self.criterionSmoothL1, self.centerness_maps,
                                                        self.label_centerness_map_all, v1, weight=opt.lambda_centerness)

            self.loss_G += self.loss_G_centerness

            # backward

        if opt.use_high_semantic:
            self.loss_G_high_semantic = opt.lambda_high * self.criterionHighSemantic(self.coarse_target,
                                                                                     self.label_high_semantic)
            self.loss_G += self.loss_G_high_semantic

        if isBackward:
            self.loss_G.backward()


    def optimize_parameters(self):
        self.forward_ema()
        # self.forward()  # compute fake images: G(A)
        if self.use_GAN:
            # update D
            set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D()  # calculate gradients for D
            self.optimizer_D.step()  # update D's weights
            self.scheduler_D.step(self.step)
        # update G
        if self.use_GAN:
            set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
        self.scheduler_G.step(self.step)
