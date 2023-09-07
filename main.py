# -*- coding: utf-8 -*-

import os
import torch
import torchvision.datasets

from models.sw_gan import SW_GAN
from opt_UKBB import Dataloader_general

import argparse
import imp
from Tools.utils import LogMaker

def train(opt, config_file):
    # load dataset [BCHW, BCHW, BCHW]

    if opt.use_centerness and not opt.use_global_semantic:
        train_patch_data, train_patch_label, train_label_centerness_maps = \
            Dataloader_general(path=opt.trainset_path, use_centermap=True, use_CAM=opt.use_CAM,
                               use_resize=opt.use_resize, resize_w_h=opt.resize_w_h)
    if opt.use_centerness and opt.use_global_semantic:
        train_patch_data, train_patch_label, train_label_centerness_maps, train_data,train_label_data = \
            Dataloader_general(path=opt.trainset_path, use_centermap=True, use_CAM=opt.use_CAM,
                               use_resize=opt.use_resize, resize_w_h=opt.resize_w_h, use_global_semantic=True)

    if not opt.use_centerness and not opt.use_global_semantic:
        train_patch_data, train_patch_label = \
            Dataloader_general(path=opt.trainset_path,  use_CAM=opt.use_CAM,
                               use_resize=opt.use_resize, resize_w_h=opt.resize_w_h)

    if not opt.use_centerness and opt.use_global_semantic:
        train_patch_data, train_patch_label, train_data,train_label_data = \
            Dataloader_general(path=opt.trainset_path,  use_CAM=opt.use_CAM,
                               use_resize=opt.use_resize, resize_w_h=opt.resize_w_h, use_global_semantic=True)
    # make log
    logger = LogMaker(opt, config_file)

    swgan = SW_GAN(opt)
    swgan.setup(opt, logger.log_folder)

    for i in range(opt.model_step, opt.max_step + 1):

        if opt.use_centerness and not opt.use_global_semantic:
            swgan.set_input(i,
                        train_patch_data=train_patch_data,
                        train_patch_label=train_patch_label,
                        train_label_centerness_maps=train_label_centerness_maps
                        )
        if opt.use_centerness and opt.use_global_semantic:
            swgan.set_input(i,
                        train_patch_data=train_patch_data,
                        train_patch_label=train_patch_label,
                        train_label_data_centerness=train_label_centerness_maps,
                        train_data=train_data,
                        train_label_data=train_label_data
                        )
        if not opt.use_centerness and not opt.use_global_semantic:
            swgan.set_input(i,
                        train_patch_data=train_patch_data,
                        train_patch_label=train_patch_label
                        )
        if not opt.use_centerness and opt.use_global_semantic:
            swgan.set_input(i,
                        train_patch_data=train_patch_data,
                        train_patch_label=train_patch_label,
                        train_data=train_data,
                        train_label_data=train_label_data
                        )

        swgan.optimize_parameters()

        if i % opt.save_iter == 0 :
            swgan.save_model()
        # if i % 1 == 0 or (i >= opt.first_display_metric_iter and i % opt.print_iter == 0):
        #     swgan.save_model()
        losses = swgan.get_current_losses()
        logger.write(i, losses)
        # draw the predicted images in summary
        if i % opt.display_iter == 0:
            swgan.log(logger)

        if i % opt.print_iter == 0:
            logger.print(losses, i)
            if i % 5000== 0 and i == 0:
                swgan.test(logger.result_folder)
        # if i >= 1:
        #     swgan.test(logger.result_folder)
    logger.writer.close()

import numpy as np
import random
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    set_seed(4)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("config_file", help="The path to the training configuration file",
    #                 type=str, default='default')
    # args = parser.parse_args()
    # config = imp.load_source('config', args.config_file)
    #
    # train(config, args.config_file)

    config_file = 'config/config_train_general.py'
    config = imp.load_source('config', config_file)


    train(config, config_file);
