# -*- coding: utf-8 -*-

import os
import torch
import torchvision.datasets

from models.sw_gan import SW_GAN
from opt import Dataloader_general

import argparse
import imp
from Tools.utils import LogMaker

def train(opt, config_file):

    train_data, train_label_data, train_label_centerness_maps,train_data_mask,train_data_mask2 = \
        Dataloader_general(path=opt.trainset_path, use_centermap=True,
                           use_resize=opt.use_resize, resize_w_h=opt.resize_w_h)

    # train_data, train_label_data, train_label_centerness_maps,train_data_mask = \
    #     Dataloader_general(path=opt.trainset_path, use_centermap=True, use_CAM=opt.use_CAM,
    #                        use_resize=opt.use_resize, resize_w_h=opt.resize_w_h)

    # make log
    logger = LogMaker(opt, config_file)

    swgan = SW_GAN(opt)
    swgan.setup(opt, logger.log_folder)

    for i in range(opt.model_step, opt.max_step + 1):


        swgan.set_input(i,
                    train_data=train_data,
                    train_label_data=train_label_data,
                    train_label_data_centerness=train_label_centerness_maps,
                    train_data_mask=train_data_mask,
                    train_data_mask2 = train_data_mask2,
                    )


        swgan.optimize_parameters()

        if i % opt.save_iter == 0 :
            swgan.save_model()

        losses = swgan.get_current_losses()
        logger.write(i, losses)
        # draw the predicted images in summary
        if i % opt.display_iter == 0:
            swgan.log(logger)

        if i % opt.print_iter == 0:
            logger.print(losses, i)
            if i % opt.save_iter== 0 and i == 0:
                swgan.test(logger.result_folder)
                #swgan.test(logger.result_folder,dataset='DRIVE')
                #swgan.test(logger.result_folder, dataset='hrf')
                #swgan.test(logger.result_folder, dataset='LES')
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

    # torch.backends.cudnn.benchmark = True

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


    train(config, config_file)
