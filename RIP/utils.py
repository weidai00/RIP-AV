# -*- coding: utf-8 -*-


import os
import cv2
import sys
import json
import pickle
import random
import math
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from loss import multiLabelLossV2
import torch.nn.functional as F





def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = multiLabelLossV2()
    #loss_function=torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # ÀÛ¼ÆËðÊ§
    accu_num = torch.zeros(1).to(device)   # ÀÛ¼ÆÔ¤²âÕýÈ·µÄÑù±¾Êý
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        #loss = loss_function(pred, labels.to(device))
        loss,new_labels = loss_function(pred, labels.to(device))
        loss.backward()
        
        #pred = F.softmax(pred,dim=1)
        pred = F.sigmoid(pred)

        #pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred.ge(0.5).float(), new_labels.to(device)).all(dim=1).sum()


        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    #loss_function = torch.nn.CrossEntropyLoss()
    loss_function = multiLabelLossV2()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # ÀÛ¼ÆÔ¤²âÕýÈ·µÄÑù±¾Êý
    accu_loss = torch.zeros(1).to(device)  # ÀÛ¼ÆËðÊ§

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        loss,new_labels = loss_function(pred, labels.to(device))
        accu_loss += loss.detach()

        #pred = F.softmax(pred, dim=1)
        pred = F.sigmoid(pred)
        #pred_classes = torch.max(pred, dim=1)[1]
        #accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        accu_num += torch.eq(pred.ge(0.5).float(), new_labels.to(device)).all(dim=1).sum()
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num





def pre(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    log1 = []
    log2 = []
    model.eval()

    accu_num = torch.zeros(1).to(device)   # ÀÛ¼ÆÔ¤²âÕýÈ·µÄÑù±¾Êý
    accu_loss = torch.zeros(1).to(device)  # ÀÛ¼ÆËðÊ§
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        pred = F.softmax(pred, dim=1)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )
        log1.append(labels.tolist())
        log2.append(pred_classes.tolist())
    print('Finished Predict')
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num,log1,log2




def train_one_epoch_new(model_con,optimizer,data_loader,device,epoch,lr_scheduler):
    model_con.train()
    loss_function = torch.nn.CrossEntropyLoss()

    accu_loss = torch.zeros(1).to(device)  # ÀÛ¼ÆËðÊ§
    accu_num = torch.zeros(1).to(device)  # ÀÛ¼ÆÔ¤²âÕýÈ·µÄÑù±¾Êý


    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        optimizer.zero_grad()
        full_images, part_images, labels = data
        sample_num += full_images.shape[0]

        pred = model_con(full_images.to(device), part_images.to(device))

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train.py epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        lr_scheduler.step()
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())



