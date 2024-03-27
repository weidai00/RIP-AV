# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:25:18 2022

@author: Lenovo
"""
import numpy as np
import os
import json
import torch
import torch.optim as optim
from pandas.core.frame import DataFrame
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from data import ImageFolderCustom
from utils import get_params_groups, train_one_epoch, evaluate, setup_seed, create_lr_scheduler
from network_multi_label import MultiAV
from loss import multiLabelLoss
global log
log = []
import time


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    start_time = time.time()
    start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime(start_time))

    image_path ='./dataset'

    csv_path = f'./{start_time}.csv'
    batch_size = 64
    freeze_layers = False

    learning_rate = 5e-5
    weight_decay = 5e-4

    step_size = 50
    gamma = 0.5

    early_stop_step = 50
    epochs = 10000 
    best_acc = 0.5

    data_transform = {
        "train": transforms.Compose([transforms.Resize((256,256)),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomVerticalFlip(p=0.5),
                                     transforms.RandomRotation(180),
                                     #transforms.RandomAffine(degrees=(30, 70)),
                                     #transforms.RandomGrayscale(p=0.2),
                                     transforms.ColorJitter(0.1, 0.1, 0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                                     #transforms.Normalize([0.423, 0.423, 0.423], [0.146, 0.146, 0.146])]),
        "test": transforms.Compose([transforms.Resize((256,256)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
                                   #  transforms.Normalize([0.423, 0.423, 0.423], [0.146, 0.146, 0.146])]),}
    # ÊµÀý»¯ÑµÁ·Êý¾Ý¼¯
    # ÑµÁ·¼ÓÔØÊý¾Ý
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)  # È·±£Í¼Æ¬Â·¾¶ÎÞÎó
    train_dataset = ImageFolderCustom(os.path.join(image_path, "training"),[],
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    # ×ªÎªdataloaderÐÍ
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)
    # ¼ÓÔØÑéÖ¤Êý¾Ý¼¯
    validate_dataset =ImageFolderCustom(os.path.join(image_path, "test"),[],
                                         transform=data_transform["test"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    model = MultiAV(num_classes=2).to(device)

    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=weight_decay)
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = optim.Adam(pg, lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # Ã¿Ê®´Îµü´ú£¬Ñ§Ï°ÂÊ¼õ°ë


    # ÉèÖÃÔçÍ£
    total_batch = 0
    last_decrease = 0
    min_loss = 1000
    flag = False
    val_best_loss = np.inf
    train_best_loss = np.inf
    for epoch in range(epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch, )
        # validate
        val_loss,val_acc = evaluate(model=model,
                                     data_loader=validate_loader,
                                     device=device,
                                     epoch=epoch)
        lr_scheduler.step()



        # if val_loss < val_best_loss:
        #     torch.save(model.state_dict(), r"./weight/val_best_model_v3.pth")
        #     val_best_loss = val_loss

        # if train_loss < train_best_loss:
        #     torch.save(model.state_dict(), r"./weight/train_best_model_drive.pth")
        #     train_best_loss = train_loss

        if val_acc > best_acc:  # acc improve save weight
            best_acc = val_acc
            torch.save(model.state_dict(), r"./weight/train_best_acc_model_RAR_net.pth")

        if val_loss < min_loss:  # loss decrease save epoch
            min_loss = val_loss
            last_decrease = total_batch
            print((min_loss, last_decrease))
        total_batch += 1

        if total_batch - last_decrease > early_stop_step:
            print("No optimization for a long time, auto-stopping...")
            flag = True
            break
        log.append([epoch, train_loss, val_loss, train_acc, val_acc])
    print('Finished Training')
    data = DataFrame(data=log, columns=['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
    data.to_csv(csv_path)


if __name__ == '__main__':
    main()
