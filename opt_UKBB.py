# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.autograd as autograd
import os
import cv2
from Tools.ImageResize import creatMask
from Tools.data_augmentation import data_aug5
from lib.Utils import *
from models.network import PGNet
from sklearn.metrics import roc_auc_score

from DRIVE_Evalution import Evalution_drive

from Tools.AVclassifiationMetrics import AVclassifiationMetrics_skeletonPixles
from scipy import ndimage


def get_patch_trad_5(batch_size, patch_size, train_data=None, train_label_data=None, train_label_data_centerness=None,train_patch_data=None,train_patch_label=None):
    data = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label_data_centerness = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    patch_data = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    patch_label = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)

    for j in range(batch_size):

        z = np.random.randint(0, train_data.shape[0])
        choice = np.random.randint(0, 6)
        patch_data_mat = train_patch_data[z, :, :, :]
        patch_label_mat = train_patch_label[z, :, :, :]

        if train_label_data_centerness is not None:
            label_centerness_mat = train_label_data_centerness[z, :, :, :]
        if train_data is not None:
            data_mat = train_data[z, :, :, :]
            label_mat = train_label_data[z, :, :, :]

        if train_label_data_centerness is not None and train_data is None:
            patch_data_mat, patch_label_mat, label_centerness_mat = data_aug5(patch_data_mat, patch_label_mat,
                                                                              label_centerness_mat, choice=choice)
        if train_label_data_centerness is not None and train_data is not None :
            patch_data_mat, patch_label_mat,label_centerness_mat, data_mat, label_mat = data_aug5(patch_data_mat, patch_label_mat,label_centerness_mat, data_mat,
                                                                             label_mat,
                                                                            choice=choice)
        if train_label_data_centerness is None and train_data is None:
            patch_data_mat, patch_label_mat = data_aug5(patch_data_mat, patch_label_mat, choice=choice)

        if train_label_data_centerness is None and train_data is not None:
            patch_data_mat, patch_label_mat, data_mat, label_mat = data_aug5(patch_data_mat, patch_label_mat, data_mat,
                                                                             label_mat,
                                                                             choice=choice)

        patch_data[j, :, :, :] = patch_data_mat
        patch_label[j, :, :, :] = patch_label_mat

        if train_label_data_centerness is not None:
            label_data_centerness[j, :, :, :] = label_centerness_mat
        if train_data is not None:
            data[j, :, :, :] = data_mat
            #label[j, :, :, :] = label_mat


    if train_label_data_centerness is not None and train_data is not None:
        return patch_data, patch_label, label_data_centerness, data, label
    if train_label_data_centerness is not None and train_data is None:
        return patch_data, patch_label, label_data_centerness
    if train_label_data_centerness is None and train_data is not None:
        return patch_data, patch_label, data, label
    if train_label_data_centerness is None and train_data is None:
        return patch_data, patch_label


import numpy as np
import albumentations as albu
from PIL import Image
def CAM(x):
    """
    :param dataset_path: 计算整个训练数据集的平均RGB通道值
    :param image:  array， 单张图片的array 形式
    :return: array形式的cam后的结果
    """
    # 每次使用新数据集时都需要重新计算前面的RBG平均值
    # RGB-->Rshift-->CLAHE
    resize=False
    R_mea_num, G_mea_num, B_mea_num = [], [], []
    dataset_path = 'data/ukbb/training/images/training_13.png'
    # for file in os.listdir(dataset_path):
    #
    #     image = np.array(Image.open(os.path.join(dataset_path, file)))
    #     # mea.append((file,np.mean(x[:,:,0])))
    #     R_mea_num.append(np.mean(image[:, :, 0]))
    #     G_mea_num.append(np.mean(image[:, :, 1]))
    #     B_mea_num.append(np.mean(image[:, :, 2]))
    # print(np.mean(R_mea_num), np.mean(G_mea_num), np.mean(B_mea_num))
    image = np.array(Image.open(dataset_path))
    R_mea_num.append(np.mean(image[:, :, 0]))
    G_mea_num.append(np.mean(image[:, :, 1]))
    B_mea_num.append(np.mean(image[:, :, 2]))
    mea2stand = int(np.mean(R_mea_num)-np.mean(x[:,:,0]))
    mea2standg = int(np.mean(G_mea_num)-np.mean(x[:,:,1]))
    mea2standb = int(np.mean(B_mea_num)-np.mean(x[:,:,2]))

    y = albu.RGBShift(r_shift_limit=(mea2stand,mea2stand),
        g_shift_limit=(mea2standg,mea2standg),
        b_shift_limit=(mea2standb,mea2standb),p=1,always_apply=True)(image=x)
    # y = ablu.FDA(reference_images=[reference_images],beta_limit=(1,1),p=1,read_fn=lambda x:x)(image=x)
    #y = albu.CLAHE(clip_limit=(2,2),p=1,always_apply=True)(image=np.uint8(y['image']))
    return y['image']

def Dataloader_general(path, use_CAM=False,use_centermap=False,use_resize=False,resize_w_h=(256,256),use_global_semantic = False):
    use_resize = True
    ImgPath = path + "images/"
    LabelPath = path + "av/"
    centPath = path + "centerness_maps/"
    PatchImagePath = path + "image_patch/"
    PatchLabelPath = path + "av_patch/"

    if use_resize:
        resize_w_h = resize_w_h

    ImgDir = os.listdir(ImgPath)
    LabelDir = os.listdir(LabelPath)
    PatchImageDir = os.listdir(PatchImagePath)

    Img0 = cv2.imread(ImgPath + ImgDir[0])
    Label0 = cv2.imread(LabelPath + LabelDir[0])
    image_suffix = os.path.splitext(ImgDir[0])[1]
    label_suffix = os.path.splitext(LabelDir[0])[1]
    if use_resize:
        Img0 = cv2.resize(Img0, resize_w_h)
        Label0 = cv2.resize(Label0, resize_w_h)

    Img = np.zeros((len(PatchImageDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)
    Label = np.zeros((len(PatchImageDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)
    Label_center = np.zeros((len(PatchImageDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)
    Label_patch = np.zeros((len(PatchImageDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)
    Img_patch = np.zeros((len(PatchImageDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)

    for i,name in enumerate(PatchImageDir):


        prefix = os.path.splitext(name)[0].split('-')[0]


        LabelArtery = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        LabelVein = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        LabelVessel = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)


        Img0 = cv2.imread(os.path.join(ImgPath, f'{prefix}{image_suffix}'))
        Label0 = cv2.imread(os.path.join(LabelPath,f'{prefix}{label_suffix}'))

        if use_resize:
            Img0 = cv2.resize(Img0, resize_w_h)
            Label0 = cv2.resize(Label0, resize_w_h, interpolation=cv2.INTER_NEAREST)


        LabelArtery[(Label0[:, :, 2] == 255) | (Label0[:, :, 1] == 255)] = 1
        LabelArtery[(Label0[:, :, 2] == 255) & (Label0[:, :, 1] == 255) & (Label0[:, :, 0] == 255)] = 0

        LabelVein[(Label0[:, :, 1] == 255) | (Label0[:, :, 0] == 255)] = 1
        LabelVein[(Label0[:, :, 2] == 255) & (Label0[:, :, 1] == 255) & (Label0[:, :, 0] == 255)] = 0

        LabelVessel[(Label0[:, :, 2] == 255) | (Label0[:, :, 1] == 255) | (Label0[:, :, 0] == 255)] = 1


        ImgCropped = cv2.cvtColor(Img0, cv2.COLOR_BGR2RGB)

        if use_CAM:
            ImgCropped = CAM(ImgCropped)

        ImgCropped = np.float32(ImgCropped / 255.)


        Img[i, :, :, :] = np.transpose(ImgCropped, (2, 0, 1)) # HWC to CHW

        Label[i, 0, :, :] = LabelArtery
        Label[i, 1, :, :] = LabelVein
        Label[i, 2, :, :] = LabelVessel

    for i, name in enumerate(PatchImageDir):

        prefix = os.path.splitext(name)[0]
        LabelArtery = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        LabelVein = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        LabelVessel = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        Img0 = cv2.imread(os.path.join(PatchImagePath, f'{prefix}.png'))
        Label0 = cv2.imread(os.path.join(PatchLabelPath, f'{prefix}{label_suffix}'))
        if use_resize:
            Img0 = cv2.resize(Img0, resize_w_h)
            Label0 = cv2.resize(Label0, resize_w_h, interpolation=cv2.INTER_NEAREST)

        LabelArtery[(Label0[:, :, 2] == 255) | (Label0[:, :, 1] == 255)] = 1
        LabelArtery[(Label0[:, :, 2] == 255) & (Label0[:, :, 1] == 255) & (Label0[:, :, 0] == 255)] = 0

        LabelVein[(Label0[:, :, 1] == 255) | (Label0[:, :, 0] == 255)] = 1
        LabelVein[(Label0[:, :, 2] == 255) & (Label0[:, :, 1] == 255) & (Label0[:, :, 0] == 255)] = 0

        LabelVessel[(Label0[:, :, 2] == 255) | (Label0[:, :, 1] == 255) | (Label0[:, :, 0] == 255)] = 1

        if use_centermap and use_global_semantic:
            Labelcenter_artery = LabelArtery.copy()
            Labelcenter_vein = LabelVein.copy()
            Labelcenter_vessel = LabelVessel.copy()

            center_img_a = ndimage.distance_transform_edt(Labelcenter_artery)
            max_dis_a = center_img_a.max()
            min_dis_a = center_img_a.min()
            if max_dis_a!=0 or min_dis_a!=0:
                center_img_a = (center_img_a - min_dis_a) / (max_dis_a - min_dis_a)

            center_img_v = ndimage.distance_transform_edt(Labelcenter_vein)
            max_dis_v = center_img_v.max()
            min_dis_v = center_img_v.min()
            if max_dis_v!=0 or min_dis_v!=0:
                center_img_v = (center_img_v - min_dis_v) / (max_dis_v - min_dis_v)

            center_img_vessel = ndimage.distance_transform_edt(Labelcenter_vessel)
            max_dis_vessel = center_img_vessel.max()
            min_dis_vessel = center_img_vessel.min()
            if max_dis_vessel!=0 or min_dis_vessel!=0:
                center_img_vessel = (center_img_vessel - min_dis_vessel) / (max_dis_vessel - min_dis_vessel)

            Label_center[i, 0, :, :] = center_img_a
            Label_center[i, 1, :, :] = center_img_v
            Label_center[i, 2, :, :] = center_img_vessel

        ImgCropped = cv2.cvtColor(Img0, cv2.COLOR_BGR2RGB)

        if use_CAM:
            ImgCropped = CAM(ImgCropped)

        ImgCropped = np.float32(ImgCropped / 255.)

        Img_patch[i, :, :, :] = np.transpose(ImgCropped, (2, 0, 1))  # HWC to CHW

        Label_patch[i, 0, :, :] = LabelArtery
        Label_patch[i, 1, :, :] = LabelVein
        Label_patch[i, 2, :, :] = LabelVessel
    Img = Normalize(Img)
    Img_patch = Normalize(Img_patch)
    if use_centermap and not use_global_semantic:
        return Img_patch, Label_patch,Label_center
    if use_centermap and use_global_semantic:
        return Img_patch,Label_patch,Label_center,Img, Label
    if not use_centermap and not use_global_semantic:
        return Img_patch, Label_patch
    if not use_centermap and use_global_semantic:
        return Img_patch, Label_patch, Img, Label
    return Img_patch, Label_patch, Label_center


def modelEvalution(i,net,savePath, use_cuda=False, dataset='DRIVE', is_kill_border=True, input_ch=3, strict_mode=True,config=None):

    # path for images to save
    dataset_dict = {'IOSTSAR':'IOSTSAR','LES':'LES_AV','DRIVE': 'AV_DRIVE', 'hrf': 'hrf', 'ukbb': 'ukbb','all_combine':'all_combine'}
    dataset_name = dataset_dict[dataset]

    image_basename = sorted(os.listdir(f'./data/{dataset_name}/test/images'))
    label_basename = sorted(os.listdir(f'./data/{dataset_name}/test/av'))
    assert len(image_basename) == len(label_basename)

    image0 = cv2.imread(f'./data/{dataset_name}/test/images/{image_basename[0]}')



    data_path = os.path.join(savePath, dataset)
    metrics_file_path = os.path.join(savePath, 'metrics.txt')#_'+str(config.model_step_pretrained_G)+'.txt')
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    test_image_num = len(image_basename)
    test_image_height = image0.shape[0]
    test_image_width = image0.shape[1]
    if config.use_resize:
        test_image_height = config.resize_w_h[1]
        test_image_width = config.resize_w_h[0]
    ArteryPredAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    VeinPredAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    VesselPredAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    LabelArteryAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    LabelVeinAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    LabelVesselAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    ProMap = np.zeros((test_image_num, 3, test_image_height, test_image_width), np.float32)
    LabelMap = np.zeros((test_image_num, 3, test_image_height, test_image_width), np.float32)
    MaskAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)

    #Vessel = VesselProMap('./data/AV_DRIVE/test/images')

    n_classes = 3
    if config.use_global_semantic:
        Net = PGNet(resnet=config.use_network, input_ch=input_ch, num_classes= n_classes, use_cuda=use_cuda, pretrained=False, centerness=config.use_centerness, centerness_map_size=config.centerness_map_size,use_global_semantic=config.use_global_semantic)
    else:
        Net = PGNet(resnet=config.use_network, input_ch=input_ch, num_classes= n_classes, use_cuda=use_cuda, pretrained=False, centerness=config.use_centerness, centerness_map_size=config.centerness_map_size)

    Net.load_state_dict(net)

    if use_cuda:
        Net.cuda()
    Net.eval()

    for k in tqdm(range(test_image_num)):
        ArteryPred,VeinPred,VesselPred,LabelArtery,LabelVein,LabelVessel,Mask = GetResult(Net,k,
                                                                                          use_cuda=use_cuda,
                                                                                          dataset_name=dataset_name,
                                                                                          is_kill_border=is_kill_border,
                                                                                          input_ch=input_ch,
                                                                                          config=config)
        ArteryPredAll[k,:,:,:] = ArteryPred
        VeinPredAll[k,:,:,:] = VeinPred
        VesselPredAll[k,:,:,:] = VesselPred
        LabelArteryAll[k,:,:,:] = LabelArtery
        LabelVeinAll[k,:,:,:] = LabelVein
        LabelVesselAll[k,:,:,:] = LabelVessel

        MaskAll[k,:,:,:] = Mask


    ProMap[:,0,:,:] = ArteryPredAll[:,0,:,:]
    ProMap[:,1,:,:] = VeinPredAll[:,0,:,:]
    ProMap[:,2,:,:] = VesselPredAll[:,0,:,:]
    LabelMap[:,0,:,:] = LabelArteryAll[:,0,:,:]
    LabelMap[:,1,:,:] = LabelVeinAll[:,0,:,:]
    LabelMap[:,2,:,:] = LabelVesselAll[:,0,:,:]

    VesselAUC,VesselAcc,VesselSp,VesselSe,VesselF1,VesselDice,VesselIou = Evalution_drive(VesselPredAll, LabelVesselAll, MaskAll)

    #filewriter = centerline_eval(ProMap, config)
    np.save(os.path.join(savePath, "ProMap_testset.npy"),ProMap)
    np.save(os.path.join(savePath, "LabelMap_testset.npy"), LabelMap)
    np.save(os.path.join(savePath, "MaskAll_testset.npy"), MaskAll)
    ArteryAUC,AveAcc,VeinAcc,ArteryAcc,AveF1,AveDice,AveIou,VeinAUC = Evalution_AV_skeletonPixles(ArteryPredAll,VeinPredAll,VesselPredAll,LabelArteryAll,LabelVeinAll,LabelVesselAll,MaskAll,test_image_num,strict_mode=strict_mode)

    for k in range(0,test_image_num):

        #a_sofmax,v_sofmax = AV_based_vessel(ArteryPredAll[k,0,:,:],VeinPredAll[k,0,:,:],VesselPredAll[k,0,:,:])
        cv2.imwrite(os.path.join(data_path, f"{dataset}_Artery_"+str(k).zfill(3)+".png"),ArteryPredAll[k,0,:,:]*255)

        cv2.imwrite(os.path.join(data_path, f"{dataset}_Vein"+str(k).zfill(3)+".png"),VeinPredAll[k,0,:,:]*255)
        cv2.imwrite(os.path.join(data_path, f"{dataset}_Vessel"+str(k).zfill(3)+".png"),VesselPredAll[k,0,:,:]*255)
        # draw the thresh > 0.5转int
        cv2.imwrite(os.path.join(data_path, f"{dataset}_Artery_"+str(k).zfill(3)+"_thresh.png"),(ArteryPredAll[k,0,:,:]>0.5)*255)
        cv2.imwrite(os.path.join(data_path, f"{dataset}_Vein"+str(k).zfill(3)+"_thresh.png"),(VeinPredAll[k,0,:,:]>=0.5)*255)
        cv2.imwrite(os.path.join(data_path, f"{dataset}_Vessel"+str(k).zfill(3)+"_thresh.png"),(VesselPredAll[k,0,:,:]>=0.5)*255)
        # # draw the artery and vein base on vessel
        # cv2.imwrite(os.path.join(data_path, f"{dataset}_Artery_"+str(k).zfill(3)+"_softmax.png"),(a_sofmax>=0.5)*255)
        # cv2.imwrite(os.path.join(data_path, f"{dataset}_Vein"+str(k).zfill(3)+"_softmax.png"),(v_sofmax>=0.5)*255)

    print(f"========================={dataset}=============================")
    print("Strict mode:{}".format(strict_mode))
    print(f"The {i} step Average Acc is:{AveAcc}")
    print(f"The {i} step Average F1 is:{AveF1}")
    print(f"The {i} step Average Dice is:{AveDice}")
    print(f"The {i} step Average Iou is:{AveIou}")
    print("-----------------------------------------------------------")
    print(f"The {i} step Artery AUC is:{ArteryAUC}")
    print(f"The {i} step Artery Acc is:{ArteryAcc}")
    print("-----------------------------------------------------------")
    print(f"The {i} step Vein AUC is:{VeinAUC}")
    print(f"The {i} step Vein Acc is:{VeinAcc}")
    print("-----------------------------------------------------------")
    print(f"The {i} step Vessel AUC is:{VesselAUC}")
    print(f"The {i} step Vessel Acc is:{VesselAcc}")
    print(f"The {i} step Vessel Sens is:{VesselSe}")
    print(f"The {i} step Vessel Spec is:{VesselSp}")
    print(f"The {i} step Vessel F1 is:{VesselF1}")
    print(f"The {i} step Vessel Dice is:{VesselDice}")
    print(f"The {i} step Vessel Iou is:{VesselIou}")
    print("-----------------------------------------------------------")

    if not os.path.exists(metrics_file_path):
         file_w = open(metrics_file_path,'w')
    file_w = open(metrics_file_path,'r+')
    file_w.read()
    file_w.write(f"========================={dataset}=============================" + '\n' +
                 "Strict mode:{}".format(strict_mode) + '\n' +
                 f"The {i} step Average Acc is:{AveAcc}" + '\n' +
                 f"The {i} step Average F1 is:{AveF1}" + '\n' +
                 f"The {i} step Average Dice is:{AveDice}" + '\n' +
                 f"The {i} step Average Iou is:{AveIou}" + '\n' +
                 "-----------------------------------------------------------" + '\n' +
                 f"The {i} step Artery AUC is:{ArteryAUC}" + '\n' +
                 f"The {i} step Artery Acc is:{ArteryAcc}" + '\n' +
                 "-----------------------------------------------------------" + '\n' +
                 f"The {i} step Vein AUC is:{VeinAUC}" + '\n' +
                 f"The {i} step Vein Acc is:{VeinAcc}" + '\n' +
                 "-----------------------------------------------------------" + '\n' +
                 f"The {i} step Vessel AUC is:{VesselAUC}" + '\n' +
                 f"The {i} step Vessel Acc is:{VesselAcc}" + '\n' +
                 f"The {i} step Vessel Sens is:{VesselSe}" + '\n' +
                 f"The {i} step Vessel Spec is:{VesselSp}" + '\n' +
                 f"The {i} step Vessel F1 is:{VesselF1}" + '\n' +
                 f"The {i} step Vessel Dice is:{VesselDice}" + '\n' +
                 f"The {i} step Vessel Iou is:{VesselIou}" + '\n' +
                 "-----------------------------------------------------------" + '\n')
    #file_w.write(filewriter)
    file_w.close()


def GetResult(Net, k, use_cuda=False, dataset_name='DRIVE', is_kill_border=True, input_ch=3, config=None):

    image_basename = sorted(os.listdir(f'./data/{dataset_name}/test/images'))[k]
    label_basename = sorted(os.listdir(f'./data/{dataset_name}/test/av'))[k]
    assert image_basename.split('.')[0] == label_basename.split('.')[0]  # check if the image and label are matched


    ImgName = os.path.join(f'./data/{dataset_name}/test/images/',image_basename)
    LabelName = os.path.join(f'./data/{dataset_name}/test/av/',label_basename)

    Img0 = cv2.imread(ImgName)
    Label0 = cv2.imread(LabelName)
    _,Mask0 = creatMask(Img0,threshold=10)
    Mask = np.zeros((Img0.shape[0],Img0.shape[1]),np.float32)
    Mask[Mask0>0] = 1

    if config.use_resize:
        Img0 = cv2.resize(Img0, config.resize_w_h)
        Label0 = cv2.resize(Label0, config.resize_w_h, interpolation=cv2.INTER_NEAREST)
        Mask = cv2.resize(Mask,config.resize_w_h,interpolation=cv2.INTER_NEAREST)

    LabelArtery = np.zeros((Label0.shape[0], Label0.shape[1]), np.float32)
    LabelVein = np.zeros((Label0.shape[0], Label0.shape[1]), np.float32)
    LabelVessel = np.zeros((Label0.shape[0], Label0.shape[1]), np.float32)
    LabelArtery[(Label0[:,:,2]>=128)|(Label0[:,:,1]>=128)] = 1
    LabelArtery[(Label0[:,:,2]>=128)&(Label0[:,:,1]>=128)&(Label0[:,:,0]>=128)] = 0
    LabelVein[(Label0[:,:,1]>=128)|(Label0[:,:,0]>=128)] = 1
    LabelVein[(Label0[:,:,2]>=128)&(Label0[:,:,1]>=128)&(Label0[:,:,0]>=128)] = 0
    LabelVessel[(Label0[:,:,2]>=128)|(Label0[:,:,1]>=128)|(Label0[:,:,0]>=128)] = 1

    Img = Img0
    height, width = Img.shape[:2]
    n_classes = 3
    patch_height = config.patch_size
    patch_width = config.patch_size
    stride_height = config.stride_height
    stride_width = config.stride_width
    
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    # rgb2rgg
    if config.use_CAM:
        Img = CAM(Img)
    Img = np.float32(Img/255.)
    Img_enlarged = paint_border_overlap(Img, patch_height, patch_width, stride_height, stride_width)
    patch_size = config.patch_size
    batch_size = 4
    patches_imgs = extract_ordered_overlap(Img_enlarged, patch_height, patch_width, stride_height, stride_width)
    patches_imgs = np.transpose(patches_imgs,(0,3,1,2))
    patches_imgs = Normalize(patches_imgs)
    patchNum = patches_imgs.shape[0]
    max_iter = int(np.ceil(patchNum/float(batch_size)))
    if config.use_global_semantic:
        golbal_img = Img_enlarged.copy()
        golbal_img = cv2.resize(golbal_img, (config.patch_size, config.patch_size))
        golbal_img_batch = [golbal_img] * batch_size
        golbal_img_batch = np.stack(golbal_img_batch, axis=0)
        golbal_img_batch = np.transpose(golbal_img_batch, (0, 3, 1, 2))
        golbal_img_batch = Normalize(golbal_img_batch)
    pred_patches = np.zeros((patchNum, n_classes, patch_size, patch_size), np.float32)
    for i in range(max_iter):
        begin_index = i*batch_size
        end_index = (i+1)*batch_size
    
        patches_temp1 = patches_imgs[begin_index:end_index, :, :, :]

        patches_input_temp1 = torch.FloatTensor(patches_temp1)
        if config.use_global_semantic:
            global_temp1 = golbal_img_batch[0:patches_temp1.shape[0], :, :, :]
            global_input_temp1 = torch.FloatTensor(global_temp1)
        if use_cuda:
            patches_input_temp1 = autograd.Variable(patches_input_temp1.cuda())
            if config.use_global_semantic:
                global_input_temp1 = autograd.Variable(global_input_temp1.cuda())
        else:
            patches_input_temp1 = autograd.Variable(patches_input_temp1)
            if config.use_global_semantic:
                global_input_temp1 = autograd.Variable(global_input_temp1)
               
        output_temp,_1, = Net(patches_input_temp1,global_input_temp1)

        pred_patches_temp = np.float32(output_temp.data.cpu().numpy())

        pred_patches_temp_sigmoid = sigmoid(pred_patches_temp)
    
        pred_patches[begin_index:end_index, :,:,:] = pred_patches_temp_sigmoid
    
        del patches_input_temp1
        del pred_patches_temp
        del patches_temp1
        del output_temp
        del pred_patches_temp_sigmoid
    
    
    new_height, new_width = Img_enlarged.shape[0], Img_enlarged.shape[1]
    pred_img = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)  # predictions
    pred_img = pred_img[:,0:height,0:width]
    if is_kill_border:
        pred_img = kill_border(pred_img, Mask)
    
    ArteryPred = np.float32(pred_img[0,:,:])
    VeinPred = np.float32(pred_img[2,:,:])
    VesselPred = np.float32(pred_img[1,:,:])
    
    ArteryPred = ArteryPred[np.newaxis,:,:]
    VeinPred = VeinPred[np.newaxis,:,:]
    VesselPred = VesselPred[np.newaxis,:,:]
    LabelArtery = LabelArtery[np.newaxis,:,:]
    LabelVein = LabelVein[np.newaxis,:,:]
    LabelVessel = LabelVessel[np.newaxis,:,:]
    Mask = Mask[np.newaxis,:,:]
    
    
    return ArteryPred,VeinPred,VesselPred,LabelArtery,LabelVein,LabelVessel,Mask


def Evalution_AV_skeletonPixles(PredAll1,PredAll2,VesselPredAll,LabelAll1,LabelAll2, LabelVesselAll,MaskAll,DataSet=0,onlyMeasureSkeleton=False,strict_mode=False):
    """
    OUTPUT:
    AUC1: 动脉
    AUC2: 静脉
    accuracy1: 动静脉联合准确率
    specificity1: 静脉准确率
    sensitivity1: 动脉准确率
    f_score1: 动静脉联合F1
    dice_score1: 动静脉联合Dice
    iou_score1: 动静脉联合IoU

    """

    threshold_confusion = 0.5
    y_scores1, y_true1,y_scores2, y_true2 = pred_only_FOV_AV(PredAll1,PredAll2,LabelAll1,LabelAll2, MaskAll,threshold_confusion)
    AUC1 = roc_auc_score(y_true1,y_scores1)   # 动脉
    AUC2 = roc_auc_score(y_true2,y_scores2)  # 静脉
    if onlyMeasureSkeleton:
        accuracy1,specificity1,sensitivity1,f_score1,dice_score1,iou_score1, bad_case_index = AVclassifiationMetrics_skeletonPixles(PredAll1,PredAll2,VesselPredAll,LabelAll1,LabelAll2,LabelVesselAll,DataSet,onlyMeasureSkeleton=onlyMeasureSkeleton,strict_mode=strict_mode)

    else:
        accuracy1,specificity1,sensitivity1,f_score1,dice_score1,iou_score1  = AVclassifiationMetrics_skeletonPixles(PredAll1,PredAll2,VesselPredAll,LabelAll1,LabelAll2,LabelVesselAll,DataSet,onlyMeasureSkeleton=onlyMeasureSkeleton,strict_mode=strict_mode)
    #accuracy2,specificity2,sensitivity2 = AVclassifiationMetrics(PredAll2,PredAll1,VesselPredAll,LabelAll2,LabelAll1,LabelVesselAll,DataSet)
    return AUC1, accuracy1, specificity1,sensitivity1, f_score1, dice_score1, iou_score1, AUC2

def AV_based_vessel(PredAll1,PredAll2,VesselPredAll):


    height, width = PredAll1.shape

    VesselProb = VesselPredAll

    ArteryProb = PredAll1
    VeinProb = PredAll2



    vesselPixels = np.where(VesselProb >= 0.5)

    ArteryProb2 = np.zeros((height, width))
    VeinProb2 = np.zeros((height, width))
    for i in range(len(vesselPixels[0])):
        row = vesselPixels[0][i]
        col = vesselPixels[1][i]
        probA = ArteryProb[row, col]
        probV = VeinProb[row, col]

        tmp = np.exp(np.array([probA, probV]) - np.max(np.array([probA, probV])))

        softmaxProb = tmp/tmp.sum()
        ArteryProb2[row, col] = softmaxProb[0]
        VeinProb2[row, col] = softmaxProb[1]
    return ArteryProb2,VeinProb2

def draw_prediction(writer, pred, targs, step):
    target_artery = targs[0:1,0,:,:]
    target_vein   = targs[0:1,1,:,:]
    target_all    = targs[0:1,2,:,:]
    
    pred_sigmoid = pred #nn.Sigmoid()(pred)
    
    writer.add_image('artery',  torch.cat([pred_sigmoid[0:1,0,:,:], target_artery], dim=1), global_step=step)
    writer.add_image('vessel',  torch.cat([pred_sigmoid[0:1,1,:,:], target_vein   ], dim=1), global_step=step)
    writer.add_image('vein',    torch.cat([pred_sigmoid[0:1,2,:,:], target_all  ], dim=1), global_step=step)
    

if __name__ == '__main__':
    import pathlib

    i = sorted(os.listdir(f'./data/AV_DRIVE/test/images'), reverse=False)[0]

    print(i)
