import os

import torch.autograd as autograd
import torch
import cv2
import numpy as np
from tqdm import tqdm
import os
from Tools.ImageResize import creatMask
from models.network import PGNet
from lib.Utils import *
from Tools.AVclassifiationMetrics import AVclassifiation


def modelEvalution_out(i,net,savePath, use_cuda=False, dataset='DRIVE', is_kill_border=True, input_ch=3, strict_mode=True,config=None):

    # path for images to save
    dataset_dict = {'out':'out', 'LES': 'LES_AV', 'DRIVE': 'AV_DRIVE', 'hrf': 'hrf'}
    dataset_name = dataset_dict[dataset]
    print(f'evaluating {dataset_name} dataset...')
    image_basename = sorted(os.listdir(f'./data/{dataset_name}/test/images'))

    image0 = cv2.imread(f'./data/{dataset_name}/test/images/{image_basename[0]}')

    data_path = os.path.join(savePath, dataset)

    if not os.path.exists(data_path):
        os.mkdir(data_path)
    print(f'num of test images: {len(image_basename)}')
    test_image_num = len(image_basename)
    #test_image_num = test_image_num//5
    test_image_height = image0.shape[0]
    test_image_width = image0.shape[1]
    if config.use_resize:
        test_image_height = config.reszie_w_h[1]
        test_image_width = config.reszie_w_h[0]
    ArteryPredAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    VeinPredAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    VesselPredAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    ProMap = np.zeros((test_image_num, 3, test_image_height, test_image_width), np.float32)
    MaskAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)

    #Vessel = VesselProMap('./data/AV_DRIVE/test/images')

    n_classes = 3
    Net = PGNet(resnet=config.use_network,use_global_semantic=True, input_ch=input_ch, num_classes= n_classes, use_cuda=use_cuda, pretrained=False, centerness=config.use_centerness, centerness_map_size=config.centerness_map_size)
    Net.load_state_dict(net)

    if use_cuda:
        Net.cuda()
    Net.eval()

    for k in tqdm(range(test_image_num)):
        ArteryPred,VeinPred,VesselPred,Mask = GetResult_out(Net,k,
                                                                                          use_cuda=use_cuda,
                                                                                          dataset_name=dataset_name,
                                                                                          is_kill_border=is_kill_border,
                                                                                          input_ch=input_ch,
                                                                                          config=config)
        ArteryPredAll[k,:,:,:] = ArteryPred
        VeinPredAll[k,:,:,:] = VeinPred
        VesselPredAll[k,:,:,:] = VesselPred

        MaskAll[k,:,:,:] = Mask


    ProMap[:,0,:,:] = ArteryPredAll[:,0,:,:]
    ProMap[:,1,:,:] = VeinPredAll[:,0,:,:]
    ProMap[:,2,:,:] = VesselPredAll[:,0,:,:]

    # filewriter = centerline_eval(ProMap, config)
    #np.save(os.path.join(savePath, "ProMap_testset.npy"), ProMap)
    AVclassifiation(savePath,ArteryPredAll,VeinPredAll,VesselPredAll,test_image_num,image_basename)


def GetResult_out(Net, k, use_cuda=False, dataset_name='DRIVE', is_kill_border=True, input_ch=3, config=None):
    image_basename = sorted(os.listdir(f'./data/{dataset_name}/test/images'))[k]
    # label_basename = sorted(os.listdir(f'./data/{dataset_name}/test/av'))[k]
    # assert image_basename.split('.')[0] == label_basename.split('.')[0]  # check if the image and label are matched

    ImgName = os.path.join(f'./data/{dataset_name}/test/images/', image_basename)
    # LabelName = os.path.join(f'./data/{dataset_name}/test/av/', label_basename)

    Img0 = cv2.imread(ImgName)
    # Label0 = cv2.imread(LabelName)
    _, Mask0 = creatMask(Img0, threshold=10)
    Mask = np.zeros((Img0.shape[0], Img0.shape[1]), np.float32)
    Mask[Mask0 > 0] = 1

    if config.use_resize:
        Img0 = cv2.resize(Img0, config.resize_w_h)
        # Label0 = cv2.resize(Label0, config.resize_w_h, interpolation=cv2.INTER_NEAREST)
        Mask = cv2.resize(Mask, config.resize_w_h, interpolation=cv2.INTER_NEAREST)

    Img = Img0
    height, width = Img.shape[:2]
    n_classes = 3
    patch_height = config.patch_size
    patch_width = config.patch_size
    stride_height = config.stride_height
    stride_width = config.stride_width

    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    # rgb2rgg

    Img = np.float32(Img / 255.)
    Img_enlarged = paint_border_overlap(Img, patch_height, patch_width, stride_height, stride_width)
    patch_size = config.patch_size
    batch_size = 8
    patches_imgs,_ = extract_ordered_overlap(Img_enlarged, patch_height, patch_width, stride_height, stride_width)
    patches_imgs = np.transpose(patches_imgs, (0, 3, 1, 2))
    patches_imgs = Normalize(patches_imgs)
    patchNum = patches_imgs.shape[0]
    max_iter = int(np.ceil(patchNum / float(batch_size)))
    if config.use_global_semantic:
        golbal_img = Img_enlarged.copy()
        golbal_img = cv2.resize(golbal_img, (config.patch_size, config.patch_size))
        golbal_img_batch = [golbal_img] * batch_size
        golbal_img_batch = np.stack(golbal_img_batch, axis=0)
        golbal_img_batch = np.transpose(golbal_img_batch, (0, 3, 1, 2))
        golbal_img_batch = Normalize(golbal_img_batch)
    pred_patches = np.zeros((patchNum, n_classes, patch_size, patch_size), np.float32)
    for i in range(max_iter):
        begin_index = i * batch_size
        end_index = (i + 1) * batch_size

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

        output_temp, _1, = Net(patches_input_temp1, global_input_temp1)

        pred_patches_temp = np.float32(output_temp.data.cpu().numpy())

        pred_patches_temp_sigmoid = sigmoid(pred_patches_temp)

        pred_patches[begin_index:end_index, :, :, :] = pred_patches_temp_sigmoid

        del patches_input_temp1
        del pred_patches_temp
        del patches_temp1
        del output_temp
        del pred_patches_temp_sigmoid

    new_height, new_width = Img_enlarged.shape[0], Img_enlarged.shape[1]
    pred_img = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)  # predictions
    pred_img = pred_img[:, 0:height, 0:width]
    if is_kill_border:
        pred_img = kill_border(pred_img, Mask)

    ArteryPred = np.float32(pred_img[0, :, :])
    VeinPred = np.float32(pred_img[2, :, :])
    VesselPred = np.float32(pred_img[1, :, :])

    ArteryPred = ArteryPred[np.newaxis, :, :]
    VeinPred = VeinPred[np.newaxis, :, :]
    VesselPred = VesselPred[np.newaxis, :, :]

    Mask = Mask[np.newaxis, :, :]

    return ArteryPred, VeinPred, VesselPred, Mask


def out_test(cfg):
    device = torch.device("cuda" if cfg.use_cuda else "cpu")
    model_root = cfg.model_path_pretrained_G
    model_path = os.path.join(model_root, 'G_' + str(cfg.model_step_pretrained_G) + '.pkl')
    net = torch.load(model_path, map_location=device)
    result_folder = os.path.join(model_root, 'running_result')
    modelEvalution_out(cfg.model_step_pretrained_G, net,
                   result_folder,
                   use_cuda=cfg.use_cuda,
                   dataset='DRIVE',
                   input_ch=cfg.input_nc,
                   config=cfg,
                   strict_mode=True)


if __name__ == '__main__':
    from config import config_test_general as cfg

    out_test(cfg)