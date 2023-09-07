# -*- coding: utf-8 -*-
"""
@author: DW
Code for generate shuffled augmentation for batch image.
"""
import numpy as np
import cv2
import os
import config.config_train_general as cfg
from Tools.ImageResize import creatMask

def patchJudge(mask,a,v,av, y, x, patch_h, patch_w,type='only_a',limit=100):
    '''
    :param a: artery
    :param v: vein
    :param av: artery and vein
    :param y: y coordinate of the patch
    :param x: x coordinate of the patch
    :param patch_h: height of the patch
    :param patch_w: width of the patch
    :param type: 'only_a', 'only_v', 'only_av', 'a_or_v'
    :param limit: the limit of the number of pixels in the patch
    0 represents only_a
    1 represents only_v
    2 represents only_av
    3 represents a_or_v

    '''
    # print(type)
    # print(patch_w)
    # print(limit)
    # print(f'av:{np.sum(av[y:y + patch_h, x:x + patch_w])}')
    # print(f'a:{np.sum(a[y:y + patch_h, x:x+patch_w])}')
    # print(f'v:{np.sum(v[y:y + patch_h, x:x + patch_w])}')

    b_rate = np.sum(mask[y-patch_h//2:y+patch_h//2,x-patch_w//2:x+patch_w//2] == 0)/patch_h/patch_w
    # print(b_rate)
    b_rate_threshold = 0.1
    if type=='no':
        return True

    if type ==0 or type=='only_a':
        return np.sum(a[y-patch_h//2:y+patch_h//2,x-patch_w//2:x+patch_w//2])>=limit and np.sum(v[y-patch_h//2:y+patch_h//2,x-patch_w//2:x+patch_w//2])==0 and b_rate<=b_rate_threshold
    elif type ==1 or type=='only_v':
        return np.sum(v[y-patch_h//2:y+patch_h//2,x-patch_w//2:x+patch_w//2])>=limit and np.sum(a[y-patch_h//2:y+patch_h//2,x-patch_w//2:x+patch_w//2])==0  and b_rate<=b_rate_threshold
    elif type ==2 or type=='only_av':
        return np.sum(av[y-patch_h//2:y+patch_h//2,x-patch_w//2:x+patch_w//2])>=limit and np.sum(a[y-patch_h//2:y+patch_h//2,x-patch_w//2:x+patch_w//2])>0 and np.sum(v[y-patch_h//2:y+patch_h//2,x-patch_w//2:x+patch_w//2])>0 and b_rate<=b_rate_threshold and (np.sum(a[y-patch_h//2:y+patch_h//2,x-patch_w//2:x+patch_w//2])/np.sum(v[y-patch_h//2:y+patch_h//2,x-patch_w//2:x+patch_w//2])>0.2 and np.sum(a[y-patch_h//2:y+patch_h//2,x-patch_w//2:x+patch_w//2])/np.sum(v[y-patch_h//2:y+patch_h//2,x-patch_w//2:x+patch_w//2])<5)

    if type ==3 or type=='only_black':
        return np.sum(av[y-patch_h//2:y+patch_h//2,x-patch_w//2:x+patch_w//2])==0 and np.sum(v[y-patch_h//2:y+patch_h//2,x-patch_w//2:x+patch_w//2])==0 and np.sum(a[y-patch_h//2:y+patch_h//2,x-patch_w//2:x+patch_w//2])==0 and b_rate<=b_rate_threshold
    else: # type ==3 or type=='a_or_v'
        return np.sum(av[y-patch_h//2:y+patch_h//2,x-patch_w//2:x+patch_w//2])>=limit

def patch_select(ind,patch_size,Mask ,LabelA, LabelV, LabelVessel, type=0,limit=100):
    '''
    :param patch_size:
    :param Mask: HW
    :param LabelA: HW
    :param LabelV: HW
    :param LabelVessel: HW
    :param type: 'a_or_v' or 'only_a' or 'only_v' or 'only_av'
    0 represents only_a
    1 represents only_v
    2 represents only_av
    3 represents a_or_v
    '''
    H,W = LabelA.shape
    patch_size = patch_size
    y = np.random.randint(0, LabelVessel.shape[0] - patch_size//2 + 1)
    x = np.random.randint(0, LabelVessel.shape[1] - patch_size//2 + 1)
    #corners = cv2.goodFeaturesToTrack(LabelVessel, maxCorners = 100, qualityLevel=0.001, minDistance=30)
    #num = np.random.randint(0, len(corners))
    #x, y = int(corners[num][0][0]), int(corners[num][0][1])



    #limit = int(patch_size * patch_size * rate)
    #print("limit:",limit)
    # if limit<150 and type<2:
    #     return y, x,limit
    find_patch= True
    cn=1


    while  over_window(H,W,x,y,patch_size) or not patchJudge(Mask,LabelA, LabelV, LabelVessel, y, x, patch_size, patch_size, type=type,limit=100)  and cn<1000:
        #num = np.random.randint(0, len(corners))
        y = np.random.randint(0, LabelVessel.shape[0] - patch_size//2 + 1)
        x = np.random.randint(0, LabelVessel.shape[1] - patch_size//2 + 1)
        cn+=1
    if cn>=1000:
        find_patch = False
    #print("x,y:",x,y)
    return y,x,find_patch


print("---------------------------------------")


def over_window(h,w,x,y,patch_size):
    # 以x,y为中心，patch_size为边长的正方形，是否超出h,w
    if x-patch_size//2<0 or x+patch_size//2>=w or y-patch_size//2<0 or y+patch_size//2>=h:
        return True
    else:
        return False
    pass

def check_overlap(patch, selected_patches, threshold=0.7):
    """
    检查当前patch与已选择的所有patch的重叠率是否超过阈值,只用于做数据集
    """
    if selected_patches==[]:
        return False
    for selected_patch in selected_patches:
        overlap = np.sum(patch & selected_patch) / np.sum(patch | selected_patch)
        if overlap >= threshold:
            return True
    return False

if __name__ == '__main__':
    import cv2
    from PIL import Image
    from tqdm import tqdm
    np.random.seed(4)
    #a = cv2.imread(r'E:\eye_paper\AUV-GAN\data\ukbb\test\av\test_02.png')
    dataset_mean_a = 0.0
    dataset_mean_v = 0.0
    dataset_mean_av = 0.0
    dataset_mean_a_list = []
    dataset_mean_v_list = []
    dataset_mean_av_list = []
    av_type = 0
    dataset = cfg.dataset
    dataset_dict = {'DRIVE': 'AV_DRIVE','LES':'LES_AV','IOSTSAR':'IOSTSAR', 'hrf': 'hrf', 'ukbb': 'ukbb', 'all_combine': 'all_combine'}
    dataset_name = dataset_dict[dataset]
    train_or_test = 'training'
    if not os.path.exists(os.path.join(r'E:\eye_paper\AUV-GAN\data', dataset_name, train_or_test, 'label00')):
        os.makedirs(os.path.join(r'E:\eye_paper\AUV-GAN\data', dataset_name, train_or_test, 'label00'))
    if not os.path.exists(os.path.join(r'E:\eye_paper\AUV-GAN\data', dataset_name, train_or_test, 'label01')):
        os.makedirs(os.path.join(r'E:\eye_paper\AUV-GAN\data', dataset_name, train_or_test, 'label01'))
    if not os.path.exists(os.path.join(r'E:\eye_paper\AUV-GAN\data', dataset_name, train_or_test, 'label10')):
        os.makedirs(os.path.join(r'E:\eye_paper\AUV-GAN\data', dataset_name, train_or_test, 'label10'))
    if not os.path.exists(os.path.join(r'E:\eye_paper\AUV-GAN\data', dataset_name, train_or_test, 'label11')):
        os.makedirs(os.path.join(r'E:\eye_paper\AUV-GAN\data', dataset_name, train_or_test, 'label11'))
    if not os.path.exists(os.path.join(r'E:\eye_paper\AUV-GAN\data', dataset_name, train_or_test, 'image_patch')):
        os.makedirs(os.path.join(r'E:\eye_paper\AUV-GAN\data', dataset_name, train_or_test, 'image_patch'))
    if not os.path.exists(os.path.join(r'E:\eye_paper\AUV-GAN\data', dataset_name, train_or_test, 'av_patch')):
        os.makedirs(os.path.join(r'E:\eye_paper\AUV-GAN\data', dataset_name, train_or_test, 'av_patch'))
    if not os.path.exists(os.path.join(r'E:\eye_paper\AUV-GAN\data', dataset_name, train_or_test, 'images00')):
        os.makedirs(os.path.join(r'E:\eye_paper\AUV-GAN\data', dataset_name, train_or_test, 'images00'))
    if not os.path.exists(os.path.join(r'E:\eye_paper\AUV-GAN\data', dataset_name, train_or_test, 'images01')):
        os.makedirs(os.path.join(r'E:\eye_paper\AUV-GAN\data', dataset_name, train_or_test, 'images01'))
    if not os.path.exists(os.path.join(r'E:\eye_paper\AUV-GAN\data', dataset_name, train_or_test, 'images10')):
        os.makedirs(os.path.join(r'E:\eye_paper\AUV-GAN\data', dataset_name, train_or_test, 'images10'))
    if not os.path.exists(os.path.join(r'E:\eye_paper\AUV-GAN\data', dataset_name, train_or_test, 'images11')):
        os.makedirs(os.path.join(r'E:\eye_paper\AUV-GAN\data', dataset_name, train_or_test, 'images11'))

    #dataset_name = 'hrf'
    for ind,i in enumerate(tqdm(os.listdir(os.path.join(r'E:\eye_paper\AUV-GAN\data',dataset_name,train_or_test,'av')))):
        prefix_i = i.split('.')[0]
        # 判断文件夹是否存在包含前缀加 tif jpg png其中一个的文件
        if os.path.exists(os.path.join(r'E:\eye_paper\AUV-GAN\data',dataset_name,train_or_test,'images',prefix_i+'.tif')) :
            imagepath = os.path.join(r'E:\eye_paper\AUV-GAN\data',dataset_name,train_or_test,'images',prefix_i+'.tif')
        elif os.path.exists(os.path.join(r'E:\eye_paper\AUV-GAN\data',dataset_name,train_or_test,'images',prefix_i+'.jpg')):
            imagepath = os.path.join(r'E:\eye_paper\AUV-GAN\data',dataset_name,train_or_test,'images',prefix_i+'.jpg')
        elif os.path.exists(os.path.join(r'E:\eye_paper\AUV-GAN\data',dataset_name,train_or_test,'images',prefix_i+'.png')):
            imagepath = os.path.join(r'E:\eye_paper\AUV-GAN\data',dataset_name,train_or_test,'images',prefix_i+'.png')


        path = os.path.join(r'E:\eye_paper\AUV-GAN\data',dataset_name,train_or_test,'av',i)
        #imagepath = os.path.join(r'E:\eye_paper\AUV-GAN\data',dataset_name,'training','images',i)

        #imagepath_ori = os.path.join(r'E:\eye_paper\AUV-GAN\data', dataset_name, 'training', 'images_ori', i)
        a = cv2.imread(path)

        a_image = cv2.imread(imagepath)

        a_image_bak = a_image.copy()

        a_image = cv2.cvtColor(a_image, cv2.COLOR_BGR2RGB)
        #a_image = np.transpose(a_image, (2, 0, 1))
        #a_image_ori = cv2.imread(imagepath_ori)
        #a_image_ori = cv2.cvtColor(a_image_ori, cv2.COLOR_BGR2RGB)
        #a = cv2.imread(r'E:\eye_paper\AUV-GAN\data\AV_DRIVE\training\av\21_training.png')
        LabelVessel = np.zeros((a.shape[0], a.shape[1]), dtype=np.uint8)
        LabelA = np.zeros((a.shape[0], a.shape[1]), dtype=np.uint8)
        LabelV = np.zeros((a.shape[0], a.shape[1]), dtype=np.uint8)
        Mask = np.zeros((a.shape[0], a.shape[1]), dtype=np.uint8)
        _, Mask0 = creatMask(a_image_bak, threshold=10)
        Mask[Mask0 > 0] = 1


        LabelA[(a[:, :, 2] == 255) | (a[:, :, 1] == 255)] = 1
        LabelA[(a[:, :, 2] == 255) & (a[:, :, 1] == 255) & (a[:, :, 0] == 255)] = 0

        LabelV[(a[:, :, 1] == 255) | (a[:, :, 0] == 255)] = 1
        LabelV[(a[:, :, 2] == 255) & (a[:, :, 1] == 255) & (a[:, :, 0] == 255)] = 0
        LabelVessel[(a[:, :, 2] == 255) & (a[:, :, 1] == 255) & (a[:, :, 0] == 255)] = 0
        LabelVessel[(a[:, :, 2] == 255) | (a[:, :, 1] == 255) | (a[:, :, 0] == 255)] = 1

        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

        for av_type in range(3):
            if av_type <2:
                patch_list = [64,96,128,256]
                max_epoch = 100
                threshold = 0.5
                limit = 100
            else:
                threshold = 0.5
                max_epoch = 100
                if a.shape[0]>=1000 or a.shape[1]>=1000:
                    patch_list = [128,256,min(512,a.shape[0]//2,a.shape[1]//2)]
                else:
                    patch_list = [96,128, 256]
                limit = 1
            for  patch_size in patch_list:
                patchs = []
                patch_size = patch_size
                max_epoch = max_epoch
                limit = limit
                location = []
                for epoch in range(max_epoch):
                    find_patch = True
                    y,x,find_patch = patch_select(ind,patch_size,Mask ,LabelA, LabelV, LabelVessel, type=av_type,limit=limit)
                    location.append((y, x))
                    if len(set(location))==len(set(location[:-1])):
                        location = location[:-1]
                        continue
                    patch_select2_ve = LabelVessel[y-patch_size//2:y + patch_size//2, x-patch_size//2:x + patch_size//2]

                    patch_select2_a = LabelA[y-patch_size//2:y + patch_size//2, x-patch_size//2:x + patch_size//2]
                    patch_select2_v = LabelV[y-patch_size//2:y + patch_size//2, x-patch_size//2:x + patch_size//2]

                    patch_select2_a_image = a_image[y-patch_size//2:y + patch_size//2, x-patch_size//2:x + patch_size//2,:]
                    patch_select2_a_label = a[y-patch_size//2:y + patch_size//2, x-patch_size//2:x + patch_size//2, :]
                        #Image.fromarray(patch_select2_a_label).show()
                        #Image.fromarray(patch_select2_a_image).show()
                    if find_patch and not check_overlap(patch_select2_a_image,patchs,threshold=threshold):
                        patchs.append(patch_select2_a_image)
                        if av_type==0:
                            Image.fromarray(patch_select2_a_image).save(fr'E:\eye_paper\AUV-GAN\data\{dataset_name}\{train_or_test}\image_patch\{prefix_i}-{dataset_name}_a_{epoch}_{patch_size}.png')
                            Image.fromarray(patch_select2_a_label).save(fr'E:\eye_paper\AUV-GAN\data\{dataset_name}\{train_or_test}\av_patch\{prefix_i}-{dataset_name}_a_{epoch}_{patch_size}.png')
                        elif av_type==1:
                            Image.fromarray(patch_select2_a_image).save(fr'E:\eye_paper\AUV-GAN\data\{dataset_name}\{train_or_test}\image_patch\{prefix_i}-{dataset_name}_v_{epoch}_{patch_size}.png')
                            Image.fromarray(patch_select2_a_label).save(fr'E:\eye_paper\AUV-GAN\data\{dataset_name}\{train_or_test}\av_patch\{prefix_i}-{dataset_name}_v_{epoch}_{patch_size}.png')

                        elif av_type==2:
                            Image.fromarray(patch_select2_a_image).save(fr'E:\eye_paper\AUV-GAN\data\{dataset_name}\{train_or_test}\image_patch\{prefix_i}-{dataset_name}_av_{epoch}_{patch_size}.png')
                            Image.fromarray(patch_select2_a_label).save(fr'E:\eye_paper\AUV-GAN\data\{dataset_name}\{train_or_test}\av_patch\{prefix_i}-{dataset_name}_av_{epoch}_{patch_size}.png')




