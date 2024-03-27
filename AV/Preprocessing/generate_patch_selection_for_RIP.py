# -*- coding: utf-8 -*-

import numpy as np
import os
import config.config_train_general as cfg

def patchJudge(mask, a, v, av, y, x, patch_h, patch_w, type='only_a', limit=50):
    '''
    :param mask: mask
    :param a: artery
    :param v: vein
    :param av: artery and vein
    :param y: y coordinate of the patch
    :param x: x coordinate of the patch
    :param patch_h: height of the patch
    :param patch_w: width of the patch
    :param type: 'only_a', 'only_v', 'only_av'
    :param limit: the limit of the number of pixels in the patch
    0 represents only_a
    1 represents only_v
    2 represents only_av
    '''
    # print(type)
    # print(patch_w)
    # print(limit)
    # print(f'av:{np.sum(av[y:y + patch_h, x:x + patch_w])}')
    # print(f'a:{np.sum(a[y:y + patch_h, x:x+patch_w])}')
    # print(f'v:{np.sum(v[y:y + patch_h, x:x + patch_w])}')

    # print(b_rate)

    # b_rate = np.sum(mask[y:y+patch_h,x:x+patch_w] == 0)/patch_h/patch_w
    # print(b_rate)
    b_rate = 0
    b_rate_threshold = 1

    if type == 0 or type == 'only_a':
        return np.sum(a[y:y + patch_h, x:x + patch_w]) >= limit and np.sum(
            v[y:y + patch_h, x:x + patch_w]) == 0 and b_rate <= b_rate_threshold
    elif type == 1 or type == 'only_v':
        return np.sum(v[y:y + patch_h, x:x + patch_w]) >= limit and np.sum(
            a[y:y + patch_h, x:x + patch_w]) == 0 and b_rate <= b_rate_threshold
    elif type == 2 or type == 'only_av':

        return np.sum(av[y:y + patch_h, x:x + patch_w]) >= 2*limit and np.sum(
            a[y:y + patch_h, x:x + patch_w]) >= limit  and np.sum(
            v[y:y + patch_h, x:x + patch_w]) >= limit  and b_rate <= b_rate_threshold
    else:
        return True


def patch_select_for_RIP(patch_size, Mask, LabelA, LabelV, LabelVessel, av_type=0):
    '''
    :param patch_size:
    :param Mask: 3HW
    :param LabelA: HW
    :param LabelV: HW
    :param LabelVessel: HW
    :param av_type: 'a_or_v' or 'only_a' or 'only_v' or 'only_av'
    0 represents only_a
    1 represents only_v
    2 represents only_av
    '''
    H, W = LabelA.shape
    # Ve_rate_back = np.count_nonzero(LabelVessel) / (H * W)
    # V_rate_back = np.count_nonzero(LabelV) / (H * W)
    # A_rate_back = np.count_nonzero(LabelA) / (H * W)
    # rate = min(Ve_rate_back, V_rate_back, A_rate_back)

    limit = 50

    if type == 0:
        skel = Mask[0, :, :]
    elif type == 1:
        skel = Mask[1, :, :]
    else:
        skel = Mask[2, :, :]

    skels = np.argwhere(skel)
    find_patch = False
    dot_pix = np.random.randint(0, len(skels))
    y_aixs = skels[dot_pix][0]
    x_aixs = skels[dot_pix][1]
    roi_area_x_left = max(0, x_aixs - patch_size)
    roi_area_y_left = max(0, y_aixs - patch_size)
    roi_area_x_right = min(x_aixs + patch_size, LabelVessel.shape[1])
    roi_area_y_right = min(y_aixs + patch_size, LabelVessel.shape[0])

    y = np.random.randint(roi_area_y_left, roi_area_y_right - patch_size + 1)
    x = np.random.randint(roi_area_x_left, roi_area_x_right - patch_size + 1)

    cn = 0
    limit_cn = 2000
    while (not patchJudge(Mask, LabelA, LabelV, LabelVessel, y, x, patch_size, patch_size, type=av_type,
                          limit=limit)) and cn < limit_cn:
        if cn % 50 == 0 and cn > 0:
            dot_pix = np.random.randint(0, skels.shape[0])
            y_aixs = skels[dot_pix][0]  # w
            x_aixs = skels[dot_pix][1]  # h
            roi_area_x_left = max(0, x_aixs - patch_size)
            roi_area_y_left = max(0, y_aixs - patch_size)
            roi_area_x_right = min(x_aixs + patch_size, LabelVessel.shape[1])
            roi_area_y_right = min(y_aixs + patch_size, LabelVessel.shape[0])
        y = np.random.randint(roi_area_y_left, roi_area_y_right - patch_size + 1)
        x = np.random.randint(roi_area_x_left, roi_area_x_right - patch_size + 1)
        cn+=1

    if cn < limit_cn:
        find_patch = True

    return y, x,patch_size,find_patch,roi_area_y_left,roi_area_x_left,roi_area_y_right,roi_area_x_right


def check_overlap(patch, selected_patches, threshold=0.8,patch_size=64):
    """
    Check if the patch overlaps with the selected patches
    patch:[x,y]
    selected_patches: [[x,y],[x,y]...]
    """


    if selected_patches==[]:
        return False
    for selected_patch in selected_patches:
        patch_x,patch_y = patch
        patch_x_min = patch_x-patch_size//2
        patch_x_max = patch_x+patch_size//2
        patch_y_min = patch_y-patch_size//2
        patch_y_max = patch_y+patch_size//2

        selected_patch_x,selected_patch_y = selected_patch
        selected_patch_x_min = selected_patch_x-patch_size//2
        selected_patch_x_max = selected_patch_x+patch_size//2
        selected_patch_y_min = selected_patch_y-patch_size//2
        selected_patch_y_max = selected_patch_y+patch_size//2


        #IOU
        x_left = max(patch_x_min,selected_patch_x_min)
        x_right = min(patch_x_max,selected_patch_x_max)
        y_top = max(patch_y_min,selected_patch_y_min)
        y_bottom = min(patch_y_max,selected_patch_y_max)

        iou_area = max(0,x_right-x_left)*max(0,y_bottom-y_top)
        overlap = iou_area/(patch_size*patch_size)

        if overlap >= threshold:
            return True
    return False

print("---------------------------------------")


if __name__ == '__main__':
    import cv2
    from PIL import Image
    from tqdm import tqdm,trange
    import argparse
    from skimage import morphology

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='../data/AV_DRIVE', help='dataset absolute path')
    parser.add_argument('--train_or_test', type=str, default='training', help='training or test')
    parser.add_argument('--av_type', type=list, default=[0,1,2], help='av_type range from 0 to 2')
    parser.add_argument('--patch_size', type=list, default=[64,96,128,256], help='patch_size')
    parser.add_argument('--output', type=str, default='./', help='output path')
    args = parser.parse_args()
    np.random.seed(4)
    av_type = args.av_type
    patch_size = args.patch_size
    dataset_path = args.dataset_path
    train_or_test = args.train_or_test
    output = args.output


    if not os.path.exists(os.path.join(output,train_or_test,'images10')):
        os.makedirs(os.path.join(output,train_or_test,'images10'))
        os.makedirs(os.path.join(output, train_or_test, 'label10'))
    if not os.path.exists(os.path.join(output,train_or_test,'images01')):
        os.makedirs(os.path.join(output,train_or_test,'images01'))
        os.makedirs(os.path.join(output, train_or_test, 'label01'))
    if not os.path.exists(os.path.join(output,train_or_test,'images11')):
        os.makedirs(os.path.join(output,train_or_test,'images11'))
        os.makedirs(os.path.join(output, train_or_test, 'label11'))

    for ind,i in enumerate(tqdm(os.listdir(os.path.join(dataset_path,train_or_test,'av')))):
        prefix_i = i.split('.')[0]
        if os.path.exists(os.path.join(dataset_path,train_or_test,'images',prefix_i+'.tif')) :
            imagepath = os.path.join(dataset_path,train_or_test,'images',prefix_i+'.tif')
        elif os.path.exists(os.path.join(dataset_path,train_or_test,'images',prefix_i+'.jpg')):
            imagepath = os.path.join(dataset_path,train_or_test,'images',prefix_i+'.jpg')
        elif os.path.exists(os.path.join(dataset_path,train_or_test,'images',prefix_i+'.png')):
            imagepath = os.path.join(dataset_path,train_or_test,'images',prefix_i+'.png')

        av_path = os.path.join(dataset_path,train_or_test,'av',i)
        av = cv2.imread(av_path)
        image = cv2.imread(imagepath)
        image_bak = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        LabelVessel = np.zeros((av.shape[0], av.shape[1]), dtype=np.uint8)
        LabelA = np.zeros((av.shape[0], av.shape[1]), dtype=np.uint8)
        LabelV = np.zeros((av.shape[0], av.shape[1]), dtype=np.uint8)
        Mask = np.zeros((3,av.shape[0], av.shape[1]), dtype=np.uint8)

        LabelA[(av[:, :, 2] == 255) | (av[:, :, 1] == 255)] = 1
        LabelA[(av[:, :, 2] == 255) & (av[:, :, 1] == 255) & (av[:, :, 0] == 255)] = 0

        LabelV[(av[:, :, 1] == 255) | (av[:, :, 0] == 255)] = 1
        LabelV[(av[:, :, 2] == 255) & (av[:, :, 1] == 255) & (av[:, :, 0] == 255)] = 0

        LabelVessel[(av[:, :, 2] == 255) | (av[:, :, 1] == 255) | (av[:, :, 0] == 255)] = 1
        LabelVessel[(av[:, :, 2] == 255) & (av[:, :, 1] == 255) & (av[:, :, 0] == 255)] = 1

        print(f've: {np.count_nonzero(LabelVessel)/(av.shape[0]*av.shape[1]-np.count_nonzero(LabelVessel))} , a: {np.count_nonzero(LabelA)/(av.shape[0]*av.shape[1]-np.count_nonzero(LabelA))} , v: {np.count_nonzero(LabelV)/(av.shape[0]*av.shape[1]-np.count_nonzero(LabelV))} ')

        skel_A = morphology.medial_axis(LabelA)
        skel_V = morphology.medial_axis(LabelV)
        skel = morphology.medial_axis(LabelVessel)

        Mask[0] = skel_A
        Mask[1] = skel_V
        Mask[2] = skel

        a_count = []
        v_count = []
        ve_count = []

        for av_type_i in av_type:
            if av_type_i == 2:
                max_epoch = 25
                patch_size = [96,128, 256]
            else:
                max_epoch = 50
                patch_size = [64, 96]
            for patch_size_i in patch_size:
                patchs = []
                for i in trange(max_epoch):
                    y, x, patch_size,find_patch,_,_,_,_ = patch_select_for_RIP(patch_size_i, Mask, LabelA, LabelV, LabelVessel, av_type=av_type_i)
                    if not find_patch:
                        continue
                    patch_select_ve = LabelVessel[y:y + patch_size, x:x + patch_size]
                    patch_select_a = LabelA[y:y + patch_size, x:x + patch_size]
                    patch_select_v = LabelV[y:y + patch_size, x:x + patch_size]

                    patch_select_image = image[y:y + patch_size, x:x + patch_size, :]
                    patch_select_av = av[y:y + patch_size, x:x + patch_size, :]


                    a_count.append(np.count_nonzero(patch_select_a))
                    v_count.append(np.count_nonzero(patch_select_v))
                    ve_count.append(np.count_nonzero(patch_select_ve))

                    # Image.fromarray(patch_select2_ve*255).show()
                    # Image.fromarray(patch_select2_a_image).show()

                    if not check_overlap([x,y], patchs,patch_size=patch_size_i):
                        patchs.append([x,y])
                        if av_type_i == 0:
                            Image.fromarray(patch_select_image).save(os.path.join(output, train_or_test, 'images10',
                                                                                     f'a_{ind}_{ind * max_epoch + i}_{patch_size}.png'))

                            Image.fromarray(patch_select_av).save(os.path.join(output, train_or_test, 'label10',
                                                                                  f'a_{ind}_{ind * max_epoch + i}_{patch_size}.png'))
                        elif av_type_i == 1:
                            Image.fromarray(patch_select_image).save(os.path.join(output, train_or_test, 'images01',
                                                                                     f'v_{ind}_{ind * max_epoch + i}_{patch_size}.png'))
                            Image.fromarray(patch_select_av).save(os.path.join(output, train_or_test, 'label01',
                                                                               f'v_{ind}_{ind * max_epoch + i}_{patch_size}.png'))
                        elif av_type_i == 2:
                            Image.fromarray(patch_select_image).save(os.path.join(output, train_or_test, 'images11',
                                                                                     f'av_{ind}_{ind * max_epoch + i}_{patch_size}.png'))
                            Image.fromarray(patch_select_av).save(os.path.join(output, train_or_test, 'label11',
                                                                               f'av_{ind}_{ind * max_epoch + i}_{patch_size}.png'))


    # print("dataset_mean_a_list:",dataset_mean_a_list)
    # print("dataset_mean_v_list:",dataset_mean_v_list)
    # print("dataset_mean_av_list:",dataset_mean_av_list)
    #
    # print("min_a:",min(dataset_mean_a_list))
    # print("min_v:",min(dataset_mean_v_list))
    # print("min_av:",min(dataset_mean_av_list))
    #
    # print("dataset_mean_a:",np.mean(dataset_mean_a_list))
    # print("dataset_mean_v:",np.mean(dataset_mean_v_list))
    # print("dataset_mean_av:",np.mean(dataset_mean_av_list))
