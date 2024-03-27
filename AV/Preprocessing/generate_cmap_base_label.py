# -*- coding: utf-8 -*-
"""
GENERATE CENTERNESS MAPS

[label_root] need change by distribute of dataset directory
"""
from scipy import ndimage
import cv2
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
dataset_name = 'ukbb'
number_class = 3


print(f"Generating centerness maps for {dataset_name} dataset.")
print(f"The centerness maps are saved to : ../data/{dataset_name}/[training|test]/centerness_maps")


dataset_type_list = ['training', 'test']
class_name_list = ["art", "ven", "ves"]

for dataset_type in dataset_type_list:
    print("Dataset type:", dataset_type)
    if os.path.exists(f'../data/{dataset_name}/{dataset_type}/centerness_maps'):
        print("centerness_maps exist.")
    else:
        os.makedirs(f'../data/{dataset_name}/{dataset_type}/centerness_maps', exist_ok=True)
        print("centerness_maps is built.")
    save_root = f'../data/{dataset_name}/{dataset_type}/centerness_maps/'

    # change
    label_root = f'../data/{dataset_name}/{dataset_type}/av/'

    for class_type in range(number_class):
        # 0 for artery
        # 1 for vein
        # 2 for vessel
        t = tqdm(os.listdir(label_root))
        for img_name in t:
            t.set_description(f"Processing {dataset_name}== {dataset_type}== "
                              f"{class_name_list[class_type]}==={img_name}")
            prefix = os.path.splitext(img_name)[0]
            suffix = os.path.splitext(img_name)[1]

            cen_img_save = f'{prefix}_{class_name_list[class_type]}_cen{suffix}'

            label_img = cv2.imread(os.path.join(label_root, img_name))

            LabelArtery = np.zeros((label_img.shape[0], label_img.shape[1]), np.uint8)
            LabelVein = np.zeros((label_img.shape[0], label_img.shape[1]), np.uint8)
            LabelVessel = np.zeros((label_img.shape[0], label_img.shape[1]), np.uint8)

            LabelArtery[(label_img[:, :, 2] == 255) | (label_img[:, :, 1] == 255)] = 1
            LabelArtery[(label_img[:, :, 2] == 255) & (label_img[:, :, 1] == 255) & (label_img[:, :, 0] == 255)] = 0
            LabelVein[(label_img[:, :, 1] == 255) | (label_img[:, :, 0] == 255)] = 1
            LabelVein[(label_img[:, :, 2] == 255) & (label_img[:, :, 1] == 255) & (label_img[:, :, 0] == 255)] = 0
            LabelVessel[(label_img[:, :, 2] == 255) | (label_img[:, :, 1] == 255) | (label_img[:, :, 0] == 255)] = 1

            Labels = [LabelArtery, LabelVein, LabelVessel]

            center_img = ndimage.distance_transform_edt(Labels[class_type])

            #std = center_img.std()
            #mean = center_img.mean()
            max_dis = center_img.max()
            min_dis = center_img.min()

            norm_dis = np.uint8((center_img-min_dis) / (max_dis-min_dis)*255)

            distance_map = cv2.applyColorMap(np.uint8(norm_dis), cv2.COLORMAP_JET)
            #cv2.imwrite(os.path.join(save_root, f'{cen_img_save}_distance_map.png'), distance_map)

            #save distance transform from original image
            cv2.imwrite(os.path.join(save_root, cen_img_save), norm_dis)


print("---------------------------------------")