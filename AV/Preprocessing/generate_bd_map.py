# -*- coding: utf-8 -*-
"""
GENERATE BOUNDARY MAPS

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


print(f"Generating boundary maps for {dataset_name} dataset.")
print(f"The boundary maps are saved to : ../data/{dataset_name}/[training|test]/boundary_maps")


dataset_type_list = ['training', 'test']
class_name_list = ["art", "ven", "ves"]

for dataset_type in dataset_type_list:
    print("Dataset type:", dataset_type)
    if os.path.exists(f'../data/{dataset_name}/{dataset_type}/boundary_maps'):
        print("boundary_maps exist.")
    else:
        os.makedirs(f'../data/{dataset_name}/{dataset_type}/boundary_maps', exist_ok=True)
        print("boundary_maps is built.")
    save_root = f'../data/{dataset_name}/{dataset_type}/boundary_maps/'

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

            bd_img_save = f'{prefix}_{class_name_list[class_type]}_bd{suffix}'

            label_img = cv2.imread(os.path.join(label_root, img_name))

            LabelArtery = np.zeros((label_img.shape[0], label_img.shape[1]), np.uint8)
            LabelVein = np.zeros((label_img.shape[0], label_img.shape[1]), np.uint8)
            LabelVessel = np.zeros((label_img.shape[0], label_img.shape[1]), np.uint8)

            LabelArtery[(label_img[:, :, 2] == 255) | (label_img[:, :, 1] == 255)] = 255
            LabelArtery[(label_img[:, :, 2] == 255) & (label_img[:, :, 1] == 255) & (label_img[:, :, 0] == 255)] = 0
            LabelVein[(label_img[:, :, 1] == 255) | (label_img[:, :, 0] == 255)] = 255
            LabelVein[(label_img[:, :, 2] == 255) & (label_img[:, :, 1] == 255) & (label_img[:, :, 0] == 255)] = 0
            LabelVessel[(label_img[:, :, 2] == 255) | (label_img[:, :, 1] == 255) | (label_img[:, :, 0] == 255)] = 255

            Labels = [LabelArtery, LabelVein, LabelVessel]

            contours, _ = cv2.findContours(Labels[class_type], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            boundary_map = np.zeros_like(LabelArtery)

            min_area_threshold = 50
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area_threshold]
            cv2.drawContours(boundary_map, filtered_contours, -1, (255), thickness=1)



            #save boundary_map from original image
            cv2.imwrite(os.path.join(save_root, bd_img_save), boundary_map)


print("---------------------------------------")