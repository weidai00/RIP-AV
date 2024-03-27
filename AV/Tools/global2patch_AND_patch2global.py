# -*- encoding: utf-8 -*-
# author Victorshengw


import os
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
import  torch
from PIL import Image


def get_patch_info(shape, p_size):
    '''
    shape: origin image size, (x, y)
    p_size: patch size (square)
    return: n_x, n_y, step_x, step_y
    '''
    x = shape[0]
    y = shape[1]
    if x==p_size and y==p_size:
        return 1, 1, 0, 0

    n = m = 1
    while x > n * p_size:
        n += 1
    while p_size - 1.0 * (x - p_size) / (n - 1) < p_size/4:
        n += 1
    while y > m * p_size:
        m += 1
    while p_size - 1.0 * (y - p_size) / (m - 1) < p_size/4:
        m += 1
    return n, m, (x - p_size) * 1.0 / (n - 1), (y - p_size) * 1.0 / (m - 1)



def global2patch(images, p_size):
    '''
    image/label => patches
    p_size: patch size
    return: list of PIL patch images; coordinates: images->patches; ratios: (h, w)
    '''
    patches = []; coordinates = []; templates = []; sizes = []; ratios = [(0, 0)] * len(images); patch_ones = np.ones(p_size)
    for i in range(len(images)):
        w, h = images[i].size
        size = (h, w)
        sizes.append(size)
        ratios[i] = (float(p_size[0]) / size[0], float(p_size[1]) / size[1])
        template = np.zeros(size)
        n_x, n_y, step_x, step_y = get_patch_info(size, p_size[0])
        patches.append([images[i]] * (n_x * n_y))
        coordinates.append([(0, 0)] * (n_x * n_y))
        for x in range(n_x):
            if x < n_x - 1: top = int(np.round(x * step_x))
            else: top = size[0] - p_size[0]
            for y in range(n_y):
                if y < n_y - 1: left = int(np.round(y * step_y))
                else: left = size[1] - p_size[1]
                template[top:top+p_size[0], left:left+p_size[1]] += patch_ones
                coordinates[i][x * n_y + y] = (1.0 * top / size[0], 1.0 * left / size[1])
                patches[i][x * n_y + y] = transforms.functional.crop(images[i], top, left, p_size[0], p_size[1])

                # patches[i][x * n_y + y].show()
        templates.append(Variable(torch.Tensor(template).expand(1, 1, -1, -1)))
    return patches, coordinates, templates, sizes, ratios

def patch2global(patches, n_class, sizes, coordinates, p_size,flag = 0):
    '''
    predicted patches (after classify layer) => predictions
    return: list of np.array
    '''
    patches = np.array(torch.detach(patches).cpu().numpy())
    predictions = [ np.zeros((n_class, size[0], size[1])) for size in sizes]

    for i in range(len(sizes)):
        for j in range(len(coordinates[i])):
            top, left = coordinates[i][j]
            top = int(np.round(top * sizes[i][0])); left = int(np.round(left * sizes[i][1]))

            patches_tmp = np.zeros(patches[j][:,:,:].shape)
            whole_img_tmp = predictions[i][:, top: top + p_size[0], left: left + p_size[1]]
            #俩小块儿最大(max)的成为最终的prediction
            #patches[j][:,:,:] 就是每个要贴到大图中的小块儿，whole_img_tmp是整个目标大图中对应patches_tmp的那一小块儿，然后将这俩及逆行比较，谁大就取谁
            if flag == 0:
                patches_tmp[patches[j][:, :, :] > whole_img_tmp] = patches[j][:,:,:][patches[j][:, :, :] > whole_img_tmp]  # 要贴上去的小块中的值大于大图中的值 patches[j][:, :, :] > whole_img_tmp
                patches_tmp[patches[j][:, :, :] < whole_img_tmp] = whole_img_tmp[patches[j][:, :, :] < whole_img_tmp]  # 要贴上去的小块中的值小于于大图中的值 patches[j][:, :, :] < whole_img_tmp
                predictions[i][:, top: top + p_size[0], left: left + p_size[1]] += patches_tmp
            else:


                predictions[i][:, top: top + p_size[0], left: left + p_size[1]] += patches[j][:, :, :]


    return predictions


if __name__ == '__main__':
    images = []

    img = Image.open(os.path.join(r"../train_valid/003DRIVE/image", f"01.png"))
    images.append(img)
    # print(len(images))  = 3
    p_size = (224,224)
    patches, coordinates, templates, sizes, ratios = global2patch(images, p_size)
    # predictions = patch2global(patches, 3, sizes, coordinates, p_size)
    # print(type(predictions))