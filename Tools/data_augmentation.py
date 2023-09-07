import numpy as np
import tensorlayer as tl

def data_augmentation1_5(*args):
    # image3 = np.expand_dims(image3,-1)
    args = tl.prepro.rotation_multi(args, rg=180, is_random=True,
                                                        fill_mode='constant')
    args = np.squeeze(args).astype(np.float32)

    return args

def data_augmentation3_5(*args):
    # image3 = np.expand_dims(image3,-1)
    args = tl.prepro.shift_multi(args, wrg=0.10, hrg=0.10, is_random=True,
                                                     fill_mode='constant')
    args = np.squeeze(args).astype(np.float32)

    return args

def data_augmentation4_5(*args):

    args = tl.prepro.swirl_multi(args,is_random=True)
    args = np.squeeze(args).astype(np.float32)

    return args

def data_augmentation2_5(*args):
    # image3 = np.expand_dims(image3,-1)
    args = tl.prepro.zoom_multi(args, zoom_range=[0.5, 1.5], is_random=True,
                                                    fill_mode='constant')
    args = np.squeeze(args).astype(np.float32)

    return args

def data_aug5_old(data_mat, label_mat, label_data_centerness, choice):
    data_mat = np.transpose(data_mat, (1, 2, 0))
    label_mat = np.transpose(label_mat, (1, 2, 0))
    label_data_centerness = np.transpose(label_data_centerness, (1, 2, 0))

    if choice == 0:
        data_mat = data_mat
        label_mat = label_mat
        label_data_centerness = label_data_centerness

    elif choice == 1:
        data_mat = np.fliplr(data_mat)
        label_mat = np.fliplr(label_mat)
        label_data_centerness = np.fliplr(label_data_centerness)

    elif choice == 2:
        data_mat = np.flipud(data_mat)
        label_mat = np.flipud(label_mat)
        label_data_centerness = np.flipud(label_data_centerness)

    elif choice == 3:
        data_mat, label_mat, label_data_centerness= data_augmentation1_5(data_mat, label_mat, label_data_centerness)
    elif choice == 4:
        data_mat, label_mat, label_data_centerness= data_augmentation2_5(data_mat, label_mat, label_data_centerness)
    elif choice == 5:
        data_mat, label_mat, label_data_centerness= data_augmentation3_5(data_mat, label_mat, label_data_centerness)
    elif choice == 6:
        data_mat, label_mat, label_data_centerness= data_augmentation4_5(data_mat, label_mat, label_data_centerness)

    data_mat = np.transpose(data_mat, (2, 0, 1))
    label_mat = np.transpose(label_mat, (2, 0, 1))
    label_data_centerness = np.transpose(label_data_centerness, (2, 0, 1))


    return data_mat, label_mat, label_data_centerness

# data augmentation for variable number of input
def data_aug5(*args,choice):
    datas=[np.transpose(item, (1, 2, 0)) for item in args]

    if choice==1:
        datas=[np.fliplr(item) for item in datas]
    elif choice==2:
        datas = [np.flipud(item) for item in datas]
    elif choice==3:
        datas = data_augmentation1_5(*datas)
    elif choice==4:
        datas = data_augmentation2_5(*datas)
    elif choice==5:
        datas = data_augmentation3_5(*datas)
    elif choice==6:
        datas = data_augmentation4_5(*datas)

    datas = [np.transpose(item, (2, 0, 1)) for item in datas]

    return tuple(datas)




