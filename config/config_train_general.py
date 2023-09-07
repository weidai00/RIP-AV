import torch
import os

# Check GPU availability
use_cuda = torch.cuda.is_available()
gpu_ids = [0] if use_cuda else []
device = torch.device('cuda' if use_cuda else 'cpu')

# dataset_name = 'ukbb'  # ukbb
dataset_name = 'DRIVE'  # DRIVE
# dataset_name = 'LES'  # LES
# dataset_name = 'IOSTSAR'  # IOSTSAR
# dataset_name = 'hrf'  # HRF
# dataset_name = 'all_combine'
dataset = dataset_name
max_step = 40000  # 30000 for ukbb
patch_size_list = [96, 128, 256] if dataset_name == 'DRIVE' else [96, 512, 256]
patch_size = patch_size_list[2]
batch_size = 2*4 # default: 4
print_iter = 100 # default: 100
display_iter = 100 # default: 100
save_iter = 5000 # default: 5000
first_display_metric_iter = max_step-save_iter # default: 25000
lr = 0.0002 if dataset_name!='LES' else 0.00005 # default: 0.0002
step_size = 7000  # 7000 for DRIVE
lr_decay_gamma = 0.5  # default: 0.5
use_SGD = False # default:False

input_nc = 3
ndf = 32
netD_type = 'basic'
n_layers_D = 5
norm = 'instance'
no_lsgan = False
init_type = 'normal'
init_gain = 0.02
use_sigmoid = no_lsgan

# torch.cuda.set_device(gpu_ids[0])
use_GAN = True # default: True

GAN_type = 'vanilla'  # 'vanilla' ,'wgan', 'rank'
treat_fake_cls0 = False
use_noise_input_D = False  # whether use the noisy image as the input of discriminator
use_dropout_D = False  # whether use dropout in each layer of discriminator
vgg_type = 'vgg19'
vgg_layers = [4, 9, 18, 27]
lambda_vgg = 1


# settings for topo loss
use_topo_loss = False  # whether use triplet loss
lambda_topo_list = [1,1,1] # A,V,Vessel
lambda_topo = 0.01

# adam
beta1 = 0.5

# settings for GAN loss
num_classes_D = 1
lambda_GAN_D = 0.01
lambda_GAN_G = 0.01
lambda_GAN_gp = 100
lambda_BCE = 5
lambda_DICE = 5
lambda_recon = 0
overlap_vessel = 0  # default: 0 (both artery and vein); 1 (artery) ; 2 (vein)

input_nc_D = input_nc + 3

# settings for centerness
use_centerness =True # default: True
dilation_list =  [0] #
lambda_centerness = 1
center_loss_type = 'centerness' # centerness or smoothl1
centerness_map_size =  [128,128]

# pretrained model
use_pretrained_G = True
use_pretrained_D = False
model_path_pretrained_G = './log/patch_pretrain'
model_path_pretrained_G = './log/2023_07_18_19_51_48'
model_step_pretrained_G = 0



# path for dataset
stride_height = 50
stride_width = 50


n_classes = 3

model_step = 0


# use CAM
use_CAM = False

#use resize
use_resize = False
resize_w_h = (256,256)

#use av_cross
use_av_cross = False

use_high_semantic = False
lambda_high = 1 # A,V,Vessel

# use global semantic
use_global_semantic = True
global_warmup_step = 0 if use_pretrained_G else 5000

# use network
use_network = 'convnext_tiny' # swin_t,convnext_tiny

dataset_path = {'DRIVE': './data/AV_DRIVE/training/',
                'DRIVE_centerness': './data/AV_DRIVE/training/centerness_maps/',

                'hrf': './data/hrf/training/',
                'hrf_centerness': './data/hrf/training/centerness_maps/',

                'LES': './data/LES_AV/training/',
                'LES_centerness': './data/LES_AV/training/centerness_maps/',

                'IOSTSAR': './data/IOSTSAR/training',
                'IOSTSAR_centerness': './data/IOSTSAR/training/centerness_maps/',

                'ukbb': './data/ukbb/training/',
                'ukbb_centerness': './data/ukbb/training/centerness_maps/',

                'all_combine': './data/all_combine/training/',
                'all_combine_centerness': './data/all_combine/training/centerness_maps/',

                }
trainset_path = dataset_path[dataset_name]
trainset_centerness_path = dataset_path[dataset_name + '_centerness']

print("Dataset:")
print(trainset_path)
print(trainset_centerness_path)
print(use_network)



