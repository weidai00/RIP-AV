# -*- coding: utf-8 -*-
import torchvision.models
from torch import nn
import torch

import torch.nn.functional as F
from models.layers import *
from torchvision.models.convnext import convnext_tiny, ConvNeXt_Tiny_Weights
import numpy as np
import math
from torchvision import models
import copy

class PGNet(nn.Module):
    def __init__(self, input_ch=3, resnet='convnext_tiny', num_classes=3, use_cuda=False, pretrained=True,centerness=False, centerness_map_size=[128,128],use_global_semantic=False):
        super(PGNet, self).__init__()
        self.resnet = resnet
        base_model = convnext_tiny
        # layers = list(base_model(pretrained=pretrained,num_classes=num_classes,input_ch=input_ch).children())[:cut]
        self.use_high_semantic = False

        cut = 6
        if pretrained:
            layers = list(base_model(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).features)[:cut]
        else:
            layers = list(base_model().features)[:cut]

        base_layers = nn.Sequential(*layers)
        self.use_global_semantic = use_global_semantic
        ### global momentum
        if self.use_global_semantic:

            self.pg_fusion = PGFusion()
            self.base_layers_global_momentum = copy.deepcopy(base_layers)
            set_requires_grad(self.base_layers_global_momentum,requires_grad=False)

        # self.stage = [SaveFeatures(base_layers[0][1])]  # stage 1  c=96

        self.stage = []
        self.stage.append(SaveFeatures(base_layers[0][1]))  # stem c=96
        self.stage.append(SaveFeatures(base_layers[1][2]))  # stage 1 c=96
        self.stage.append(SaveFeatures(base_layers[3][2]))  # stage 2 c=192
        self.stage.append(SaveFeatures(base_layers[5][8]))  # stage 3 c=384
        # self.stage.append(SaveFeatures(base_layers[7][2]))  # stage 5 c=768

        self.up2 = DBlock(384, 192)
        self.up3 = DBlock(192, 96)
        self.up4 = DBlock(96, 96)

        # final convolutional layers
        # predict artery, vein and vessel

        self.seg_head = SegmentationHead(96, num_classes, 3, upsample=4)

        self.sn_unet = base_layers
        self.num_classes = num_classes

        self.bn_out = nn.BatchNorm2d(3)
        #self.av_cross = AV_Cross(block=4,kernel_size=1)
        # use centerness block
        self.centerness = centerness

        if self.centerness and centerness_map_size[0] == 128:

            # block 1
            self.cenBlock1 = [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ]
            self.cenBlock1 = nn.Sequential(*self.cenBlock1)

            # centerness block
            self.cenBlockMid = [
                nn.Conv2d(96, 48, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(48),
                # nn.Conv2d(48, 48, kernel_size=3, padding=3, bias=False),
                # nn.BatchNorm2d(48),
                nn.Conv2d(48, 96, kernel_size=1, padding=0, bias=False),
            ]
            self.cenBlockMid = nn.Sequential(*self.cenBlockMid)
            self.cenBlockFinal = [
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.Conv2d(96, 3, kernel_size=1, padding=0, bias=True),
                nn.Sigmoid()

            ]
            self.cenBlockFinal = nn.Sequential(*self.cenBlockFinal)
    def forward(self, x,y=None):

        x = self.sn_unet(x)
        global_rep = None
        if self.use_global_semantic:
            global_rep = self.base_layers_global_momentum(y)
            x = self.pg_fusion(x,global_rep)
        if len(x.shape) == 4 and x.shape[2] != x.shape[3]:
            B, H, W, C = x.shape
            x = x.permute(0, 3, 1, 2).contiguous()
        elif len(x.shape) == 3:
            B, L, C = x.shape
            h = int(L ** 0.5)
            x = x.view(B, h, h, C)
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            x = x

        if self.use_high_semantic:
            high_out = x.clone()
        else:
            high_out = x.clone()
        if self.resnet == 'swin_t' or self.resnet == 'convnext_tiny':
            # feature = self.stage[1:]
            feature = self.stage[::-1]
            # head = feature[0]
            skip = feature[1:]

            # x = self.up1(x,skip[0].features)
            x = self.up2(x, skip[0].features)
            x = self.up3(x, skip[1].features)
            x = self.up4(x, skip[2].features)
        x_out = self.seg_head(x)
        ########################
        # baseline output
        # artery, vein and vessel
        output = x_out.clone()
        #av cross
        #output = self.av_cross(output)
        #output = F.relu(self.bn_out(output))
        # use centerness block
        centerness_maps = None
        if self.centerness:

            block1 = self.cenBlock1(self.stage[1].features)  # [96,64]
            _block1 = self.cenBlockMid(block1)  # [96,64]
            block1 = block1 + _block1
            blocks = [block1]
            blocks = torch.cat(blocks, dim=1)

            # print("blocks", blocks.shape)
            centerness_maps = self.cenBlockFinal(blocks)
            # print("maps:", centerness_maps.shape)

        return output, centerness_maps


    def forward_patch_rep(self, x):
        patch_rep = self.sn_unet(x)
        return patch_rep

    def forward_global_rep_momentum(self, x):
        global_rep = self.base_layers_global_momentum(x)
        return global_rep
    def close(self):
        for sf in self.stage: sf.remove()




def close(self):
        for sf in self.stage: sf.remove()


# set requies_grad=Fasle to avoid computation

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
def choose_vgg(name):
    f = None

    if name == 'vgg11':
        f = models.vgg11(pretrained=True)
    elif name == 'vgg11_bn':
        f = models.vgg11_bn(pretrained=True)
    elif name == 'vgg13':
        f = models.vgg13(pretrained=True)
    elif name == 'vgg13_bn':
        f = models.vgg13_bn(pretrained=True)
    elif name == 'vgg16':
        f = models.vgg16(pretrained=True)
    elif name == 'vgg16_bn':
        f = models.vgg16_bn(pretrained=True)
    elif name == 'vgg19':
        f = models.vgg19(pretrained=True)
    elif name == 'vgg19_bn':
        f = models.vgg19_bn(pretrained=True)

    for params in f.parameters():
        params.requires_grad = False

    return f


pretrained_mean = torch.tensor([0.485, 0.456, 0.406], requires_grad=False).view((1, 3, 1, 1))
pretrained_std = torch.tensor([0.229, 0.224, 0.225], requires_grad=False).view((1, 3, 1, 1))

class VGGNet(nn.Module):

    def __init__(self, name, layers, cuda=True):

        super(VGGNet, self).__init__()
        self.vgg = choose_vgg(name)
        self.layers = layers

        features = list(self.vgg.features)[:max(layers) + 1]
        self.features = nn.ModuleList(features).eval()

        self.mean = pretrained_mean.cuda() if cuda else pretrained_mean
        self.std = pretrained_std.cuda() if cuda else pretrained_std

    def forward(self, x, retn_feats=None, layers=None):

        x = (x - self.mean) / self.std

        results = []

        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in self.layers:
                results.append(x.reshape(x.shape[0], -1))

        return results

if __name__ == '__main__':
    s = PGNet(input_ch=3, resnet='convnext_tiny',centerness=True, pretrained=False,use_global_semantic=True)


    x = torch.randn(2, 3, 256, 256)
    y,Y2 = s(x,x)



    print(y.shape)
    print(Y2.shape)



    # pt = torch.load(r'F:\dw\MICCAI2023-STS-2D\segmentation\log\2023_07_25_18_10_10\G_0.pkl')
    # print(pt)
    # import torchvision.models as models
    # m = models.vit_b_16(pretrained=False)
    # print(m)
    # m = resnet18()
    # m_list = list(m.children())
    # def hook(module, input, output):
    #     print('fafafafgafa')
    #     print(input[0].shape)
    #     print(output[0].shape)
    # m_list[0].register_forward_hook(hook)
    #
    #
    # y = m(x)
