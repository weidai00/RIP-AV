# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
# from timm.models.layers.cbam import CbamModule
import numpy as np
from einops import rearrange, repeat
import math


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class sSE(nn.Module):
    def __init__(self, out_channels):
        super(sSE, self).__init__()
        self.conv = ConvBn2d(in_channels=out_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        # print('spatial',x.size())
        x = F.sigmoid(x)
        return x


class cSE(nn.Module):
    def __init__(self, out_channels):
        super(cSE, self).__init__()
        self.conv1 = ConvBn2d(in_channels=out_channels, out_channels=int(out_channels / 2), kernel_size=1, padding=0)
        self.conv2 = ConvBn2d(in_channels=int(out_channels / 2), out_channels=out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = nn.AvgPool2d(x.size()[2:])(x)
        # print('channel',x.size())
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.sigmoid(x)
        return x


class scSEBlock(nn.Module):
    def __init__(self, out_channels):
        super(scSEBlock, self).__init__()
        self.spatial_gate = sSE(out_channels)
        self.channel_gate = cSE(out_channels)

    def forward(self, x):
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = g1 * x + g2 * x
        return x


class SaveFeatures():
    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # print('input',input)
        # print('output',output.size())
        if len(output.shape) == 3:
            B, L, C = output.shape
            h = int(L ** 0.5)
            output = output.view(B, h, h, C)

            output = output.permute(0, 3, 1, 2).contiguous()
        if len(output.shape) == 4 and output.shape[2] != output.shape[3]:
            output = output.permute(0, 3, 1, 2).contiguous()
        # print(module)
        self.features = output

    def remove(self):
        self.hook.remove()


class DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_batchnorm=True, attention_type=None):

        super(DBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        if attention_type == 'scse':
            self.attention1 = scSEBlock(in_channels)
        elif attention_type == 'cbam':
            self.attention1 = nn.Identity()

        elif attention_type == 'transformer':

            self.attention1 = nn.Identity()


        else:
            self.attention1 = nn.Identity()

        self.conv2 = \
            nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        if attention_type == 'scse':
            self.attention2 = scSEBlock(out_channels)
        elif attention_type == 'cbam':
            self.attention2 = CbamModule(channels=out_channels)

        elif attention_type == 'transformer':
            self.attention2 = nn.Identity()

        else:
            self.attention2 = nn.Identity()

    def forward(self, x, skip):
        if x.shape[1] != skip.shape[1]:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        # print(x.shape,skip.shape)
        x = self.attention1(x)
        x = self.conv1(x)

        x = torch.cat([x, skip], dim=1)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.attention2(x)

        return x


class DBlock_res(nn.Module):

    def __init__(self, in_channels, out_channels, use_batchnorm=True, attention_type=None):

        super(DBlock_res, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        if attention_type == 'scse':
            self.attention1 = scSEBlock(in_channels)
        elif attention_type == 'cbam':
            self.attention1 = CbamModule(channels=in_channels)

        elif attention_type == 'transformer':

            self.attention1 = nn.Identity()


        else:
            self.attention1 = nn.Identity()

        self.conv2 = \
            nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        if attention_type == 'scse':
            self.attention2 = scSEBlock(out_channels)
        elif attention_type == 'cbam':
            self.attention2 = CbamModule(channels=out_channels)

        elif attention_type == 'transformer':
            self.attention2 = nn.Identity()

        else:
            self.attention2 = nn.Identity()

    def forward(self, x, skip):

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        # print(x.shape,skip.shape)
        x = self.attention1(x)
        x = self.conv1(x)

        x = torch.cat([x, skip], dim=1)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.attention2(x)

        return x


class DBlock_att(nn.Module):

    def __init__(self, in_channels, out_channels, use_batchnorm=True, attention_type='transformer'):

        super(DBlock_att, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        if attention_type == 'scse':
            self.attention1 = scSEBlock(in_channels)
        elif attention_type == 'cbam':
            self.attention1 = CbamModule(channels=in_channels)

        elif attention_type == 'transformer':

            self.attention1 = nn.Identity()


        else:
            self.attention1 = nn.Identity()

        self.conv2 = \
            nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        if attention_type == 'scse':
            self.attention2 = scSEBlock(out_channels)
        elif attention_type == 'cbam':
            self.attention2 = CbamModule(channels=out_channels)

        elif attention_type == 'transformer':
            self.attention2 = nn.Identity()

        else:
            self.attention2 = nn.Identity()

    def forward(self, x, skip):
        if x.shape[1] != skip.shape[1]:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        # print(x.shape,skip.shape)
        x = self.attention1(x)
        x = self.conv1(x)

        x = torch.cat([x, skip], dim=1)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.attention2(x)

        return x


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_class, kernel_size=3, upsample=4):
        super(SegmentationHead, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=upsample) if upsample > 1 else nn.Identity()
        self.conv = nn.Conv2d(in_channels, num_class, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class AV_Cross(nn.Module):

    def __init__(self, channels=2, r=2, residual=True, block=4, kernel_size=1):
        super(AV_Cross, self).__init__()
        out_channels = int(channels // r)
        self.residual = residual
        self.block = block
        self.bn = nn.BatchNorm2d(3)
        self.relu = False
        self.kernel_size = kernel_size
        self.a_ve_att = nn.ModuleList()
        self.v_ve_att = nn.ModuleList()
        self.ve_att = nn.ModuleList()
        for i in range(self.block):
            self.a_ve_att.append(nn.Sequential(
                nn.Conv2d(channels, out_channels, kernel_size=self.kernel_size, stride=1,
                          padding=(self.kernel_size - 1) // 2),
                nn.BatchNorm2d(out_channels),
            ))
            self.v_ve_att.append(nn.Sequential(
                nn.Conv2d(channels, out_channels, kernel_size=self.kernel_size, stride=1,
                          padding=(self.kernel_size - 1) // 2),
                nn.BatchNorm2d(out_channels),
            ))
            self.ve_att.append(nn.Sequential(
                nn.Conv2d(3, out_channels, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2),
                nn.BatchNorm2d(out_channels),
            ))
        self.sigmoid = nn.Sigmoid()
        self.final = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        a, ve, v = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:, :, :]
        for i in range(self.block):
            # x = self.relu(self.bn(x))
            a_ve = torch.concat([a, ve], dim=1)
            v_ve = torch.concat([v, ve], dim=1)
            a_v_ve = torch.concat([a, ve, v], dim=1)
            x_a = self.a_ve_att[i](a_ve)
            x_v = self.v_ve_att[i](v_ve)
            x_a_v = self.ve_att[i](a_v_ve)
            a_weight = self.sigmoid(x_a)
            v_weight = self.sigmoid(x_v)
            ve_weight = self.sigmoid(x_a_v)
            if self.residual:
                a = a + a * a_weight
                v = v + v * v_weight
                ve = ve + ve * ve_weight
            else:
                a = a * a_weight
                v = v * v_weight
                ve = ve * ve_weight

        out = torch.concat([a, ve, v], dim=1)

        if self.relu:
            out = F.relu(out)
        out = self.final(out)
        return out


class AV_Cross_v2(nn.Module):

    def __init__(self, channels=2, r=2, residual=True, block=1, relu=False, kernel_size=1):
        super(AV_Cross_v2, self).__init__()
        out_channels = int(channels // r)
        self.residual = residual
        self.block = block
        self.relu = relu
        self.kernel_size = kernel_size
        self.a_ve_att = nn.ModuleList()
        self.v_ve_att = nn.ModuleList()
        self.ve_att = nn.ModuleList()
        for i in range(self.block):
            self.a_ve_att.append(nn.Sequential(
                nn.Conv2d(channels, out_channels, kernel_size=self.kernel_size, stride=1,
                          padding=(self.kernel_size - 1) // 2),
                nn.BatchNorm2d(out_channels)
            ))
            self.v_ve_att.append(nn.Sequential(
                nn.Conv2d(channels, out_channels, kernel_size=self.kernel_size, stride=1,
                          padding=(self.kernel_size - 1) // 2),
                nn.BatchNorm2d(out_channels)
            ))
            self.ve_att.append(nn.Sequential(
                nn.Conv2d(channels, out_channels, kernel_size=self.kernel_size, stride=1,
                          padding=(self.kernel_size - 1) // 2),
                nn.BatchNorm2d(out_channels)
            ))

        self.sigmoid = nn.Sigmoid()
        self.final = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        a, ve, v = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:, :, :]

        for i in range(self.block):
            tmp = torch.cat([a, ve, v], dim=1)
            a_ve = torch.concat([a, ve], dim=1)
            a_ve = torch.cat([torch.max(a_ve, dim=1, keepdim=True)[0], torch.mean(a_ve, dim=1, keepdim=True)], dim=1)
            v_ve = torch.concat([v, ve], dim=1)
            v_ve = torch.cat([torch.max(v_ve, dim=1, keepdim=True)[0], torch.mean(v_ve, dim=1, keepdim=True)], dim=1)
            a_v_ve = torch.concat([torch.max(tmp, dim=1, keepdim=True)[0], torch.mean(tmp, dim=1, keepdim=True)], dim=1)

            a_ve = self.a_ve_att[i](a_ve)
            v_ve = self.v_ve_att[i](v_ve)
            a_v_ve = self.ve_att[i](a_v_ve)
            a_weight = self.sigmoid(a_ve)
            v_weight = self.sigmoid(v_ve)
            ve_weight = self.sigmoid(a_v_ve)
            if self.residual:
                a = a + a * a_weight
                v = v + v * v_weight
                ve = ve + ve * ve_weight
            else:
                a = a * a_weight
                v = v * v_weight
                ve = ve * ve_weight

        out = torch.concat([a, ve, v], dim=1)

        if self.relu:
            out = F.relu(out)
        out = self.final(out)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = (embedding_dim // head_num) ** (1 / 2)

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, mask=None):
        qkv = self.qkv_layer(x)

        query, key, value = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num))
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk

        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.out_attention(x)

        return x


class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.mlp_layers(x)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        _x = self.multi_head_attention(x)
        _x = self.dropout(_x)
        x = x + _x
        x = self.layer_norm1(x)

        _x = self.mlp(x)
        x = x + _x
        x = self.layer_norm2(x)

        return x


class TransformerEncoder(nn.Module):
    """
    embedding_dim: token 向量长度
    head_num: 自注意力头
    block_num: transformer个数
    """

    def __init__(self, embedding_dim, head_num, block_num=2):
        super().__init__()
        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embedding_dim, head_num, 2 * embedding_dim) for _ in range(block_num)])

    def forward(self, x):
        for layer_block in self.layer_blocks:
            x = layer_block(x)
        return x


class PathEmbedding(nn.Module):
    """
    img_dim: 输入图的大小
    in_channels: 输入的通道数
    embedding_dim: 每个token的向量长度
    patch_size：输入图token化，token的大小
    """

    def __init__(self, img_dim, in_channels, embedding_dim, patch_size):
        super().__init__()

        self.patch_size = patch_size
        self.num_tokens = (img_dim // patch_size) ** 2
        self.token_dim = in_channels * (patch_size ** 2)
        # 1. projection
        self.projection = nn.Linear(self.token_dim, embedding_dim)
        # 2. position embedding
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))
        # 3. cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

    def forward(self, x):
        img_patches = rearrange(x,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_size, patch_y=self.patch_size)

        batch_size, tokens_num, _ = img_patches.shape

        patch_token = self.projection(img_patches)
        cls_token = repeat(self.cls_token, 'b ... -> (b batch_size) ...',
                           batch_size=batch_size)

        patches = torch.cat([cls_token, patch_token], dim=1)
        # add postion embedding
        patches += self.embedding[:tokens_num + 1, :]

        # B,tokens_num+1,embedding_dim
        return patches


class TransformerBottleNeck(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num,
                 block_num, patch_size=1, classification=False, dropout=0.1, num_classes=1):
        super().__init__()
        self.patch_embedding = PathEmbedding(img_dim, in_channels, embedding_dim, patch_size)
        self.transformer = TransformerEncoder(embedding_dim, head_num, block_num)
        self.dropout = nn.Dropout(dropout)
        self.classification = classification
        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]
        return x


class PGFusion(nn.Module):

    def __init__(self, in_channel=384, out_channel=384):

        super(PGFusion, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.patch_query = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.patch_key = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.patch_value = nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False)
        self.patch_global_query = nn.Conv2d(in_channel, in_channel, kernel_size=1)

        self.global_key = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.global_value = nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False)

        self.fusion = nn.Conv2d(in_channel * 2, in_channel * 2, kernel_size=1)

        self.out_patch = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.out_global = nn.Conv2d(in_channel, out_channel, kernel_size=1)

        self.softmax = nn.Softmax(dim=2)
        self.softmax_concat = nn.Softmax(dim=0)

        self.gamma_patch_self = nn.Parameter(torch.ones(1))
        self.gamma_patch_global = nn.Parameter(torch.ones(1))

        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    # nn.init.constant_(m.bias, 0)
                m.inited = True

    def forward(self, patch_rep, global_rep):
        patch_rep_ = patch_rep.clone()
        patch_value = self.patch_value(patch_rep)
        patch_value = patch_value.view(patch_value.size(0), patch_value.size(1), -1)
        patch_key = self.patch_key(patch_rep)
        patch_key = patch_key.view(patch_key.size(0), patch_key.size(1), -1)
        dim_k = patch_key.shape[-1]
        patch_query = self.patch_query(patch_rep)
        patch_query = patch_query.view(patch_query.size(0), patch_query.size(1), -1)

        patch_global_query = self.patch_global_query(patch_rep)
        patch_global_query = patch_global_query.view(patch_global_query.size(0), patch_global_query.size(1), -1)

        global_value = self.global_value(global_rep)
        global_value = global_value.view(global_value.size(0), global_value.size(1), -1)
        global_key = self.global_key(global_rep)
        global_key = global_key.view(global_key.size(0), global_key.size(1), -1)

        ### patch self attention
        patch_self_sim_map = patch_query @ patch_key.transpose(-2, -1) / math.sqrt(dim_k)
        patch_self_sim_map = self.softmax(patch_self_sim_map)
        patch_self_sim_map = patch_self_sim_map @ patch_value
        patch_self_sim_map = patch_self_sim_map.view(patch_self_sim_map.size(0), patch_self_sim_map.size(1),
                                                     *patch_rep.size()[2:])
        patch_self_sim_map = self.gamma_patch_self * patch_self_sim_map
        # patch_self_sim_map = 1 * patch_self_sim_map
        ### patch global attention
        patch_global_sim_map = patch_global_query @ global_key.transpose(-2, -1) / math.sqrt(dim_k)
        patch_global_sim_map = self.softmax(patch_global_sim_map)
        patch_global_sim_map = patch_global_sim_map @ global_value
        patch_global_sim_map = patch_global_sim_map.view(patch_global_sim_map.size(0), patch_global_sim_map.size(1),
                                                         *patch_rep.size()[2:])
        patch_global_sim_map = self.gamma_patch_global * patch_global_sim_map
        # patch_global_sim_map = 1 * patch_global_sim_map

        fusion_sim_weight_map = torch.cat((patch_self_sim_map, patch_global_sim_map), dim=1)
        fusion_sim_weight_map = self.fusion(fusion_sim_weight_map)
        fusion_sim_weight_map = 1 * fusion_sim_weight_map

        patch_self_sim_weight_map = torch.split(fusion_sim_weight_map, dim=1, split_size_or_sections=self.in_channel)[0]
        patch_self_sim_weight_map = torch.sigmoid(patch_self_sim_weight_map)  # 0-1

        patch_global_sim_weight_map = torch.split(fusion_sim_weight_map, dim=1, split_size_or_sections=self.in_channel)[
            1]
        patch_global_sim_weight_map = torch.sigmoid(patch_global_sim_weight_map)  # 0-1

        patch_self_sim_weight_map = torch.unsqueeze(patch_self_sim_weight_map, 0)
        patch_global_sim_weight_map = torch.unsqueeze(patch_global_sim_weight_map, 0)

        ct = torch.concat((patch_self_sim_weight_map, patch_global_sim_weight_map), 0)
        ct = self.softmax_concat(ct)

        out = patch_rep_ + patch_self_sim_map * ct[0] + patch_global_sim_map * (1 - ct[0])

        return out


if __name__ == '__main__':
    x = torch.randn((2, 384, 16, 16))
    m = PGFusion()
    print(m)
    # y = TransformerBottleNeck(x.shape[2],x.shape[1],x.shape[1],8,4)
    print(m(x, x).shape)
