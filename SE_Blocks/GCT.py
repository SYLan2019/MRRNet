import time

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np
import cv2
from models import CBAM
from utils import utils


class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate


class NormLayer(nn.Module):
    """Normalization Layers.
    ------------
    # Arguments
        - channels: input channels, for batch norm and instance norm.
        - input_size: input shape without batch size, for layer norm.
    """

    def __init__(self, channels, normalize_shape=None, norm_type='bn'):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=True)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(32, channels, affine=True)
        elif norm_type == 'pixel':
            self.norm = lambda x: F.normalize(x, p=2, dim=1)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(normalize_shape)
        elif norm_type == 'none':
            self.norm = lambda x: x
        else:
            assert 1 == 0, 'Norm type {} not support.'.format(norm_type)

    def forward(self, x):
        return self.norm(x)


class ReluLayer(nn.Module):
    """Relu Layer.
    ------------
    # Arguments
        - relu type: type of relu layer, candidates are
            - ReLU
            - LeakyReLU: default relu slope 0.2
            - PRelu
            - SELU
            - none: direct pass
    """

    def __init__(self, channels, relu_type='relu'):
        super(ReluLayer, self).__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'selu':
            self.func = nn.SELU(True)
        elif relu_type == 'none':
            self.func = lambda x: x
        else:
            assert 1 == 0, 'Relu type {} not support.'.format(relu_type)

    def forward(self, x):
        return self.func(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale='none', norm_type='none', relu_type='none',
                 use_pad=True):
        super(ConvLayer, self).__init__()
        self.scale = scale
        self.use_pad = use_pad

        bias = True if norm_type in ['pixel', 'none'] else False
        stride = 2 if scale == 'down' else 1

        self.scale_func = lambda x: x
        if scale == 'up':
            self.scale_func = lambda x: nn.functional.interpolate(x, scale_factor=2, mode='nearest')

        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

        self.norm = NormLayer(out_channels, norm_type=norm_type)
        self.relu = ReluLayer(out_channels, relu_type)

    def forward(self, x):
        out = self.scale_func(x)
        # if self.scale == "up":
        #     print(out.shape)
        if self.use_pad:
            out = self.reflection_pad(out)
            # print("--",out.shape)
        out = self.conv2d(out)
        out = self.norm(out)
        out = self.relu(out)
        return out


class ResidualBlock(nn.Module):
    """
    Residual block recommended in: http://torch.ch/blog/2016/02/04/resnets.html
    ------------------
    # Args
        - hg_depth: depth of HourGlassBlock. 0: don't use attention map.
        - use_pmask: whether use previous mask as HourGlassBlock input.
    """

    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none', hg_depth=2, att_name='spar',
                 isPrint=False):
        super(ResidualBlock, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.hg_depth = hg_depth

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)

        self.preact_func = nn.Sequential(
            NormLayer(c_in, norm_type=self.norm_type),
            ReluLayer(c_in, self.relu_type),
        )

        if scale == 'down':
            scales = ['none', 'down']
        elif scale == 'up':
            scales = ['up', 'none']
        elif scale == 'none':
            scales = ['none', 'none']

        self.conv1 = ConvLayer(c_in, c_out, 3, scales[0], **kwargs)
        self.conv2 = ConvLayer(c_out, c_out, 3, scales[1], norm_type=norm_type, relu_type='none')

        if att_name.lower() == 'spar':
            c_attn = 1
        elif att_name.lower() == 'spar3d':
            c_attn = c_out
        else:
            raise Exception("Attention type {} not implemented".format(att_name))

        self.att_func = HourGlassBlock(self.hg_depth, c_out, c_attn, **kwargs, isPrint=isPrint)

    def forward(self, x):
        identity = self.shortcut_func(x)
        out = self.preact_func(x)
        out = self.conv1(out)
        out = self.conv2(out)

        # ------------------------------------ 2021/10/11 注释掉,为了测试noInception noGAN 的baseline, 消融实验
        # print("==========",identity.shape,)
        # out = identity + self.att_func(out)
        # ------------------------------------

        # new base_line net, no self.att_func() 2021/10/11 被启动 。。。 消融实验
        out = identity + self.att_func(out)
        # out = identity + out
        return out



class HourGlassBlock(nn.Module):
    """Simplified HourGlass block.
    Reference: https://github.com/1adrianb/face-alignment
    --------------------------
    """

    def __init__(self, depth, c_in, c_out,
                 c_mid=64,
                 norm_type='bn',
                 relu_type='prelu',
                 isPrint=False,
                 ):
        super(HourGlassBlock, self).__init__()
        self.depth = depth
        self.c_in = c_in
        self.c_mid = c_mid
        self.c_out = c_out
        self.isPrint = isPrint
        # print("=========", self.c_mid)
        self.kwargs = {'norm_type': norm_type, 'relu_type': relu_type}
        self.inception = Inception_Module(in_c=c_in, out_c=c_in)  # new

        # CBAM start
        # self.CBAM=CBAM.CBAM(gate_channels=c_in)
        # CBAM END
        # 注释-改动
        # if self.depth:
        #     self._generate_network(self.depth)
        #     self.out_block = nn.Sequential(
        #         ConvLayer(self.c_mid, self.c_out, norm_type='none', relu_type='none'),
        #         nn.Sigmoid()
        #     )
        self.out_block = nn.Sequential(
            ConvLayer(self.c_in, self.c_out, norm_type='none', relu_type='none'),
            nn.Sigmoid()  # output range 0-1 feature map of 1 dimension
        )

    # origin
    def forward(self, x, pmask=None):
        if self.depth == 0: return x
        input_x = x
        # 注释-改动
        # x = self._forward(self.depth, x)
        x = self.inception(x)
        # print("x shape :======", x.shape)
        self.att_map = self.out_block(x)
        if self.isPrint:
            print("---------hwk------------------")
            temp_map = self.out_block(x)  # range = 0-1
            output_weight = utils.tensor_to_img2(temp_map, normal=True)
            color_att_map = cv2.applyColorMap(output_weight, cv2.COLORMAP_JET)
            fileName = time.strftime("%d-%H-%M-%S", time.localtime()) + ".jpg"
            cv2.imwrite("results_helen/map/" + fileName, color_att_map)
        # print("att_map shape :======", self.att_map.shape)
        x = input_x * self.att_map
        return x

    # replace our inception by CBAM
    # def forward(self, x, pmask=None):
    #     if self.depth == 0: return x
    #     # input_x = x
    #     # 注释-改动
    #     # x = self._forward(self.depth, x)
    #     # x = self.inception(x)
    #
    #     #CBAM--- start
    #     out=self.CBAM(x)
    #     return out


class Inception_Module(nn.Module):  # try reflection pad
    def __init__(self, in_c, out_c):
        super(Inception_Module, self).__init__()
        self.feature_extract_1 = nn.Sequential(
            nn.Conv2d(in_c, out_c // 4, 1, 1)
        )

        self.feature_extract_3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_c, out_c // 4, 3, 1),
        )

        self.feature_extract_5 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_c, out_c // 4, 5, 1),
        )

        self.feature_extract_7 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_c, out_c // 4, 7, 1),
        )

        # self.fc1 = nn.Linear(out_c, out_c)
        # self.fc2 = nn.Linear(out_c, out_c)
        # self.out = out_c

        # self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)
        self.in_norm = nn.InstanceNorm2d(out_c)
        # self.sigmoid = nn.Sigmoid()
        # self.maxpool = nn.AdaptiveMaxPool2d(1)
        # self.avg = nn.AdaptiveAvgPool2d(1)
        self.GCT = GCT(out_c)

    def forward(self, x):

        # kernel size 1 3
        feature_1 = self.feature_extract_1(x)
        feature_3 = self.feature_extract_3(x)
        feature_5 = self.feature_extract_5(x)
        feature_7 = self.feature_extract_7(x)

        feature_fusion = torch.cat([feature_1, feature_3, feature_5, feature_7], dim=1)
        feature_fusion = self.in_norm(feature_fusion)
        F.relu(feature_fusion, inplace=True)

        out = self.GCT(feature_fusion)
        return out



# patch-discriminator
class Discriminator(nn.Module):
    def __init__(self, conv_dim, norm_fun, act_fun, use_sn, adv_loss_type):
        super(Discriminator, self).__init__()

        # scale 1 and prediction of scale 1                 32
        d_1 = [dis_conv_block(in_channels=3, out_channels=conv_dim, kernel_size=3, stride=2, padding=3, dilation=1,
                              use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        # d_1_pred = [
        #     dis_pred_conv_block(in_channels=conv_dim, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1,
        #                         use_bias=False, type=adv_loss_type)]

        # scale 2                                                         64
        d_2 = [dis_conv_block(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=3, stride=2, padding=3,
                              dilation=1, norm_fun=norm_fun, use_bias=True, act_fun=act_fun, use_sn=use_sn)]
        # d_2_pred = [dis_pred_conv_block(in_channels=conv_dim * 2, out_channels=1, kernel_size=7, stride=1, padding=3,
        #                                 dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 3 and prediction of scale 3                               128
        d_3 = [dis_conv_block(in_channels=conv_dim * 2, out_channels=conv_dim * 4, kernel_size=3, stride=2, padding=3,
                              dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        # d_3_pred = [dis_pred_conv_block(in_channels=conv_dim * 4, out_channels=1, kernel_size=7, stride=1, padding=3,
        #                                 dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 4  16
        d_4 = [dis_conv_block(in_channels=conv_dim * 4, out_channels=conv_dim * 8, kernel_size=3, stride=2, padding=2,
                              dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_4_pred = [dis_pred_conv_block(in_channels=conv_dim * 8, out_channels=1, kernel_size=3, stride=1, padding=2,
                                        dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 5 and prediction of scale 5         8
        # d_5 = [dis_conv_block(in_channels=conv_dim * 8, out_channels=conv_dim * 16, kernel_size=5, stride=2, padding=2,
        #                       dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        # d_5_pred = [dis_pred_conv_block(in_channels=conv_dim * 16, out_channels=1, kernel_size=5, stride=1, padding=2,
        #                                 dilation=1, use_bias=False, type=adv_loss_type)]

        self.d1 = nn.Sequential(*d_1)
        # self.d1_pred = nn.Sequential(*d_1_pred)
        self.d2 = nn.Sequential(*d_2)
        # self.d2_pred = nn.Sequential(*d_2_pred)
        self.d3 = nn.Sequential(*d_3)
        # self.d3_pred = nn.Sequential(*d_3_pred)
        self.d4 = nn.Sequential(*d_4)
        self.d4_pred = nn.Sequential(*d_4_pred)
        # self.d5 = nn.Sequential(*d_5)
        # self.d5_pred = nn.Sequential(*d_5_pred)

    def forward(self, x):
        ds1 = self.d1(x)
        # ds1_pred = self.d1_pred(ds1)

        ds2 = self.d2(ds1)
        # ds2_pred = self.d2_pred(ds2)

        ds3 = self.d3(ds2)
        # ds3_pred = self.d3_pred(ds3)

        ds4 = self.d4(ds3)
        ds4_pred = self.d4_pred(ds4)

        # ds5 = self.d5(ds4)
        # ds5_pred = self.d5_pred(ds5)

        # return [ds1_pred, ds2_pred, ds3_pred, ds4_pred, ds5_pred]

        return ds4_pred


def dis_conv_block(in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, norm_fun, act_fun,
                   use_sn):
    padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
    main = []
    main.append(nn.ReflectionPad2d(padding))
    main.append(SpectralNorm(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0,
                  dilation=dilation, bias=use_bias), use_sn))
    # norm_fun = get_norm_fun(norm_fun)
    # main.append(norm_fun(out_channels))
    # main.append(get_act_fun(act_fun))
    # act_fun = nn.LeakyReLU(0.2)
    main.append(nn.LeakyReLU(0.2))
    main = nn.Sequential(*main)
    return main


def dis_pred_conv_block(in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, type):
    padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
    main = []
    main.append(nn.ReflectionPad2d(padding))
    main.append(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0,
                  dilation=dilation, bias=use_bias))
    if type in ['ls', 'rals']:
        main.append(nn.Sigmoid())
    elif type in ['hinge', 'rahinge']:
        main.append(nn.Tanh())
    else:
        raise NotImplementedError("Adversarial loss [{}] is not found".format(type))
    main = nn.Sequential(*main)
    return main


def SpectralNorm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module
