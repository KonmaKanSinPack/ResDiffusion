from copy import deepcopy
from functools import partial
import math
from typing import Callable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from scipy.io import savemat
from torch.utils.data import DataLoader

from dataset.pan_dataset import DatasetUsed
from utils.metric import AnalysisPanAcc


def norm(x):
    # x range is [0, 1]
    return x * 2 - 1


def unorm(x):
    # x range is [-1, 1]
    return (x + 1) / 2


class Block(nn.Module):
    def __init__(self, channel=32, ksize=3, padding=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=ksize,
            padding=padding,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=ksize,
            padding=padding,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        
        # self.apply(init_weights)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv1(x))  # Bsx32x64x64
        rs1 = self.conv2(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs


def variance_scaling_initializer(tensor):
    # stole it from woo-xiao.
    # thanks
    def calculate_fan(shape, factor=2.0, mode="FAN_IN", uniform=False):
        # 64 9 3 3 -> 3 3 9 64
        # 64 64 3 3 -> 3 3 64 64
        if shape:
            # fan_in = float(shape[1]) if len(shape) > 1 else float(shape[0])
            # fan_out = float(shape[0])
            fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
            fan_out = float(shape[-1])
        else:
            fan_in = 1.0
            fan_out = 1.0
        for dim in shape[:-2]:
            fan_in *= float(dim)
            fan_out *= float(dim)
        if mode == "FAN_IN":
            # Count only number of input connections.
            n = fan_in
        elif mode == "FAN_OUT":
            # Count only number of output connections.
            n = fan_out
        elif mode == "FAN_AVG":
            # Average number of inputs and output connections.
            n = (fan_in + fan_out) / 2.0
        if uniform:
            raise NotImplemented
            # # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
            # limit = math.sqrt(3.0 * factor / n)
            # return random_ops.random_uniform(shape, -limit, limit,
            #                                  dtype, seed=seed)
        else:
            # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
            trunc_stddev = math.sqrt(1.3 * factor / n)
        return fan_in, fan_out, trunc_stddev


def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):  ## initialization for Conv2d
                # print("initial nn.Conv2d with var_scale_new: ", m)
                # try:
                #     import tensorflow as tf
                #     tensor = tf.get_variable(shape=m.weight.shape, initializer=tf.variance_scaling_initializer(seed=1))
                #     m.weight.data = tensor.eval()
                # except:
                #     print("try error, run variance_scaling_initializer")
                # variance_scaling_initializer(m.weight)
                variance_scaling_initializer(m.weight)  # method 1: initialization
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # method 2: initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):  ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):  ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class FusionNet(nn.Module):
    def __init__(self, spectral_num, channel=32, n_blocks=2):
        super(FusionNet, self).__init__()
        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.spectral_num = spectral_num

        self.conv1 = nn.Conv2d(
            in_channels=spectral_num,
            out_channels=channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        # self.res1 = nn.Sequential(*[Block(channel=channel, ksize=3, padding=1) for _ in range(n_blocks)])
        # self.res2 = nn.Sequential(*[Block(channel=channel, ksize=5, padding=2) for _ in range(n_blocks)])
        # self.res3 = nn.Sequential(*[Block(channel=channel, ksize=3, padding=1) for _ in range(n_blocks)])
        # self.res4 = nn.Sequential(*[Block(channel=channel, ksize=3, padding=1) for _ in range(n_blocks)])
        
        self.res1 = Block(channel)
        self.res2 = Block(channel)
        self.res3 = Block(channel)
        self.res4 = Block(channel)

        self.conv3 = nn.Conv2d(
            in_channels=channel,
            out_channels=spectral_num,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1, self.res2, self.res3, self.res4
        )

        init_weights(
            self.backbone, self.conv1, self.conv3
        )  # state initialization, important!
        self.apply(init_weights)

    def _forward_imple(self, x, y):  # x= lms; y = pan

        pan_concat = y.repeat(1, self.spectral_num, 1, 1)  # Bsx8x64x64
        input = torch.sub(pan_concat, x)  # Bsx8x64x64
        rs = self.relu(self.conv1(input))  # Bsx32x64x64

        rs = self.backbone(rs)  # ResNet's backbone!
        output = self.conv3(rs)  # Bsx8x64x64

        return output  # lms + outs

    def train_step(self, x, gt=None, criterion=None):
        # x = unorm(x)
        # gt = unorm(gt)
        lms = x[:, :self.spectral_num]
        pan = x[:, self.spectral_num:]
        res = self._forward_imple(lms, pan)
        sr = lms + res  # output:= lms + hp_sr
        # loss = criterion(sr, gt)
        return sr, 0.

    def val_step(self, x):
        # x = unorm(x)
        lms = x[:, :self.spectral_num]
        pan = x[:, self.spectral_num:]
        res = self._forward_imple(lms, pan)
        sr = lms + res  # output:= lms + hp_sr
        
        return sr

    def forward(self, *args, mode="train", **kwargs):
        if self.training:
            mode = "train"
        else:
            mode = 'val'
            
        if mode == "train":
            return self.train_step(*args, **kwargs)
        elif mode == "val":
            return self.val_step(*args, **kwargs)
        else:
            raise NotImplemented("mode should be train or val")


if __name__ == "__main__":
    # net = GuidanceNetworkMultiBranch(64)
    net = FusionNet(8, 64).cuda()
    # x = torch.randn(1, 9, 64, 64).cuda()
    # print(net)
    # print(net(x, mode='val').shape)
    # from fvcore.nn import FlopCountAnalysis, flop_count_table
    # analysis = FlopCountAnalysis(net, (x,))
    # print(
    #     flop_count_table(analysis)
    # )
    # for m in net.modules():
    #     print(m, '------------', sep='\n')
    opt_g = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-7)

    train_dataset_path = "/Data2/DataSet/pansharpening_2/training_data/train_wv3.h5"
    valid_dataset_path = "/Data2/DataSet/pansharpening_2/test_data/WV3/test_wv3_multiExm1.h5"
    d_train = h5py.File(train_dataset_path)
    d_valid = h5py.File(valid_dataset_path)
    ds_train = DatasetUsed(d_train, full_res=False, norm_range=False)
    ds_valid = DatasetUsed(d_valid, full_res=False, norm_range=False)
    dl_train = DataLoader(
        ds_train,
        batch_size=128,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=128,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )

    g_loss_fn = nn.L1Loss()
    analysis_g = AnalysisPanAcc()

    for ep in range(500):
        for i, (pan, ms, lms, hr) in enumerate(dl_train, 1):
            x = torch.cat([lms, pan], dim=1).cuda()
            hr = hr.cuda()
            sr, g_loss = net.train_step(x, hr, g_loss_fn)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            print(f"ep: {ep}, step: {i}, g_loss: {g_loss.item()}")

        if ep % 10 == 0:
            net.eval()
            analysis_g.clear_history()
            with torch.no_grad():
                for i, (pan, ms, lms, hr) in enumerate(dl_valid, 1):
                    x = torch.cat([lms, pan], dim=1).cuda()
                    hr = hr.cuda()
                    sr = net.val_step(x)
                    analysis_g(sr, hr)
            print(analysis_g.print_str())
            net.train()
