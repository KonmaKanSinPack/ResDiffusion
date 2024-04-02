import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fusionnet import variance_scaling_initializer

class Resblock(nn.Module):
    def __init__(self, channel=32, ksize=3, padding=1):
        super(Resblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=ksize,
                               padding=padding,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=ksize,
                               padding=padding,
                               bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv1(x))  # Bsx32x64x64
        rs1 = self.conv2(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs

class PANNet(nn.Module):
    def __init__(self, in_c, hidden_c=32):
        super(PANNet, self).__init__()
        # self.ghpm = GaussianHighPassModule(in_c + 1, 64, 64)
        self.conv1 = nn.Conv2d(in_channels=in_c + 1, out_channels=hidden_c, kernel_size=3, stride=1, padding=1,
                               bias=True)

        # _res_block_params = dict(in_c=hidden_c, out_c=hidden_c, ksize=(3, 3), stride=1, pad=1)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()

        self.conv2 = nn.Conv2d(in_channels=hidden_c, out_channels=in_c, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.relu = nn.ReLU()
        self.up = nn.Upsample(scale_factor=4, mode='bilinear')
        # self.deconv = nn.ConvTranspose2d(in_channels=in_c, out_channels=in_c, kernel_size=8, stride=4,
        #                                  padding=2, bias=True)
        self.backbone = nn.Sequential(
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )

        self.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):  # initialization for Conv2d
            # print("nn.Conv2D is initialized by variance_scaling_initializer")
            variance_scaling_initializer(m.weight)  # method 1: initialization
            # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # method 2: initialization
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):  # initialization for BN
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):  # initialization for nn.Linear
            # variance_scaling_initializer(m.weight)
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, lms, pan):
        # output_deconv = self.deconv(ms)  # Bsx8x64x64

        input = torch.cat([lms, pan], 1)  # Bsx9x64x64
        rs = self.relu(self.conv1(input))  # Bsx32x64x64

        rs = self.backbone(rs)  # ResNet's backbone! # Bsx32x64x64

        output = self.conv2(rs)  # Bsx8x64x64
        return output

    def train_step(self, x, gt, criterion=None):
        lms = x[:, :self.spectral_num]
        pan = x[:, self.spectral_num:]
        
        hr = self(lms, pan)
        hr += lms
        loss = criterion(hr, gt)
        return hr, 0.

    def val_step(self, x):
        lms = x[:, :self.spectral_num]
        pan = x[:, self.spectral_num:]
        
        hr = self(lms, pan)
        hr += lms
        return hr