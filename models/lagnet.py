import torch
import torch.nn as nn
from torch.nn import functional as F


class LAConv2D(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        use_bias=True,
    ):
        super(LAConv2D, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = use_bias

        # Generating local adaptive weights
        self.attention1 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_size ** 2, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_size ** 2, kernel_size ** 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_size ** 2, kernel_size ** 2, 1),
            nn.Sigmoid(),
        )  # b,9,H,W È«Í¨µÀÏñËØ¼¶ºË×¢ÒâÁ¦
        # self.attention2=nn.Sequential(
        #     nn.Conv2d(in_planes,(kernel_size**2)*in_planes,kernel_size, stride, padding,groups=in_planes),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d((kernel_size**2)*in_planes,(kernel_size**2)*in_planes,1,groups=in_planes),
        #     nn.Sigmoid()
        # ) #b,9n,H,W µ¥Í¨µÀÏñËØ¼¶ºË×¢ÒâÁ¦
        if use_bias == True:  # Global local adaptive weights
            self.attention3 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_planes, out_planes, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_planes, out_planes, 1),
            )  # b,m,1,1 Í¨µÀÆ«ÖÃ×¢ÒâÁ¦

        conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size, stride, padding, dilation, groups
        )
        self.weight = conv1.weight  # m, n, k, k

    def forward(self, x):
        (b, n, H, W) = x.shape
        m = self.out_planes
        k = self.kernel_size
        n_H = 1 + int((H + 2 * self.padding - k) / self.stride)
        n_W = 1 + int((W + 2 * self.padding - k) / self.stride)
        atw1 = self.attention1(x)  # b,k*k,n_H,n_W
        # atw2=self.attention2(x) #b,n*k*k,n_H,n_W

        atw1 = atw1.permute([0, 2, 3, 1])  # b,n_H,n_W,k*k
        atw1 = atw1.unsqueeze(3).repeat([1, 1, 1, n, 1])  # b,n_H,n_W,n,k*k
        atw1 = atw1.view(b, n_H, n_W, n * k * k)  # b,n_H,n_W,n*k*k

        # atw2=atw2.permute([0,2,3,1]) #b,n_H,n_W,n*k*k

        atw = atw1  # *atw2 #b,n_H,n_W,n*k*k
        atw = atw.view(b, n_H * n_W, n * k * k)  # b,n_H*n_W,n*k*k
        atw = atw.permute([0, 2, 1])  # b,n*k*k,n_H*n_W

        kx = F.unfold(
            x, kernel_size=k, stride=self.stride, padding=self.padding
        )  # b,n*k*k,n_H*n_W
        atx = atw * kx  # b,n*k*k,n_H*n_W

        atx = atx.permute([0, 2, 1])  # b,n_H*n_W,n*k*k
        atx = atx.view(1, b * n_H * n_W, n * k * k)  # 1,b*n_H*n_W,n*k*k

        w = self.weight.view(m, n * k * k)  # m,n*k*k
        w = w.permute([1, 0])  # n*k*k,m
        y = torch.matmul(atx, w)  # 1,b*n_H*n_W,m
        y = y.view(b, n_H * n_W, m)  # b,n_H*n_W,m
        if self.bias == True:
            bias = self.attention3(x)  # b,m,1,1
            bias = bias.view(b, m).unsqueeze(1)  # b,1,m
            bias = bias.repeat([1, n_H * n_W, 1])  # b,n_H*n_W,m
            y = y + bias  # b,n_H*n_W,m

        y = y.permute([0, 2, 1])  # b,m,n_H*n_W
        y = F.fold(y, output_size=(n_H, n_W), kernel_size=1)  # b,m,n_H,n_W
        return y


# LAC_ResBlocks
class LACRB(nn.Module):
    def __init__(self, in_planes):
        super(LACRB, self).__init__()
        self.conv1 = LAConv2D(in_planes, in_planes, 3, 1, 1, use_bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = LAConv2D(in_planes, in_planes, 3, 1, 1, use_bias=True)
        # self.bn = nn.BatchNorm2d(in_planes)  # 原先没有bn

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu1(res)
        res = self.conv2(res)
        # res = self.bn(res)
        x = x + res
        return x


# Proposed Network
class LACNET(nn.Module):
    def __init__(self, spectral_num=8, pan_channel_num=1):
        super(LACNET, self).__init__()
        self.spectral_num = spectral_num
        self.head_conv = nn.Sequential(
            LAConv2D(spectral_num + pan_channel_num, 32, 3, 1, 1, use_bias=True),
            nn.ReLU(inplace=True),
        )

        self.RB1 = LACRB(32)
        self.RB2 = LACRB(32)
        self.RB3 = LACRB(32)
        self.RB4 = LACRB(32)
        self.RB5 = LACRB(32)

        self.tail_conv = LAConv2D(32, spectral_num, 3, 1, 1, use_bias=True)

    def _forward_imple(self, lms, pan):
        x = torch.cat([lms, pan], 1)
        x = self.head_conv(x)
        x = self.RB1(x)
        x = self.RB2(x)
        x = self.RB3(x)
        x = self.RB4(x)
        x = self.RB5(x)
        x = self.tail_conv(x)
        sr = lms + x
        return sr

    def train_step(self, x, gt=None, criterion=None):
        # x = unorm(x)
        # gt = unorm(gt)
        lms = x[:, : self.spectral_num]
        pan = x[:, self.spectral_num :]
        sr = self._forward_imple(lms, pan)
        # sr = lms + res  # output:= lms + hp_sr
        # loss = criterion(sr, gt)
        return sr, 0.0

    def val_step(self, x):
        # x = unorm(x)
        lms = x[:, : self.spectral_num]
        pan = x[:, self.spectral_num :]
        sr = self._forward_imple(lms, pan)
        # sr = lms + res  # output:= lms + hp_sr

        return sr

    def forward(self, *args, mode="train", **kwargs):
        if mode == "train":
            return self.train_step(*args, **kwargs)
        elif mode == "val":
            return self.val_step(*args, **kwargs)
        else:
            raise NotImplemented("mode should be train or val")


if __name__ == "__main__":
    from torchsummary import summary
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    # N = LACNET()
    # summary(N, [(1, 64, 64), (8, 64, 64)], device="cpu")

    net = LACNET()

    lms = torch.randn(1, 8, 64, 64)
    pan = torch.randn(1, 1, 64, 64)
    x = torch.cat([pan, lms], 1)

    print(net(x)[0].shape)
    # print(flop_count_table(FlopCountAnalysis(net, (x,))))

