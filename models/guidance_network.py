from functools import partial
from typing import Callable
import torch as th
import torch.nn as nn
from models.unet_model_google import ResnetBlock, Swish, exists, Block, Downsample, Upsample
import torch.nn.functional as F

from utils.metric import AnalysisPanAcc


class SElayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SElayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GuidanceResBlock(nn.Module):
    def __init__(self, dim, dim_out, dropout=0):
        super().__init__()
        # @Block does not downsample
        self.block1 = Block(dim, dim_out, groups=1)
        self.block2 = Block(dim_out, dim_out, dropout=dropout, groups=1)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.se = SElayer(dim_out)

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        res = self.res_conv(x)
        h = self.se(h)
        return h + res

class ResBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        dropout=0,
        groups=1,
    ):
        super().__init__()

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)



class GuidanceNetworkBase(nn.Module):
    def forward(self, *args, mode="train", **kwargs):
        if mode == "train":
            return self.train_step(*args, **kwargs)
        elif mode == "val":
            return self.val_step(*args, **kwargs)
        else:
            raise NotImplemented("mode should be train or val")


# 多分支的结构
class GuidanceNetworkMultiBranch(GuidanceNetworkBase):
    def __init__(
        self,
        input_size: int,
        scales: tuple = (1, 2, 4, 8),
        img_channels=8,
        dim_range=((9, 32), (9, 32, 64), (9, 32, 64, 128), (9, 32, 64, 128)),
        drop_out: tuple = (0.0, 0.0, 0.2, 0.2),
    ) -> None:
        super().__init__()
        # 使用多分支的结构
        # 下采大小为[64, 32, 16, 8]
        assert (
            len(scales) == len(dim_range) == len(drop_out)
        ), "scales, dim_range, drop_out should have the same length"
        size_s = [input_size // s for s in scales]
        self.size_s = size_s
        self.dim_range = dim_range
        self.scale_brances = nn.ModuleList()

        for i, s in enumerate(size_s):
            print("guidance network init - guidance size: {}".format(s))
            one_branch = nn.Sequential()
            one_branch.append(nn.AdaptiveAvgPool2d(s))
            for j in range(len(dim_range[i]) - 1):
                up_dim = (dim_range[i][j], dim_range[i][j + 1])
                one_branch.append(
                    # f"up_dim_conv_{up_dim[0]}->{up_dim[1]}",
                    GuidanceResBlock(up_dim[0], up_dim[1], dropout=drop_out[i]),
                )
            # self.scale_brances.add_module(f"branch_scale_{scales[i]}", one_branch)
            self.scale_brances.append(one_branch)

        self.proj_img_conv = nn.ModuleList()
        for i in range(len(self.scale_brances)):
            branch_channels = dim_range[i][-1]
            self.proj_img_conv.append(
                # f"proj_img_conv_{size_s[i]}",
                # nn.Conv2d(branch_channels, img_channels, 1),
                nn.Sequential(
                    Block(branch_channels, branch_channels, groups=1),
                    Block(branch_channels, img_channels, groups=1),
                )
            )

    def _forward_imple(self, x):
        guidances = []
        recons = []
        for branch, proj in zip(self.scale_brances, self.proj_img_conv):
            g = branch(x)
            guidances.append(g)  # detach gradient for guide diffusion
            # print(g.shape)
            r = proj(g)
            recons.append(r)

        return guidances, recons

    def train_step(
        self, x, gt, criterion: Callable
    ) -> tuple[th.Tensor, th.Tensor, list[th.Tensor]]:
        gts = [F.interpolate(gt, size=s) for s in self.size_s]
        gs, rs = self._forward_imple(x)
        losses = []
        for r, grt in zip(rs, gts):
            l = criterion(r, grt)
            losses.append(l)
        return gs, rs, losses

    def val_step(self, x) -> tuple[list[th.Tensor], list[th.Tensor]]:
        gs, rs = self._forward_imple(x)
        return gs, rs


# 单分支的结构
class GuidanceNetworkSingleBranch(GuidanceNetworkBase):
    def __init__(self, img_channel, dim_range=(9, 32, 64, 128, 128)) -> None:
        super().__init__()
        self.enc = nn.ModuleList()
        n_dims = len(dim_range)
        for i in range(n_dims - 1):
            self.enc.append(
                nn.Sequential(
                    GuidanceResBlock(dim_range[i], dim_range[i + 1]),
                    Downsample(dim_range[i + 1]) if i != 0 else nn.Identity(),
                )
            )

        self.dec = nn.ModuleList()

        dec_dim_range = list(reversed(dim_range))
        dec_dim_range[-1] = img_channel
        for i in range(n_dims - 1):
            self.dec.append(
                nn.Sequential(
                    ResBlock(dec_dim_range[i], dec_dim_range[i + 1], groups=1),
                    Upsample(dec_dim_range[i + 1]) if i != 0 else nn.Identity(),
                )
            )

    def _forward_implem(self, x):
        h = x
        fms = []
        for enc in self.enc:
            h = enc(h)
            # print(h.shape)
            fms.append(h)
        for dec in self.dec:
            h = dec(h)
        return h, fms

    def train_step(self, x, gt, criterion):
        h, fms = self._forward_implem(x)
        h = h + x[:, :8]
        loss = criterion(h, gt)
        return fms, h.detach(), loss

    def val_step(self, x):
        h, fms = self._forward_implem(x)
        h = h + x[:, :8]
        # print(h.shape)
        return fms, h


if __name__ == "__main__":
    from copy import deepcopy
    import h5py
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import torch
    import torch.nn as nn
    import torchvision as tv
    import tqdm
    from scipy.io import savemat
    from torch.utils.data import DataLoader
    
    from dataset.pan_dataset import DatasetUsed
    
    # net = GuidanceNetworkMultiBranch(64)
    net = GuidanceNetworkSingleBranch(8).cuda()
    x = th.randn(1, 9, 64, 64).cuda()
    # print(net)
    print(list(map(lambda x: x.shape, net(x, mode="val")[0])))
    # from fvcore.nn import FlopCountAnalysis, flop_count_table
    # analysis = FlopCountAnalysis(net, (x,))
    # print(
    #     flop_count_table(analysis)
    # )
    opt_g = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-7)
    
    train_dataset_path = "/Data2/DataSet/pansharpening_2/training_data/train_wv3.h5"
    valid_dataset_path = "/Data2/DataSet/pansharpening_2/validation_data/valid_wv3.h5"
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
        drop_last=True,
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=128,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=True,
    )
    
    g_loss_fn = nn.L1Loss()
    analysis_g = AnalysisPanAcc()

    for ep in range(500):
        for i, (pan, ms, lms, hr) in enumerate(dl_train, 1):
            x = torch.cat([lms, pan], dim=1).cuda()
            hr = hr.cuda()
            _, sr, g_loss = net.train_step(x, hr, g_loss_fn)
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
                    _, sr = net.val_step(x)
                    analysis_g(sr, hr)
            print(analysis_g.print_str())
            net.train()