import torch
import torch.nn as nn
import torch as th
from models.sr3_dwt import Block

class ResBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            dropout=0,
            norm_groups=32,
    ):
        super().__init__()

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h - self.res_conv(x)

class BufferLayer(nn.Module):
    def __init__(self,in_c,num):
        super().__init__()
        self.encode = nn.Sequential(nn.Conv2d(in_c,3,1,1,0),nn.InstanceNorm2d(3))
        self.res_en = ResBlock(3*num,3*num,0.2,norm_groups=8)
        self.decode = nn.Sequential(nn.InstanceNorm2d(3),nn.Conv2d(3,in_c,1,1,0))
        self.res_de = ResBlock(num, num, 0.2,norm_groups=8)
    def forward(self,x,phase="encode"):
        if phase == "encode":
            h =[self.encode(x[:,i,:,:].unsqueeze(1)) for i in range(x.size(1))]
            out = torch.cat(h,dim=1)
            out = self.res_en(out)
            return out
        if phase == "decode":
            h = [self.decode(x[:, i:i+3, :, :]) for i in range(0,x.size(1)-2,3)]
            out = torch.cat(h, dim=1)
            out = self.res_de(out)
            return out

