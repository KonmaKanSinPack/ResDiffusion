import math
from einops import rearrange
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction

# from diffusion_engine import norm


def base2fourier_features(inputs, freq_start=7, freq_stop=8, step=1): #做傅里叶变换？
    freqs = range(freq_start, freq_stop, step)
    w = (
        2.0 ** (torch.tensor(freqs, dtype=inputs.dtype, device=inputs.device))
        * 2
        * torch.pi
    )
    w = w[None, :].repeat(1, inputs.shape[1])

    # compute features
    h = inputs.repeat_interleave(len(freqs), dim=1)
    h = w[..., None, None] * h
    h = torch.cat([torch.sin(h), torch.cos(h)], dim=1)
    return h

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class UNetSR3(nn.Module):
    def __init__(
        self,
        in_channel=8,
        out_channel=3,
        inner_channel=32,
        cond_channel=8,
        h_x = 64,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8,),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128,
        self_condition=False,
        fourier_features=False,
        fourier_min=7,
        fourier_max=8,
        fourier_step=1,
        pred_var=False,
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel),
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        if self_condition:
            in_channel += out_channel
        if fourier_features:
            n = np.ceil((fourier_max - fourier_min) /
                        fourier_step).astype("int")
            in_channel += in_channel * n * 2

        self.fourier_features = fourier_features
        self.fourier_min = fourier_min
        self.fourier_max = fourier_max
        self.fourier_step = fourier_step

        self.pred_var = pred_var

        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = ind == num_mults - 1
            use_attn = now_res in attn_res
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(
                    ResnetBlocWithAttn(
                        pre_channel,
                        channel_mult,
                        h_x = h_x//2**ind,
                        cond_dim=cond_channel,
                        noise_level_emb_dim=noise_level_channel,
                        norm_groups=norm_groups,
                        dropout=dropout,
                        with_attn=use_attn,
                    )
                )
                feat_channels.append(channel_mult)
                pre_channel = channel_mult

            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList(
            [
                ResnetBlocWithAttn(
                    pre_channel,
                    pre_channel,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout,
                    with_attn=True,
                ),
                ResnetBlocWithAttn(
                    pre_channel,
                    pre_channel,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout,
                    with_attn=False,
                ),
            ]
        )

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = ind < 1
            use_attn = now_res in attn_res
            if use_attn:
                print("use attn: res {}".format(now_res))
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(
                    ResnetBlocWithAttn(
                        pre_channel + feat_channels.pop(),
                        channel_mult,
                        noise_level_emb_dim=noise_level_channel,
                        norm_groups=norm_groups,
                        dropout=dropout,
                        with_attn=use_attn,
                    )
                )
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(
            pre_channel, default(out_channel, in_channel), groups=norm_groups
        )

        self.res_blocks = res_blocks
        self.self_condition = self_condition
        # self.add_cond_layer_wise = add_cond_layer_wise

    def forward(self, x, time, cond=None, self_cond=None):
        # res = x

        # self-conditioning
        if self.self_condition:
            self_cond = default(self_cond, x)
            x = torch.cat([self_cond, x], dim=1)

        # if cond is not None:
        #     x = torch.cat([cond, x], dim=1)

        if self.fourier_features:
            x = torch.cat(
                [
                    x,
                    base2fourier_features(
                        x, self.fourier_min, self.fourier_max, self.fourier_step
                    ),
                ],
                dim=1,
            )

        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        feats = []
        # TODO: 在encoder中加入cross attn
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t, cond)
            else:
                x = layer(x)
            feats.append(x)
        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t, time)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t,time)
            else:
                x = layer(x)

        return self.final_conv(x)  # + res


# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = (
            torch.arange(count, dtype=noise_level.dtype,
                         device=noise_level.device)
            / count
        )
        encoding = noise_level.unsqueeze(1) * torch.exp(
            -math.log(1e4) * step.unsqueeze(0)
        )
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = (
                self.noise_func(noise_embed).view(
                    batch, -1, 1, 1).chunk(2, dim=1)
            )
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            # nn.BatchNorm2d(dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1),
        )

    def forward(self, x):
        return self.block(x)

class Block_2(nn.Module):
    def __init__(self, dim, dim_out, groups = 32,dropout=0):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.Sequential(nn.SiLU(),
                                 nn.Dropout(dropout) if dropout != 0 else nn.Identity(),)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout=0,groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block_2(dim, dim_out, groups = groups,dropout=dropout)
        self.block2 = Block_2(dim_out, dim_out, groups = groups,dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        noise_level_emb_dim=None,
        dropout=0,
        use_affine_level=False,
        norm_groups=32,
    ):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level
        )

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(
            dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        # self.norm = torch.nn.functional.normalize
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class CondInjection(nn.Module):
    def __init__(self, fea_dim, cond_dim, hidden_dim, groups=32) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cond_dim, hidden_dim * 4, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim * 4),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, 1, bias=True),
        )
        self.x_conv = nn.Conv2d(fea_dim, hidden_dim, 1, bias=True)
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x, cond):
        cond = self.body(cond)
        scale, shift = cond.chunk(2, dim=1)

        x = self.x_conv(x)

        x = x * (1 + scale) + shift
        return x

class Frequency_CondInjection(nn.Module):
    def __init__(self, fea_dim, cond_dim, hidden_dim, groups=32) -> None:
        super().__init__()
        self.body_real = nn.Sequential(
            nn.Conv2d(cond_dim, hidden_dim * 4, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim * 4),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, 1, bias=True),
        )
        self.body_imag = nn.Sequential(
            nn.Conv2d(cond_dim, hidden_dim * 4, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim * 4),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, 1, bias=True),
        )

        self.x_conv = nn.Conv2d(fea_dim, hidden_dim, 1, bias=True)
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x, cond):
        cond = torch.fft.rfft2(cond, dim=(2, 3), norm='ortho')
        cond_real = self.body_real(cond.real)
        cond_imag = self.body_imag(cond.imag)
        cond = torch.view_as_complex(torch.cat([cond_real,cond_imag],dim=1))
        scale,shift = cond.chunk(2,dim=1)

        x = self.x_conv(x)
        x = torch.fft.rfft2(x,dim=(2,3),norm='ortho')
        x = x * (1 + scale) + shift
        x = torch.fft.irfft2(x, s=(self.hidden_dim, self.hidden_dim),  norm='ortho')

        return x

class Frequency_Condition_Attention(nn.Module):
    def __init__(self, dim_x=32, dim_c=8, hidden_dim=64, h=64,  groups=32,dropout=0):
        super(Frequency_Condition_Attention, self).__init__()
        self.W_q_real = nn.Linear(hidden_dim,hidden_dim)
        self.W_q_imag = nn.Linear(hidden_dim, hidden_dim)
        self.W_k_real = nn.Linear(hidden_dim,hidden_dim)
        self.W_k_imag = nn.Linear(hidden_dim, hidden_dim)
        #self.W_v_real = nn.Linear(hidden_dim,hidden_dim)
        #self.W_v_imag = nn.Linear(hidden_dim, hidden_dim)


        self.q_conv = nn.Sequential(
            nn.Conv2d(dim_c, hidden_dim *2, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim , 1, bias=True),
        )
        self.k_conv = nn.Conv2d(dim_x, hidden_dim, 1, bias=True)
        self.v_conv = nn.Conv2d(dim_x, hidden_dim, 1, bias=True)
        # self.groupnorm_c = nn.GroupNorm(groups)
        # self.groupnorm_hw = nn.GroupNorm(groups)
        self.softmax = nn.Softmax(dim=-1)

        # self.time_mlp = nn.Sequential(
        #     PositionalEncoding(dim_x),
        #     nn.Linear(dim_x, time_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
        #     nn.Linear(time_dim, time_dim)
        # )
        self.a_c = nn.Parameter(torch.randn(hidden_dim, dtype=torch.float32) * 0.02)
        #self.a_hw = nn.Parameter(torch.randn(1, dtype=torch.float32) * 0.02)

        self.w_c = nn.Parameter(torch.randn([hidden_dim, hidden_dim], dtype=torch.float32) * 0.02) # C用
        #self.w_hw = nn.Parameter(torch.randn([h*w, h*w], dtype=torch.float32) * 0.02)  # hw用
#引入hw mlp等更低一点
        #self.t_block = ResBlock(hidden_dim, hidden_dim, time_emb_dim=time_dim,dropout=dropout)
        # self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim,bias=True), nn.SiLU(),
        #                          nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
        #                          nn.Linear(hidden_dim, hidden_dim),)
        self.hidden_dim = hidden_dim
    def forward(self, x, c):
        B, _, H, W = x.shape
        _, C_c,*_ = c.shape
        assert H == W, "height and width are not equal"

        q = c[:,:8,:,:].to(torch.float32)
        q = self.q_conv(q)
        q = torch.fft.rfft2(q, dim=(2, 3), norm='ortho')
        #weight_q = torch.view_as_complex(self.W_q)
        q_real = (q.real).reshape(B,self.hidden_dim,-1).permute(0,2,1)
        q_imag = (q.imag).reshape(B,self.hidden_dim,-1).permute(0,2,1)
        q = torch.cat([self.W_q_real(q_real).unsqueeze(-1),self.W_q_imag(q_imag).unsqueeze(-1)],dim=3)
        q = torch.view_as_complex(q).permute(0,2,1)


        k = x.to(torch.float32)
        k = self.k_conv(k)
        k = torch.fft.rfft2(k, dim=(2, 3), norm='ortho')
        k_real = (k.real).reshape(B, self.hidden_dim, -1).permute(0, 2, 1)
        k_imag = (k.imag).reshape(B, self.hidden_dim, -1).permute(0, 2, 1)
        k = torch.cat([self.W_k_real(k_real).unsqueeze(-1), self.W_k_imag(k_imag).unsqueeze(-1)], dim=-1)
        k = torch.view_as_complex(k).permute(0, 2, 1)

        v = x.to(torch.float32)
        v = self.v_conv(v)
        v = v.reshape(B, self.hidden_dim, -1)

        # [B,C,C]
        A_C = torch.bmm(q.reshape(B,self.hidden_dim,-1),k.reshape(B,self.hidden_dim,-1).permute(0,2,1))
        #A_C = self.a_c*torch.sin(self.w_c*A_C)
        A_C = torch.fft.irfft2(A_C, s=(self.hidden_dim, self.hidden_dim),  norm='ortho') #这里是否换成实部虚部分别attn比较好？
        A_C = F.softmax(A_C,dim=-1)

        v_c = torch.bmm(A_C,v) #[B,C,hw]

        # [B,hw,hw]
        '''
        这里hw的处理是否妥当有待商酌
        '''
        # A_hw = torch.bmm(q.reshape(B, self.hidden_dim, -1).permute(0, 2, 1), k.reshape(B, self.hidden_dim, -1))
        # #A_hw = torch.sin(A_hw)
        # _,_,si = A_hw.shape
        # A_hw = torch.fft.irfft2(A_hw, s=(H*W, H*W), norm='ortho')
        # A_hw = F.softmax(A_hw, dim=-1)
        # v_hw = torch.bmm(A_hw, v.permute(0, 2, 1)).permute(0, 2, 1)

        v = v_c# + v_hw
        v = v.reshape(B, self.hidden_dim, H, W)

        #t = self.time_mlp(time)
        #v = self.t_block(v, t)
        #v = self.mlp(v.reshape(B, self.hidden_dim, -1).permute(0,2,1)).permute(0,2,1).reshape(B, self.hidden_dim, H, W)

        return v

class ResnetBlocWithAttn(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        h_x = 64,
        cond_dim=None,
        noise_level_emb_dim=None,
        norm_groups=32,
        dropout=0,
        with_attn=False,
    ):
        super().__init__()
        self.with_attn = with_attn
        self.with_cond = exists(cond_dim)
        self.res_block = ResnetBlock(
            dim_out if exists(cond_dim) else dim,
            dim_out,
            noise_level_emb_dim,
            norm_groups=norm_groups,
            dropout=dropout,
        )
        if with_attn:
            self.attn = SelfAttention(
                dim_out, norm_groups=norm_groups, n_head=8)
        if self.with_cond:
            self.cond_inj = CondInjection(
                dim, cond_dim, hidden_dim=dim_out, groups=norm_groups
            )
            self.fre_inj = Frequency_CondInjection(
                dim, cond_dim, hidden_dim=dim_out, groups=norm_groups
            )
            self.fre_attn = Frequency_Condition_Attention(
                dim,cond_dim-1, hidden_dim=dim_out, groups=norm_groups,dropout=dropout,
            )

            self.w_f = nn.Parameter(torch.randn(1,dtype=torch.float32) * 0.02)
            # self.w_f = nn.Conv2d(dim_out,dim_out,1,1,0,bias=True)
            # self.w_c = nn.Conv2d(dim_out, dim_out, 1, 1, 0,bias=True)
            # self.fuse = nn.Sequential(nn.Conv2d(dim_out*2,dim_out,3,1,1,bias=False),
            #                           nn.GroupNorm(norm_groups,dim_out),
            #                           nn.SiLU(),
            #                           nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            #                           nn.Conv2d(dim_out,dim_out,1,1,0,bias=True))

    def forward(self, x, time_emb, cond=None):
        if self.with_cond:
            x_c = self.cond_inj(
                x, F.interpolate(cond, size=x.shape[-2:], mode="bilinear")
            )
            x_f = self.fre_inj(x,F.interpolate(cond, size=x.shape[-2:], mode="bilinear"))
            x =  (1-self.w_f)*x_c + self.w_f*x_f

        x = self.res_block(x, time_emb)
        if self.with_attn:
            x = self.attn(x)
        return x


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


if __name__ == "__main__":
    net = UNetSR3(
        in_channel=8,
        out_channel=8,
        cond_channel=9,
        image_size=64,
        self_condition=False,
        inner_channel=64,
        norm_groups=1,
    )
    x = torch.randn(1, 8, 64, 64)
    cond = torch.randn(1, 9, 64, 64)
    t = torch.LongTensor([1])
    y = net(x, t, cond)
    print(y.shape)
