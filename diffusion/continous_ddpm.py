import math
from functools import partial, wraps
from typing import Sequence

import torch
from torch import sqrt
from torch import nn, einsum
import torch.nn.functional as F
from torch.special import expm1

from tqdm import tqdm
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange


def exists(val):
    return val is not None


def is_lambda(f):
    return callable(f) and f.__name__ == "<lambda>"


def default(val, d):
    if exists(val):
        return val
    return d() if is_lambda(d) else d


# normalization functions


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# diffusion helpers


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# logsnr schedules and shifting / interpolating decorators
# only cosine for now


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def logsnr_schedule_cosine(t, logsnr_min=-15, logsnr_max=15):
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * log(torch.tan(t_min + t * (t_max - t_min)))


def logsnr_schedule_shifted(fn, image_d, noise_d):
    shift = 2 * math.log(noise_d / image_d)

    @wraps(fn)
    def inner(*args, **kwargs):
        nonlocal shift
        return fn(*args, **kwargs) + shift
    return inner


def logsnr_schedule_interpolated(fn, image_d, noise_d_low, noise_d_high):
    logsnr_low_fn = logsnr_schedule_shifted(fn, image_d, noise_d_low)
    logsnr_high_fn = logsnr_schedule_shifted(fn, image_d, noise_d_high)

    @wraps(fn)
    def inner(t, *args, **kwargs):
        nonlocal logsnr_low_fn
        nonlocal logsnr_high_fn
        return t * logsnr_low_fn(t, *args, **kwargs) + (1 - t) * logsnr_high_fn(t, *args, **kwargs)

    return inner

# main gaussian diffusion class

# simple diffusion model
# which will shift on more noise schedule on
# higher resolution image


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        image_size,
        channels=3,
        pred_objective='v',
        noise_schedule=logsnr_schedule_cosine,
        noise_d=None,
        noise_d_low=None,
        noise_d_high=None,
        num_sample_steps=500,
        clip_sample_denoised=True,
    ):
        super().__init__()
        assert pred_objective in {
            'v', 'eps', 'x_start'}, 'whether to predict v-space (progressive distillation paper), noise or x_start'

        self.model = model

        # image dimensions

        self.channels = channels
        self.image_size = image_size

        # training objective

        self.pred_objective = pred_objective

        # noise schedule

        assert not all([*map(exists, (noise_d, noise_d_low, noise_d_high))]
                       ), 'you must either set noise_d for shifted schedule, or noise_d_low and noise_d_high for shifted and interpolated schedule'

        # determine shifting or interpolated schedules

        self.log_snr = noise_schedule

        if exists(noise_d):
            self.log_snr = logsnr_schedule_shifted(
                self.log_snr, image_size, noise_d)

        if exists(noise_d_low) or exists(noise_d_high):
            assert exists(noise_d_low) and exists(
                noise_d_high), 'both noise_d_low and noise_d_high must be set'

            self.log_snr = logsnr_schedule_interpolated(
                self.log_snr, image_size, noise_d_low, noise_d_high)

        # sampling

        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised
        
    def reset_logsnr(self, noise_d=None, noise_d_low=None, noise_d_high=None):
        if exists(noise_d):
            self.log_snr = logsnr_schedule_shifted(
                self.log_snr, self.image_size, noise_d)

        if exists(noise_d_low) or exists(noise_d_high):
            assert exists(noise_d_low) and exists(
                noise_d_high), 'both noise_d_low and noise_d_high must be set'

            self.log_snr = logsnr_schedule_interpolated(
                self.log_snr, self.image_size, noise_d_low, noise_d_high)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def p_mean_variance(self, x, time, time_next, cond=None):

        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (
            -log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(
            sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b=x.shape[0])
        pred = self.model(x, batch_log_snr, cond)

        if self.pred_objective == 'v':
            x_start = alpha * x - sigma * pred

        elif self.pred_objective == 'eps':
            x_start = (x - sigma * pred) / alpha
        
        elif self.pred_objective == 'x_start':
            x_start = pred

        x_start.clamp_(-1., 1.)

        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance

    # sampling related functions

    @torch.no_grad()
    def p_sample(self, x, time, time_next, cond=None):
        batch, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance(
            x=x, time=time, time_next=time_next, cond=cond)

        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond=None):
        batch = shape[0]

        img = torch.randn(shape, device=self.device)
        steps = torch.linspace(
            1., 0., self.num_sample_steps + 1, device=self.device)

        for i in tqdm(range(self.num_sample_steps), desc='sampling loop time step', total=self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next, cond)

        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, shape: tuple[int, int]=None, cond=None):
        if exists(cond):
            shape = cond.shape[-2:]
        else:
            if not exists(shape):
                shape = (self.image_size, self.image_size)
        return self.p_sample_loop((batch_size, self.channels, *shape), cond)

    # training related functions - noise prediction

    def q_sample(self, x_start, times, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        log_snr = self.log_snr(times)

        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt(
            (-log_snr_padded).sigmoid())
        x_noised = x_start * alpha + noise * sigma

        return x_noised, log_snr

    def p_losses(self, x_start, times, noise=None, cond=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x, log_snr = self.q_sample(x_start=x_start, times=times, noise=noise)
        model_out = self.model(x, log_snr, cond)

        if self.pred_objective == 'v':
            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha, sigma = padded_log_snr.sigmoid().sqrt(), (-padded_log_snr).sigmoid().sqrt()
            target = alpha * noise - sigma * x_start
            pred_x_start = alpha * x - sigma * model_out
        
        elif self.pred_objective == 'x_start':
            target = x_start
            pred_x_start = model_out

        elif self.pred_objective == 'eps':
            target = noise
            pred_x_start = (x - model_out * sigma) / alpha

        return F.l1_loss(model_out, target), unnormalize_to_zero_to_one(pred_x_start)

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        img = normalize_to_neg_one_to_one(img)
        times = torch.zeros(
            (img.shape[0],), device=self.device).float().uniform_(0, 1)

        return self.p_losses(img, times, *args, **kwargs)


if __name__ == '__main__':
    from models.uvit import UViT

    uvit = UViT(32, out_dim=8, channels=9, patch_size=1, attn_dim_head=8)
    x = (torch.randn(1, 8, 64, 64) + 1) / 2
    cond = (torch.randn(1, 1, 64, 64) + 1) / 2
    diffusion = GaussianDiffusion(uvit, image_size=64, channels=8)
    
    loss = diffusion(x, cond=cond)
    print(loss)
    
    sample_y = diffusion.sample(1, cond=cond)
    print(sample_y)
    