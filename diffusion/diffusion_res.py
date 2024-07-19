import math
from pyexpat import model
import random
import torch
from torch import Tensor, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm

from utils.loss_utils import HybridL1SSIM


#### this is a test


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64
    )
    return betas

def make_sqrt_etas_schedule(min_noise_level=0.01, n_timestep=15, eta_T=0.99, p=0.3,k=1.0):
    power = p
    etas_start = min(min_noise_level / k, min_noise_level, math.sqrt(0.001))
    increaser = math.exp(1 / (n_timestep - 1) * math.log(eta_T / etas_start))
    base = np.ones([n_timestep, ]) * increaser
    power_timestep = np.linspace(0, 1, n_timestep, endpoint=True) ** power
    power_timestep *= (n_timestep - 1)
    sqrt_etas = np.power(base, power_timestep) * etas_start

    #〖设置η_1 为min⁡{(0.04/κ)〗^2,0.001}
    # t = np.linspace(1, n_timestep, n_timestep, dtype=np.float64)
    # eta_1 = min((min_noise_level/k)**2,0.001)
    # sqrt_eta_1 = math.sqrt(eta_1)
    # #betas = (t-1/n_timestep-1)**p*(n_timestep-1) #论文里的算法
    # betas = np.linspace(0, 1, n_timestep, endpoint=True) ** p * (n_timestep - 1) #实际上的算法
    # b_0 = math.exp(math.log(eta_T/sqrt_eta_1)/(n_timestep-1)) #论文是eta_T/eta_1,但代码实际是这样
    # sqrt_etas = math.sqrt(eta_1)*np.power(b_0,betas)
    return sqrt_etas

def make_beta_schedule(
        schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "quad":
        betas = (
                np.linspace(
                    linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float64
                )
                ** 2
        )
    elif schedule == "linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "warmup10":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
    elif schedule == "warmup50":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
    elif schedule == "const":
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]  # 1 -> 0
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)  # 0 -> 1
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    def repeat_noise():
        return torch.randn((1, *shape[1:]), device=device).repeat(
            shape[0], *((1,) * (len(shape) - 1))
        )

    def noise():
        return torch.randn(shape, device=device)

    return repeat_noise() if repeat else noise()


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def meanflat(x):
    return x.mean(dim=tuple(range(1, len(x.shape))))


def approx_standard_normal_cdf(x):
    return 0.5 * (
            1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3)))
    )


def log(t, eps=1e-15):
    return torch.log(t.clamp(min=eps))


def discretized_gaussian_log_likelihood(x, *, means, log_scales, thres=0.999):
    assert x.shape == means.shape == log_scales.shape

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = log(cdf_plus)
    log_one_minus_cdf_min = log(1.0 - cdf_min)
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
        x < -thres,
        log_cdf_plus,
        torch.where(x > thres, log_one_minus_cdf_min, log(cdf_delta)),
    )

    return log_probs


NAT = 1.0 / math.log(2)


class ShiftDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            first_stage_model=None,
            scale_factor=1,
            channels=3,
            sf=4,
            loss_type="l2",
            conditional=True,
            schedule_opt=None,
            device="cuda:0",
            clamp_range=(-1.0, 1.0),
            clamp_type="abs",
            pred_mode="residual",
            p2_loss_weight_gamma=0.0,
            # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
            p2_loss_weight_k=1,
    ):
        super().__init__()
        self.channels = channels
        self.first_stage_model = first_stage_model
        self.model = denoise_fn
        self.conditional = conditional
        self.loss_type = loss_type
        self.device = device
        self.clamp_range = clamp_range
        self.clamp_type = clamp_type
        self.kappa = 1.0
        self.sqrt_kappa = math.sqrt(self.kappa)
        self.sf = sf
        self.scale_factor = scale_factor

        assert clamp_type in ["abs", "dynamic"]
        assert pred_mode in ["noise", "x_start", "pred_v","residual_start"]
        assert loss_type in ["l1", "l2", "l1ssim"]
        # p2 loss weight
        self.p2_loss_weight_gamma = p2_loss_weight_gamma
        self.p2_loss_weight_k = p2_loss_weight_k

        if schedule_opt is not None:
            # pass
            self.set_new_noise_schedule(schedule_opt, device)
        self.set_loss(device)

        self.pred_mode = pred_mode
        self.thresholding_max_val = 1.0
        self.dynamic_thresholding_ratio = 0.8

    def set_loss(self, device):
        if self.loss_type == "l1":
            self.loss_func = nn.L1Loss().to(device)
        elif self.loss_type == "l2":
            self.loss_func = nn.MSELoss().to(device)
        elif self.loss_type == "l1ssim":
            self.loss_func = HybridL1SSIM(channel=self.channels).to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt=None, device="cpu", *, sqrt_etas=None):
        """set new schedule, include but not limited betas, alphas,
        betas_cumprod, alphas_cumprod and register them into a buffer.

        Args:
            schedule_opt (dict, optional): a dict for schedule. Defaults to None.
            device (str, optional): device. Defaults to 'cpu'.
            betas (Union[List, Set], optional): new betas for ddim sampling. Defaults to None.
        """

        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        if schedule_opt is not None:
            sqrt_etas = make_sqrt_etas_schedule(
                n_timestep=schedule_opt["n_timestep"],k=self.kappa
            )
        etas = sqrt_etas**2
        etas = (
            etas.detach().cpu().numpy() if isinstance(etas, torch.Tensor) else etas
        )
        #tem_etas = np.roll(etas,1)
        tem_etas = np.append(0.0, etas[:-1])
        tem_etas[0] = 0
        alphas = etas - tem_etas

        (timesteps,) = etas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer("sqrt_etas",to_torch(sqrt_etas))
        self.register_buffer("etas", to_torch(etas))
        self.register_buffer("alphas", to_torch(alphas))
        cumsum_alphas = torch.cumsum(to_torch(alphas),dim=0)
        self.register_buffer("cumsum_alphas",cumsum_alphas)
        etas_prev = np.append(0.0, etas[:-1])
        self.register_buffer("etas_prev", to_torch(etas_prev))
        posterior_variance = self.kappa ** 2 * etas_prev / etas * alphas
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        posterior_variance_clipped = np.append(
            posterior_variance[1], posterior_variance[1:]
        )
        posterior_mean_coef1 = etas_prev / etas
        posterior_mean_coef2 = alphas / etas
        self.register_buffer("posterior_mean_coef1", to_torch(posterior_mean_coef1))
        self.register_buffer("posterior_mean_coef2", to_torch(posterior_mean_coef2))

        self.register_buffer("posterior_variance_clipped", to_torch(posterior_variance_clipped))
        posterior_log_variance_clipped = np.log(posterior_variance_clipped)
        self.register_buffer("posterior_log_variance_clipped", to_torch(posterior_log_variance_clipped))
        weight_loss_mse = 0.5 / posterior_variance_clipped * (alphas / etas) ** 2
        self.register_buffer("weight_loss_mse",to_torch(weight_loss_mse))
        # # calculate p2 reweighting
        self.register_buffer(
            "p2_loss_weight",
            to_torch(
                (self.p2_loss_weight_k + etas / (1 - etas))
                ** -self.p2_loss_weight_gamma
            ),
        )

    def decode_first_stage(self, z_sample, first_stage_model=None):
        ori_dtype = z_sample.dtype
        if first_stage_model is None:
            return z_sample
        else:
            with torch.no_grad():
                z_sample = 1 / self.scale_factor * z_sample
                z_sample = z_sample.type(next(first_stage_model.parameters()).dtype)
                out = first_stage_model.decode(z_sample)
                return out.type(ori_dtype)

    def encode_first_stage(self, y, first_stage_model, up_sample=False):
        ori_dtype = y.dtype
        if up_sample:
            y = F.interpolate(y, scale_factor=self.sf, mode='bicubic')
        if first_stage_model is None:
            return y
        else:
            with torch.no_grad():
                y = y.type(dtype=next(first_stage_model.parameters()).dtype)
                z_y = first_stage_model.encode(y)
                out = z_y * self.scale_factor
                return out.type(ori_dtype)

    def q_mean_variance(self, x_start, t):
        mean = (torch.ones(x_start.shape,device="cuda")-extract(self.cumsum_alphas, t, x_start.shape)) * x_start
        #variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        #log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean#, variance, log_variance

    def predict_noise_from_start(self, x_t, t, x_0_pred):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_0_pred
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_start_from_xprev(self, x_t, t, xprev):
        return (
                extract(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
                - extract(
            self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
        )
                * x_t
        )

    def predict_start_from_noise(self, x_t, t, e_0,noise):

        return ( x_t-
                extract(self.etas, t, x_t.shape) * e_0
                - self.kappa**2*extract(self.etas, t, x_t.shape) * noise
        )

    def predict_v_from_start(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
                - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_xstart_from_residual(self, y, residual):
        assert y.shape == residual.shape
        return (y - residual)

    def q_posterior(self, x_start, x_t, t):
        #η_{t-1}/η_{t}*x_{t}+α_{t}/η_{t}*x_{0}     //+k^{2}*η_{t-1}/η_{t}*α_{t}*ε
        # posterior_mean = (
        #         extract(self.etas, t-1, x_t.shape)/extract(self.etas, t, x_t.shape)* x_t
        #         + extract(self.alphas, t, x_t.shape)/extract(self.etas, t, x_t.shape) * x_start
        #         #+ self.kappa**2*extract(self.etas, t-1, x_t.shape)/extract(self.etas, t, x_t.shape)*extract(self.alphas, t, x_t.shape)*
        # )
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_t
                + extract(self.posterior_mean_coef2, t, x_t.shape) * x_start
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # imagen dynamic thresholding
    def dynamic_thresholding_fn(self, x0, t):
        """
        The dynamic thresholding method.
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(
            torch.maximum(
                s, self.thresholding_max_val * torch.ones_like(s).to(s.device)
            ),
            dims,
        )
        x0 = (
                torch.clamp(x0, torch.zeros_like(s), s) / s
        )  # it should be clamp(x0, -s, s)/s if input data ranging in [-1, 1]
        return x0

    def p_mean(self,
               e_t,
               t,
               y,
               cond,
               *,
               model_out=None,):
        valid_data = {"lq": e_t+y, "t":t, "cond": cond}

        self.model.feed_data(valid_data)
        model_out = self.model.optimize_parameters(current_iter=0, mode="valid")
        #e_t = F.interpolate(e_t, size=model_out.shape[-2:], mode='bilinear')#放大2倍
        # if clip_denoised:
        #     # x_recon = (
        #     #         x_recon + condition_x[:, : self.channels]
        #     # )  # add lms to origial image(test code)
        #     if self.clamp_type == "abs":  # absolute
        #         e_recon.clamp_(*self.clamp_range)
        #     else: 阿德瓦三大三大范围无法巍峨 # dynamic
        #         e_recon = self.dynamic_thresholding_fn(e_recon, t)
        #     #x_recon = x_recon - condition_x[:, : self.channels]这一块影响很大
        model_mean,*_ = self.q_posterior(
            x_start=model_out, x_t=e_t, t=t
        )#e_t-1
        posterior_variance = extract(self.posterior_variance, t, e_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, e_t.shape
        )
        return model_mean,posterior_variance,posterior_log_variance_clipped,model_out

    @torch.no_grad()
    def p_sample(
            self,
            e_t,
            cond,
            y,
            t,
            repeat_noise=False,
     ):
        b, *_, device = *e_t.shape, e_t.device
        model_mean, variance,log_variance,pred_x_start = self.p_mean(
            e_t=e_t,
            y=y,
            cond=cond,
            t=t,
          )

        noise = noise_like(model_mean.shape, device, repeat_noise)
        #ee = extract(self.etas, t-1, x.shape)/extract(self.etas, t, x.shape)*extract(self.alphas, t, x.shape)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(model_mean.shape) - 1)))
        )  # no noise when t == 0
        #return model_mean + self.kappa**2*ee* noise
        return model_mean + nonzero_mask * torch.exp(0.5 * log_variance)*noise


    @torch.no_grad()
    def p_sample_loop(self, x_in, cond):
        device = self.etas.device

        if not self.conditional:
            shape = x_in
            # b = shape[0]
            # #noise = th.randn_like(z_y)
            # noise = torch.randn(shape, device=device)
            # ret_img = noise
            #
            # #z_sample = self.prior_sample(x_in, noise)
            # z_sample = torch.randn(shape, device=device)
            # length = list(range(self.num_timesteps))[::-1]
            # # for i in tqdm(
            # #         reversed(range(0, self.num_timesteps)),
            # #         desc="ddpm sampling loop time step",
            # #         total=self.num_timesteps,
            # # ):
            # for i in tqdm(length):
            #     self_cond = x_start+x_in if self.self_condition else None
            #     img = self.p_sample(
            #         z_sample,
            #         torch.full((b,), i, device=device, dtype=torch.long),
            #         condition_x=cond,
            #         self_cond=self_cond,
            #         clip_denoised=clip_noise,
            #         get_interm_fm=get_interm_fm,
            #     )
            #     z_sample = img
            #     if i % sample_inter == 0 and continous:
            #         ret_img = torch.cat([ret_img, img], dim=0)
            #
            #     x_start = img
            # return img
        else:
            noise = torch.randn_like(x_in, device=device)
            #z_sample = self.prior_sample(x_in, noise) #x_T
            z_sample = torch.zeros_like(x_in)
            b,_,h,w = x_in.shape
            length = list(range(self.num_timesteps))[::-1]
            #y = F.interpolate(x_in, size=(int(h // 2), int(w // 2)),mode='bilinear')
            #pan = F.interpolate(pan, size=(int(h // 2), int(w // 2)), mode='bilinear')
            for i in tqdm(length):
                #z_sample = F.interpolate(z_sample, size=(int(h // 2), int(w // 2)), mode='bilinear')
                res = self.p_sample(
                    e_t=z_sample,
                    y=x_in,
                    t=torch.full((b,), i, device=device, dtype=torch.long),
                    cond=cond,
                )#x_t+res_t->x_0
                z_sample = res
        return res

    def prior_sample(self, y, noise=None):
        """
        Generate samples from the prior distribution, i.e., q(x_T|x_0) ~= N(x_T|y, ~)

        :param y: the [N x C x ...] tensor of degraded inputs.
        :param noise: the [N x C x ...] tensor of degraded inputs.
        """
        if noise is None:
            noise = torch.randn_like(y)

        t = torch.tensor([self.num_timesteps - 1, ] * y.shape[0], device=y.device).long()

        return y + extract(self.kappa * self.sqrt_etas, t, y.shape) * noise

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(
                reversed(range(0, t)), desc="interpolation sample time step", total=t
        ):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long)
            )

        return img

    @staticmethod
    def space_timesteps(num_timesteps, section_counts):
        """
        Create a list of timesteps to use from an original diffusion process,
        given the number of timesteps we want to take from equally-sized portions
        of the original process.
        For example, if there's 300 timesteps and the section counts are [10,15,20]
        then the first 100 timesteps are strided to be 10 timesteps, the second 100
        are strided to be 15 timesteps, and the final 100 are strided to be 20.
        If the stride is a string starting with "ddim", then the fixed striding
        from the DDIM paper is used, and only one section is allowed.
        :param num_timesteps: the number of diffusion steps in the original
                            process to divide up.
        :param section_counts: either a list of numbers, or a string containing
                            comma-separated numbers, indicating the step count
                            per section. As a special case, use "ddimN" where N
                            is a number of steps to use the striding from the
                            DDIM paper.
        :return: a set of diffusion steps from the original process to use.
        """

        if isinstance(section_counts, str):
            if section_counts.startswith("ddim"):
                desired_count = int(section_counts[len("ddim"):])
                for i in range(1, num_timesteps):
                    if len(range(0, num_timesteps, i)) == desired_count:
                        return set(range(0, num_timesteps, i))
                raise ValueError(
                    f"cannot create exactly {num_timesteps} steps with an integer stride"
                )
            section_counts = [int(x) for x in section_counts.split(",")]
        size_per = num_timesteps // len(section_counts)
        extra = num_timesteps % len(section_counts)
        start_idx = 0
        all_steps = []
        for i, section_count in enumerate(section_counts):
            size = size_per + (1 if i < extra else 0)
            if size < section_count:
                raise ValueError(
                    f"cannot divide section of {size} steps into {section_count}"
                )
            if section_count <= 1:
                frac_stride = 1
            else:
                frac_stride = (size - 1) / (section_count - 1)
            cur_idx = 0.0
            taken_steps = []
            for _ in range(section_count):
                taken_steps.append(start_idx + round(cur_idx))
                cur_idx += frac_stride
            all_steps += taken_steps
            start_idx += size
        return set(all_steps)

    def space_new_betas(self, use_timesteps):
        last_alpha_cumprod = 1.0
        new_betas = []
        timestep_map = []
        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            if i in use_timesteps:
                new_betas.append((1 - alpha_cumprod / last_alpha_cumprod).item())
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)
        self.set_new_noise_schedule(etas=np.array(new_etas), device=self.betas.device)


    def q_sample(self, e_0, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(e_0))

        # residual sample
        # (x_{0}+η_{t}*e_{0})+ κ^{2}*η_{t}*ε
        # also call VP schedule

        # SNR: self.sqrt_alphas_cumprod / self.sqrt_one_minus_alphas_cumprod
        # SNR weighting: max(self.sqrt_alphas_cumprod **2 / self.sqrt_one_minus_alphas_cumprod**2, 1)
        #               or (1 + self.sqrt_alphas_cumprod **2 / self.sqrt_one_minus_alphas_cumprod**2)
        # return (
        #         _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start) + x_start
        #         + _extract_into_tensor(self.sqrt_etas * self.kappa, t, x_start.shape) * noise
        # )
        #return extract(self.etas, t, x_start.shape)*e_0 + x_start + self.kappa*extract(self.sqrt_etas, t, x_start.shape) * noise
        return e_0 - extract(self.etas, t, e_0.shape) * e_0 + extract(self.kappa*self.sqrt_etas, t,e_0.shape) * noise


    def p_losses(self,opt,x_start,cond,y=None,noise=None,current_iter=0):
        '''
        :param e_T:
        :param e_0:
        :param noise:
        :param cond:
        :return:
        '''
        [b, c, h, w] = x_start.shape
        t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device).long()
        e_0 = x_start-y
        #e_0 = F.interpolate(e_0_ori, size=(int(h // 2), int(w // 2)), mode='bilinear')
        #cond = F.interpolate(cond, size=(int(h // 2), int(w // 2)), mode='bilinear')
        #yy = F.interpolate(y,size=(int(h//2),int(w//2)),mode='bilinear')
        noise = default(noise, lambda: torch.randn_like(e_0))
        x_noisy = self.q_sample(e_0=e_0,t=t, noise=noise)  # e_t

        # self-conditioning
        train_data = {"lq": x_noisy+y, "t":t, "cond": cond, "gt": e_0}#输入x_t,输出e_0
        self.model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
        self.model.feed_data(train_data)
        loss = self.model.optimize_parameters(current_iter=current_iter)
        return loss
        #return model_predict

    def forward(self,opt=None,x=None, y=None, cond=None,mode="train", *args, **kwargs):
        #e_0 = self.preprocess_encode_x(e_0)
        if mode == "train":
            return self.p_losses(opt=opt,x_start=x,cond=cond,y=y, *args, **kwargs)
        elif mode == "ddpm_sample":
            with torch.no_grad():
                return self.p_sample_loop(x_in=y,cond=cond, *args, **kwargs)
        elif mode == "ddim_sample":
            with torch.no_grad():
                return self.ddim_sample_loop(e_0, *args, **kwargs)
        else:
            raise NotImplementedError("mode should be train or sample")


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.
    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]


if __name__ == "__main__":
    from models.sr3 import UNetSR3 as UNet

    denoise_fn = UNet(
        in_channel=8,
        out_channel=8,
        cond_channel=9,
        image_size=64,
        self_condition=True,
        pred_var=False,
    ).cuda()
    sr = torch.randn(2, 9, 64, 64).cuda()
    hr = torch.randn(2, 8, 64, 64).cuda()
    # x_in = {"SR": sr, "HR": hr}
    schedule = dict(
        schedule="linear",
        n_timestep=200,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    )
    diffusion = GaussianDiffusion(
        denoise_fn,
        64,
        loss_type="l2",
        schedule_opt=schedule,
        channels=8,
        conditional=True,
        p2_loss_weight_k=1.0,
        p2_loss_weight_gamma=1.0,
    ).cuda()
    print(diffusion.p_losses(hr, cond=sr))
    print(diffusion.p_sample_loop(sr).shape)
