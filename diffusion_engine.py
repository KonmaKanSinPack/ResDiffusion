from ast import Not
from functools import partial
import time
from copy import deepcopy

import einops
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torchvision as tv
from scipy.io import savemat
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset.pan_dataset import PanDataset
from dataset.hisr import HISRDataSets
from diffusion.diffusion_ddpm_google import make_beta_schedule
from models.diffusion_res import ShiftDiffusion, make_sqrt_etas_schedule
from utils.logger import TensorboardLogger
from utils.lr_scheduler import get_lr_from_optimizer, StepsAll
from utils.metric import AnalysisPanAcc
from utils.misc import compute_iters, exist, grad_clip, model_load, path_legal_checker
from utils.optim_utils import EmaUpdater
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def show_distribution(hr):
    plt.figure()
    gt = hr[:2].detach().cpu().flatten().numpy()
    sns.displot(data=gt)
    plt.show()


def norm(x):
    # x range is [0, 1]
    return x * 2 - 1


def unorm(x):
    # x range is [-1, 1]
    return (x + 1) / 2


def clamp_fn(g_sr):
    def inner(x):
        x = x + g_sr
        x = x.clamp(0, 1.0)
        return x - g_sr

    return inner

def save_fig(l,r,rgb_channel,path):
    hr = tv.utils.make_grid(l, nrow=4, padding=0).cpu()
    x = tv.utils.make_grid(r, nrow=4, padding=0).detach().cpu()
    hr = hr.clip(0,1)
    x = x.clip(0, 1)

    s = torch.cat([hr, x], dim=-1)  # [b, c, h, 2*w]
    fig, ax = plt.subplots(
        figsize=(s.shape[-1] // 100, s.shape[-2] // 100)
    )
    ax.imshow(
        s.permute(1, 2, 0)
        .detach()
        .numpy()[..., rgb_channel]
    )
    ax.set_axis_off()

    # plt.show()
    plt.tight_layout(pad=0)
    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0,
    )

def engine_google(
    # dataset
    train_dataset_path,
    valid_dataset_path,
    dataset_name=None,
    # image settings
    image_n_channel=8,
    image_size=64,
    # diffusion settings
    schedule_type="cosine",
    n_steps=3_000,
    max_iterations=400_000,
    # device setting
    device="cuda:0",
    # optimizer settings
    batch_size=128,
    lr_d=1e-4,
    show_recon=False,
    # pretrain settings
    pretrain_weight=None,
    pretrain_iterations=None,
    *,
    # just for debugging
    constrain_channel=None,
):
    print(show_recon)
    """train and valid function

    Args:
        train_dataset_path (str): _description_
        valid_dataset_path (str): _description_
        batch_size (int, optional): _description_. Defaults to 240.
        n_steps (int, optional): _description_. Defaults to 1500.
        epochs (int, optional): _description_. Defaults to None.
        device (str, optional): _description_. Defaults to 'cuda:0'.
        max_iterations (int, optional): _description_. Defaults to 500_000.
        lr (float, optional): _description_. Defaults to 1e-4.
        pretrain_weight (str, optional): _description_. Defaults to None.
        recon_loss (bool, optional): _description_. Defaults to False.
        show_recon (bool, optional): _description_. Defaults to False.
        constrain_channel (int, optional): _description_. Defaults to None.
    """
    # from diffusion.continous_ddpm import GaussianDiffusion
    from diffusion.diffusion_ddpm_google import GaussianDiffusion
    # from models.sr3 import UNetSR3 as Unet
    from models.sr3_dwt import UNetSR3 as Unet
    from models.buffer_layer import ResBlock
    # from models.med_fft_unt import Unet

    # init logger
    stf_time = time.strftime("%m-%d %H:%M", time.localtime())
    comment = "ddpm_wv3"
    logger = TensorboardLogger(file_logger_name="{}-{}".format(stf_time, comment))

    dataset_name = (
        train_dataset_path.strip(".h5").split("_")[-1]
        if not exist(dataset_name)
        else dataset_name
    )
    logger.print(f"dataset name: {dataset_name}")
    division_dict = {"wv3": 2047.0, "gf2": 1023.0, "qb": 2047.0, "cave": 1.0}
    logger.print(f"dataset norm division: {division_dict[dataset_name]}")
    rgb_channel = {
        "wv3": [4, 2, 0],
        "gf2": [0, 1, 2],
        "qb": [0, 1, 2],
        "cave": [2, 4, 6],
    }
    logger.print(f"rgb channel: {rgb_channel[dataset_name]}")
    add_n_channel = 1

    # initialize models
    torch.cuda.set_device(device)

    res_model = ResBlock(image_n_channel,image_n_channel,0.2,norm_groups=8).to(device)

    denoise_fn = Unet(
        in_channel=image_n_channel,
        out_channel=image_n_channel,
        lms_channel=image_n_channel,
        pan_channel=add_n_channel,
        inner_channel=32,
        norm_groups=1,
        channel_mults=(1, 2, 2, 4),  # (64, 32, 16, 8)
        attn_res=(8,),
        dropout=0.2,
        image_size=64,
           self_condition=True,
    ).to(device)
    if pretrain_weight is not None:
        if isinstance(pretrain_weight, (list, tuple)):
            model_load(pretrain_weight[0], denoise_fn, strict=True, device=device)
            # model_load(pretrain_weight[1], guidance_net, strict=True, device=device)
        else:
            model_load(pretrain_weight, denoise_fn, strict=True, device=device)

    # get dataset
    d_train = h5py.File(train_dataset_path)
    d_valid = h5py.File(valid_dataset_path)
    if dataset_name in ["wv3", "gf2", "qb"]:
        DatasetUsed = partial(
            PanDataset,
            full_res=False,
            norm_range=False,
            constrain_channel=constrain_channel,
            division=division_dict[dataset_name],
            aug_prob=0,
            wavelets=True,
        )
    elif dataset_name == "cave":
        DatasetUsed = partial(HISRDataSets, normalize=False, aug_prob=0)
    else:
        raise NotImplementedError("dataset {} not supported".format(dataset_name))

    ds_train = DatasetUsed(
        d_train,
        # full_res=False,
        # norm_range=False,
        # constrain_channel=constrain_channel,
        # division=division_dict[dataset_name],
        # aug_prob=0,
    )
    ds_valid = DatasetUsed(
        d_valid,
        # full_res=False,
        # norm_range=False,
        # constrain_channel=constrain_channel,
        # division=division_dict[dataset_name],
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )

    # diffusion
    # diffusion = GaussianDiffusion(
    #     denoise_fn,
    #     image_size=image_size,
    #     channels=image_n_channel,
    #     # num_sample_steps=n_steps,
    #     pred_mode="x_start",
    #     loss_type="l1",
    #     device=device,
    #     clamp_range=(0, 1),
    # )
    # diffusion.set_new_noise_schedule(
    #     betas=make_beta_schedule(schedule="cosine", n_timestep=n_steps, cosine_s=8e-3)
    # )

    # diffusion = ShiftDiffusion(
    #     denoise_fn,
    #     #self.autoencoder,
    #     channels=image_n_channel,
    #     # num_sample_steps=n_steps,
    #     pred_mode="x_start",
    #     loss_type="l2",
    #     device=device,
    #     clamp_range=(0, 1),
    # ).cuda()
    diffusion = ShiftDiffusion(
        denoise_fn,
        #self.autoencoder,
        channels=image_n_channel,
        # num_sample_steps=n_steps,
        pred_mode="x_start",
        loss_type="l2",
        device=device,
        clamp_range=(0, 1),
    ).cuda()
    diffusion.set_new_noise_schedule(
        sqrt_etas=make_sqrt_etas_schedule(n_timestep=n_steps)
    )

    diffusion = diffusion.to(device)

    # model, optimizer and lr scheduler
    diffusion_dp = (
        # nn.DataParallel(diffusion, device_ids=[0, 1], output_device=device)
        diffusion
    )
    ema_updater = EmaUpdater(
        diffusion_dp, deepcopy(diffusion_dp), decay=0.995, start_iter=20_000
    )
    opt_r = torch.optim.AdamW(res_model.parameters(), lr=lr_d,weight_decay=1e-4)
    opt_d = torch.optim.AdamW(denoise_fn.parameters(), lr=lr_d, weight_decay=1e-4)
    # opt_g = torch.optim.AdamW(guidance_net.parameters(), lr=1e-4, weight_decay=1e-3)

    scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
        opt_d, milestones=[60_000, 100_000, 150_000], gamma=0.2
    )
    # scheduler_g = torch.optim.lr_scheduler.MultiStepLR(
    #     opt_g, milestones=[100_000, 200_000, 300_000], gamma=0.05
    # )
    schedulers = StepsAll(scheduler_d)
    loss_mse = torch.nn.MSELoss()
    # training
    if pretrain_iterations is not None:
        iterations = pretrain_iterations
        logger.print("load previous training with {} iterations".format(iterations))
    else:
        iterations = 0
    while iterations <= max_iterations:
        for i, (pan, lms, hr, wavelets) in enumerate(dl_train, 1):
            pan, lms, hr, wavelets = map(lambda x: x.cuda(), (pan, lms, hr, wavelets))
            cond, _ = einops.pack([lms, pan, F.interpolate(wavelets, size=lms.shape[-1], mode='bilinear')], "b * h w")

            #opt_r.zero_grad()
            opt_d.zero_grad()

            e_0 = hr - lms
            # cond[:, :image_n_channel] = g_sr

            # T = torch.full((e_0.size(0),), n_steps - 1, device=device, dtype=torch.long)
            # e_T = diffusion_dp.q_sample(e_0,T)
            # noise = torch.randn_like(e_T, device=device)
            # z_sample = diffusion_dp.prior_sample(lms, noise) #x_T
            # z_sample = diffusion_dp.model.forward(z_sample, T, cond, self_cond=None)
            # e_T_pred = diffusion_dp.q_sample(z_sample, T)
            # #print(f"e_T:{e_T.max()}")
            # #print(e_T.mean())
            # #print(f"e_T_pred:{(e_T_pred-e_T).mean()}")
            # #print(e_T_pred.mean())
            # save_fig(e_T,e_T_pred,rgb_channel[dataset_name],"./samples/comp_e_T.png")

            diff_loss, recon_x = diffusion_dp(x=hr, y=lms, cond=cond)
            #res_loss = loss_mse(e_T_pred,e_T)

            #res_loss.backward()
            diff_loss.backward()
            #recon_x = recon_x + lms

            # do a grad clip on diffusion model
            grad_clip(
                diffusion_dp.model.parameters(),
                mode="norm",
                value=0.003,
            )

            opt_r.step()
            opt_d.step()
            # opt_g.step()
            ema_updater.update(iterations)
            schedulers.step()

            iterations += 1
            # logger.print(
            #     f"[iter {iterations}/{max_iterations}: "
            #     + f"d_lr {get_lr_from_optimizer(opt_d): .6f}] - "
            #     + f"denoise loss {diff_loss:.6f} "
            # )
            # if iterations %1000 == 0:
            #     save_fig(e_T,e_T_pred,rgb_channel[dataset_name],f"./samples/e_T/iter_{iterations}.png")
            #     print(e_0.max())
            #     print(e_T.max())
            #     print(e_T_pred.max())
            #     logger.print(f"res_loss:{res_loss}")

            # test predicted sr
            if show_recon and iterations % 1000 == 0:
                # NOTE: only used to validate code
                print(f"minus_recon:{(e_0-recon_x).mean()}")
                analysis_d = AnalysisPanAcc()
                analysis_d(hr, recon_x+lms)
                # analysis_g(hr, g_sr)

                logger.print("---recon result---")
                logger.print(analysis_d.last_acc)
                save_fig(e_0,recon_x,rgb_channel[dataset_name],f"./samples/recon_x/recon_{iterations}.png")
                save_fig(hr, recon_x+lms, rgb_channel[dataset_name], f"./samples/recon_x/img_{iterations}.png")
                logger.print("Finished")
                # recon_x = recon_x[:64]
                #
                # x = tv.utils.make_grid(recon_x, nrow=8, padding=0).cpu()
                # x = x.clip(0, 1)  # for no warning
                # fig, ax = plt.subplots(figsize=(x.shape[-1] // 100, x.shape[-2] // 100))
                # x_show = (
                #     x.permute(1, 2, 0).detach().numpy()[..., rgb_channel[dataset_name]]
                # )
                # ax.imshow(x_show)
                # ax.set_axis_off()
                # plt.tight_layout(pad=0)
                # # plt.show()
                # fig.savefig(
                #     f"./samples/recon_x/iter_{iterations}.png",
                #     dpi=200,
                #     bbox_inches="tight",
                #     pad_inches=0,
                # )

            # do some sampling to check quality
            if iterations % 1000 == 0:

                diffusion_dp.model.eval()
                ema_updater.ema_model.model.eval()
                # guidance_net.eval()
                # setattr(ema_updater.ema_model, "image_size", 256)
                # diffusion_dp.reset_logsnr(
                #     noise_d=None, noise_d_low=64, noise_d_high=256
                # )

                analysis_d = AnalysisPanAcc()
                # analysis_g = AnalysisPanAcc()
                with torch.no_grad():
                    for i, (pan, lms, hr, wavelets) in enumerate(dl_valid, 1):
                        pan, lms, hr, wavelets = map(lambda x: x.cuda(), (pan, lms, hr, wavelets))

                        cond, _ = einops.pack([lms, pan, F.interpolate(wavelets, size=lms.shape[-1], mode='bilinear')], "b * h w")

                        # g_sr = guidance_net(cond)
                        # clamp_diff_fn = clamp_fn(g_sr)
                        # ema_updater.ema_model.set_clamp_fn(clamp_diff_fn)
                        # cond[:, :image_n_channel] = g_sr
                        # sr = ema_updater.ema_model(
                        #     cond, mode='ddpm_sample'
                        # )
                        # sr = ema_updater.ema_model.sample(batch_size=cond.shape[0],
                        #                                   cond=cond)
                        # sr = sr + g_sr
                        #e_T_pred = res_model(lms)
                        sr = ema_updater.ema_model(y=lms,cond=cond, mode="ddpm_sample")
                        save_fig(hr-lms,sr,[4,2,0],"./samples/com_e_o.png")
                        print(f"minus:{(hr-lms-sr).mean()}")
                        #sr=sr.clamp(0,1)
                        sr = sr + lms

                        hr = hr.to(sr.device)
                        analysis_d(hr, sr)
                        # analysis_g(hr, g_sr)

                        save_fig(hr,sr,rgb_channel[dataset_name],f"./samples/valid_samples/iter_{iterations}.png")
                        logger.print("---diffusion result---")
                        logger.print(analysis_d.last_acc)
                        # logger.print(analysis_g.last_acc)
                        break
                    if i != 1:
                        logger.print("---diffusion result---")
                        logger.print(analysis_d.print_str())
                        # logger.print(analysis_g.print_str())

                diffusion_dp.model.train()
                # guidance_net.train()
                setattr(ema_updater.model, "image_size", 64)
                # diffusion_dp.reset_logsnr(
                #     noise_d=None, noise_d_low=None, noise_d_high=None
                # )

                torch.save(
                    ema_updater.on_fly_model_state_dict,
                    f"./weights/diffusion_{dataset_name}_iter_{iterations}.pth",
                )
                torch.save(
                    ema_updater.ema_model_state_dict,
                    f"./weights/ema_diffusion_{dataset_name}_iter_{iterations}.pth",
                )
                # torch.save(
                #     guidance_net.state_dict(),
                #     f"./weights/guidance_net_{dataset_name}_iter_{iterations}.pth",
                # )
                logger.print("save model")

                logger.log_scalars("diffusion_perf", analysis_d.acc_ave, iterations)
                # logger.log_scalars("guidance_net_perf", analysis_g.acc_ave, iterations)
                logger.print("saved performances")

            # log loss
            if iterations % 50 == 0:
                logger.log_scalar("denoised_loss", diff_loss.item(), iterations)


@torch.no_grad()
def test_fn(
    test_data_path,
    weight_path,
    schedule_type="cosine",
    batch_size=320,
    n_steps=1500,
    show=False,
    device="cuda:0",
    full_res=False,
    dataset_name="gf2",
    division=1023,
):
    # lazy import
    # from diffusion.diffusion_ddpm_google import GaussianDiffusion

    # from models.unet_model_google import UNet
    # from models.sr3 import UNetSR3 as UNet
    from diffusion.elucidated_diffusion import ElucidatedDiffusion
    from models.uvit import UViT
    from models.sr3 import UNetSR3 as Unet
    from diffusion.diffusion_ddpm_google import GaussianDiffusion

    torch.cuda.set_device(device)

    # load model
    image_n_channel = 4
    image_size = 256
    # denoise_fn = UViT(
    #     64,
    #     out_dim=8,
    #     channels=17,
    #     attn_dim_head=8,
    #     dual_patchnorm=True,
    # ).to(device)
    denoise_fn = Unet(
        in_channel=image_n_channel,
        out_channel=image_n_channel,
        cond_channel=image_n_channel + 1,
        norm_groups=32,
        channel_mults=(1, 2, 2, 4),  # (64, 32, 16, 8)
        attn_res=(8,),
        dropout=0.2,
        image_size=64,
        self_condition=True,
    ).to(device)
    # guidance_fn = GuidanceNetwork(spectral_num=image_n_channel).to(device)
    denoise_fn = model_load(weight_path, denoise_fn, device=device)
    # guidance_fn = model_load(weight_path[1], guidance_fn, device=device)

    denoise_fn.eval()
    # guidance_fn.eval()
    print(f"load weight {weight_path}")
    # schedule = dict(
    #     schedule=schedule_type,
    #     n_timestep=n_steps,
    #     linear_start=1e-4,
    #     linear_end=2e-2,
    #     cosine_s=8e-3,
    # )
    # diffusion = ElucidatedDiffusion(
    #     denoise_fn,
    #     image_size=image_size,
    #     channels=image_n_channel,
    #     num_sample_steps=n_steps,
    # )
    diffusion = GaussianDiffusion(
        denoise_fn,
        image_size=image_size,
        channels=image_n_channel,
        # num_sample_steps=n_steps,
        pred_mode="x_start",
        loss_type="l1",
        device=device,
        clamp_range=(0, 1),
    )
    diffusion.set_new_noise_schedule(
        betas=make_beta_schedule(schedule="cosine", n_timestep=n_steps, cosine_s=8e-3)
    )
    diffusion = diffusion.to(device)
    # diffusion = nn.DataParallel(diffusion, device_ids=[0, 1], output_device=device)
    # guidance_fn = nn.DataParallel(guidance_fn, device_ids=[0, 1], output_device=device)

    # load dataset
    d_test = h5py.File(test_data_path)
    ds_test = PanDataset(d_test, full_res=full_res, norm_range=False, division=division)
    dl_test = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0
    )

    saved_name = "reduced" if not full_res else "full"

    # do sampling
    preds = []
    sample_times = compute_iters(dl_test.dataset.size, batch_size, dl_test.drop_last)
    analysis = AnalysisPanAcc()
    # for i, (pan, _, lms, hr) in enumerate(dl_test, 1):
    # for i, (pan, lms, gt) in enumerate(dl_test):
    for i, (pan, lms) in enumerate(dl_test):
        print(f"sampling [{i}/{sample_times}]")
        cond = torch.cat([lms, pan], dim=1).to(device)
        # pre_recon = guidance_fn(cond, mode="val")
        sr = diffusion(cond, mode="ddpm_sample")
        sr = sr + lms.cuda()
        # sr = sr + pre_recon
        # sr = unorm(sr.detach().cpu())  # [0, 1]
        # hr = unorm(hr)

        # analysis(sr.detach().cpu(), gt)

        # print(analysis.last_acc)

        if show:
            # hr = hr.detach().clone()[:64]
            # hr = tv.utils.make_grid(hr, nrow=2, padding=0)
            s = tv.utils.make_grid(sr[:64], nrow=4, padding=0).cpu()

            # s = torch.cat([hr, s], dim=-1)  # [b, c, h, 2*w]
            s.clip_(0, 1)
            fig, ax = plt.subplots(
                figsize=(s.shape[2] // 100, s.shape[1] // 100), dpi=200
            )
            ax.imshow(s.permute(1, 2, 0).detach().numpy()[..., [0, 1, 2]])
            ax.set_axis_off()

            # plt.tight_layout()
            # plt.show()
            fig.savefig(
                path_legal_checker(
                    f"./samples/{dataset_name}_test/test_sample_iter_{n_steps}_{saved_name}_part_{i}.png"
                ),
                dpi=200,
                bbox_inches="tight",
                pad_inches=0,
            )

        sr = sr.detach().cpu().numpy()  # [b, c, h, w]
        sr = sr * division  # [0, 2047]
        preds.append(sr.clip(0, division))
        # preds.append(sr)

        # print(f"test acc:\n {analysis.print_str()}")

    # save results
    if not full_res:
        d = dict(  # [b, c, h, w], wv3 [0, 2047]
            gt=d_test["gt"][:],
            ms=d_test["ms"][:],
            lms=d_test["lms"][:],
            pan=d_test["pan"][:],
            sr=np.concatenate(preds, axis=0),
        )
    else:
        d = dict(
            ms=d_test["ms"][:],
            lms=d_test["lms"][:],
            pan=d_test["pan"][:],
            sr=np.concatenate(preds, axis=0),
        )
    model_iterations = weight_path[0].split("_")[-1].strip(".pth")
    savemat(
        f"./samples/mat/test_iter_{model_iterations}_{saved_name}_{dataset_name}.mat", d
    )
    print("save result")


# torch.cuda.set_device(1)
engine_google(
    "data/wv3/train_wv3.h5",
    "data/wv3/valid_wv3.h5",
    # "/home/ZiHanCao/datasets/pansharpening/gf/training_gf2/train_gf2.h5",
    # "/home/ZiHanCao/datasets/pansharpening/gf/reduced_examples/test_gf2_multiExm1.h5",
    # "/home/wutong/proj/HISR/x4/train_cave(with_up)x4_rgb.h5",
    # "/home/wutong/proj/HISR/x4/test_cave(with_up)x4_rgb.h5",
    dataset_name="wv3",
    pretrain_weight="weights/ema_diffusion_wv3_iter_49000.pth",
    # pretrain_iterations=51000,
    show_recon=True ,
    lr_d=1e-4,
    n_steps=15,
    schedule_type="cosine",
    batch_size=64,
    device="cuda:0",
    max_iterations=300_000,
    image_n_channel=8,
    # pretrain_iterations=6000,
    # pretrain_weight='./weights/ema_diffusion_cave_iter_6000.pth',
)

# WV3: 21000
# GF2: 27000
# QB: 27000

# test_fn(
#     # "/home/wutong/proj/HJM_Pansharpening/Pansharpening_new data/test data/h5/WV3/reduce_examples/test_wv3_multiExm1.h5", # raw reduce test data, size 256
#     # "/home/wutong/proj/HJM_Pansharpening/Pansharpening_new data/test data/h5/GF2/reduce_examples/test_gf2_multiExm1.h5",
#     # "/home/wutong/proj/HJM_Pansharpening/Pansharpening_new data/test data/h5/WV3/full_examples/test_wv3_OrigScale_multiExm1.h5",
#     "/home/wutong/proj/HJM_Pansharpening/Pansharpening_new data/test data/h5/QB/full_examples/test_qb_OrigScale_multiExm1.h5",
#     # "/home/wutong/proj/HJM_Pansharpening/Pansharpening_new data/test data/h5/GF2/full_examples/test_gf2_OrigScale_multiExm1.h5",
#     # orig scale test data, size 512
#     # "/home/ZiHanCao/datasets/pansharpening/pansharpening_test/test_gf2_OrigScale_multiExm1.h5",
#     # "/home/ZiHanCao/datasets/pansharpening/pansharpening_test/test_qb_OrigScale_multiExm1.h5",
#     # './clip_org_scale_test_wv3/clip_test_wv3.h5',  # clipped data, size 64
#     # ["./weights/ema_diffusion_gf2_iter_27000.pth", "./weights/guidance_gf2_iter_27000.pth"],
#     "./weights/ema_diffusion_qb_iter_21000.pth",
#     batch_size=10,
#     n_steps=500,
#     show=True,
#     dataset_name="qb",
#     division=2047.0,
#     full_res=True,
#     device="cuda:1",
# )
