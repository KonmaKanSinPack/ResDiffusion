import h5py
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torchvision as tv
from dataset.pan_dataset import PanDataset
from models.diffusion_res import ShiftDiffusion, make_sqrt_etas_schedule
from models.sr3_dwt import UNetSR3 as Unet
from functools import partial
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__=="__main__":

    d_train = h5py.File("data/wv3/train_wv3.h5")
    DatasetUsed = partial(
            PanDataset,
            full_res=False,
            norm_range=False,
            constrain_channel=None,
            division=2047.0,
            aug_prob=0,
            wavelets=True,
        )

    ds_train = DatasetUsed(
        d_train,
        )
    dl_train = DataLoader(
        ds_train,
        batch_size=16,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )

    denoise_fn = Unet(
        in_channel=8,
        out_channel=8,
        lms_channel=8,
        pan_channel=1,
        inner_channel=32,
        norm_groups=1,
        channel_mults=(1, 2, 2, 4),  # (64, 32, 16, 8)
        attn_res=(8,),
        dropout=0.2,
        image_size=64,
        self_condition=True,
    ).to("cuda")
    diffusion = ShiftDiffusion(
        denoise_fn,
        # self.autoencoder,
        channels=8,
        # num_sample_steps=n_steps,
        pred_mode="x_start",
        loss_type="l2",
        device="cuda",
        clamp_range=(0, 1),
    ).cuda()
    diffusion.set_new_noise_schedule(
        sqrt_etas=make_sqrt_etas_schedule(n_timestep=15)
    )
    for (pan,lms,hr,wav) in dl_train:
        print(diffusion.etas)
        print(diffusion.cumsum_alphas)
        e_0 = hr-lms
        b = hr.size(0)
        t = torch.full((b,), 14, device="cpu", dtype=torch.long)
        res_t = diffusion.q_mean_variance(e_0,t).cpu()
        hr = tv.utils.make_grid(e_0, nrow=4, padding=0).cpu()
        x = tv.utils.make_grid(res_t, nrow=4, padding=0).detach().cpu()
        x = x.clip(0, 1)
        s = torch.cat([hr, x], dim=-1)  # [b, c, h, 2*w]
        fig, ax = plt.subplots(
            figsize=(s.shape[-1] // 100, s.shape[-2] // 100)
        )
        ax.imshow(
            s.permute(1, 2, 0)
            .detach()
            .numpy()[..., [4,2,0]]
        )
        ax.set_axis_off()

        # plt.show()
        plt.tight_layout(pad=0)
        fig.savefig(
            f"./samples/test/iter_{114514}.png",
            dpi=200,
            bbox_inches="tight",
            pad_inches=0,
        )
        break