import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data
from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
import torchvision.utils as tvu
from matplotlib import colors
import matplotlib.pyplot as plt
from functions.denoising import generalized_steps, generalized_inpainting_steps
from runners.runner_utils import get_beta_schedule


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, mask) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                mask = mask.to(self.device)
                x_cond = x.clone()
                e_cond = torch.randn_like(x_cond)
                x_cond[mask == 0] = e_cond[mask == 0][:]

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, x_cond, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained:
            # if getattr(self.config.sampling, "ckpt_id", None) is None:
            if self.args.ckpt_id is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.args.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        elif self.args.sequence_grid_plot:
            self.sample_sequence_grid_plot(model)
        elif self.args.sample_grid_plot:
            self.sample_grid_plot(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def forward_plot(self):
        config = self.config
        args = self.args
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        n_sample = 10
        out_png_dir = os.path.join(args.exp, "image_samples", f'forward_sampling')
        print(f'Save to {out_png_dir}')
        os.makedirs(out_png_dir, exist_ok=True)
        for sample_idx, (x, y) in tqdm.tqdm(enumerate(train_loader), total=n_sample):
            if sample_idx >= n_sample:
                break

            x = data_transform(self.config, x)
            b = self.betas.cpu()
            for t in range(0, self.num_timesteps, 100):
                e = torch.randn_like(x)
                t = torch.tensor([t])
                a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
                x = x * a.sqrt() + e * (1.0 - a).sqrt()
                x = inverse_data_transform(config, x)
                out_png = os.path.join(out_png_dir, f'{sample_idx}_{t[0]}.png')
                tvu.save_image(
                    x[0], out_png
                )

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        # total_n_samples = 50000
        total_n_samples = 100
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_sequence(self, model):
        config = self.config

        out_png_dir = os.path.join(self.args.image_folder, f'sequence')
        os.makedirs(out_png_dir, exist_ok=True)

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                png_file_name = os.path.join(out_png_dir, f"{j}_{i}.png")
                print(f'Save to {png_file_name}')
                tvu.save_image(
                    # x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                    x[i][j], png_file_name
                )

    def sample_sequence_grid_plot(self, model):
        config = self.config

        out_png_dir = os.path.join(self.args.image_folder, f'sequence_grid_plot')
        os.makedirs(out_png_dir, exist_ok=True)
        print(f'Save png files to {out_png_dir}')

        n_sample = 10

        # Let's assume we run with 1000 steps DDIM.
        show_grid_idx = [
            [0, 100, 200, 300, 400],
            [500, 600, 700, 800, 900],
            [950, 970, 990, 995, 1000]
        ]

        for sample_idx in tqdm.tqdm(range(n_sample), total=n_sample):
            x = torch.randn(
                1,
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )
            with torch.no_grad():
                x_pred_sequence, _ = self.sample_image(x, model, last=False)

            # x_pred_sequence = torch.randn(
            #     1000,
            #     1,
            #     config.data.channels,
            #     config.data.image_size,
            #     config.data.image_size,
            #     device=self.device
            # )

            show_img_grid = []
            for row_idx in show_grid_idx:
                show_img_row = []
                for idx in row_idx:
                    show_img = x_pred_sequence[idx - 1][0] if idx > 0 else x[0]
                    show_img = inverse_data_transform(config, show_img)
                    show_img = show_img.cpu().numpy()
                    show_img_row.append(show_img)
                show_img_grid.append(show_img_row)

            fig = plt.figure(constrained_layout=False)
            gs = fig.add_gridspec(nrows=3, ncols=5, wspace=0.01, hspace=0.01)

            for ax_row_idx, show_row_idx_list in enumerate(show_grid_idx):
                for ax_col_idx, show_idx in enumerate(show_row_idx_list):
                    img_ax = fig.add_subplot(gs[ax_row_idx, ax_col_idx])
                    img_ax.set_axis_off()
                    img_ax.imshow(
                        show_img_grid[ax_row_idx][ax_col_idx][0],
                        interpolation=None,
                        cmap='gray',
                        norm=colors.Normalize(vmin=0, vmax=1)
                    )
                    img_ax.annotate(
                        f't = {1000 - show_idx}',
                        xy=(.95, .05),
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        xycoords='axes fraction',
                        color='yellow',
                        fontsize=5
                    )

            out_png = os.path.join(out_png_dir, f'{sample_idx}.png')
            # print(f'\nSave to {out_png}\n')
            plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1, dpi=300)

    def sample_grid_plot(self, model):
        config = self.config
        out_png_dir = self.args.image_folder
        os.makedirs(out_png_dir, exist_ok=True)

        n_row = 3
        n_col = 5
        n_sample = n_row * n_col

        p_bar = tqdm.tqdm(range(n_sample), total=n_sample)

        show_img_grid = []
        for row_idx in range(n_row):
            show_img_row = []
            for col_idx in range(n_col):
                x = torch.randn(
                    1,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                with torch.no_grad():
                    x = self.sample_image(x, model)

                x = inverse_data_transform(config, x)
                x = x.cpu().numpy()
                show_img_row.append(x[0])

                p_bar.update()

            show_img_grid.append(show_img_row)

        fig = plt.figure(constrained_layout=False)
        gs = fig.add_gridspec(nrows=3, ncols=5, wspace=0.01, hspace=0.01)

        for ax_row_idx in range(n_row):
            for ax_col_idx in range(n_col):
                img_ax = fig.add_subplot(gs[ax_row_idx, ax_col_idx])
                img_ax.set_axis_off()
                img_ax.imshow(
                    show_img_grid[ax_row_idx][ax_col_idx][0],
                    interpolation=None,
                    cmap='gray',
                    norm=colors.Normalize(vmin=0, vmax=1)
                )

        out_png = os.path.join(out_png_dir, f'{self.args.ckpt_id}.png')
        print(f'Save to {out_png}')
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass


class SampleSpecificUtils:
    def __init__(self, config, exp_path, log_name, device=None):
        self.config = config
        self.exp_path = exp_path
        self.log_name = log_name

        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

    def _load_model(self, ckpt_id):
        model = Model(self.config)
        ckpt_path = os.path.join(self.exp_path, "logs", self.log_name, f'ckpt_{ckpt_id}.pth')
        print(f'Load {ckpt_path}')
        states = torch.load(
            ckpt_path,
            map_location=self.device
        )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        return model

    def get_x0_prediction(self, x0, run_ts, show_ts, show_ckpts, out_png_dir):
        """
        1. Get the input array (with different noise levels)
        2. Get the predictions for x0

        x0 should be in range [0, 1]
        """
        os.makedirs(out_png_dir, exist_ok=True)

        x0 = data_transform(self.config, x0)
        x0 = torch.from_numpy(x0).float()
        b = self.betas.cpu()

        show_img_dict = {}

        num_timesteps = self.num_timesteps
        # num_timesteps = 5

        xs = []
        es = []
        ats = []
        show_xs = {}
        # for t in tqdm.tqdm(range(0, num_timesteps), total=num_timesteps, desc='Generate the noisy inputs'):
        for t in tqdm.tqdm(run_ts, total=len(run_ts), desc='Generate the noisy inputs'):
            e = torch.randn_like(x0)
            t = torch.tensor([t])
            a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
            x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
            xs.append(x)
            es.append(e)
            ats.append(a)

            if t in show_ts:
                show_xs[t] = inverse_data_transform(self.config, x)

        show_img_dict['input'] = show_xs
        for t, a in zip(run_ts, ats):
            a_sqrt = a.sqrt()[0][0]
            print(f'{int(t)} - {float(a_sqrt):.4f}')

        for ckpt_id in show_ckpts:
            pred_x0s = []
            errs = []
            show_pred_x0s = {}
            show_pred_diffs = {}
            model = self._load_model(ckpt_id)
            model.eval()
            # for t in tqdm.tqdm(range(0, num_timesteps), total=num_timesteps,
            #                    desc=f'Inference with ckpt {ckpt_id}'):
            for t_idx, t in tqdm.tqdm(enumerate(run_ts), total=len(run_ts),
                               desc=f'Inference with ckpt {ckpt_id}'):
                with torch.no_grad():
                    et = model(xs[t_idx].to(self.device), torch.tensor([t]).float().to(self.device))
                    et = et.to('cpu')
                    err = (es[t_idx] - et).square().sum(dim=(1, 2, 3)).mean(dim=0)
                    errs.append(err)

                    at = ats[t_idx]
                    pred_x0 = (xs[t_idx] - et * (1 - at).sqrt()) / at.sqrt()
                    pred_x0s.append(pred_x0)

                    if t in show_ts:
                        show_pred_x0s[t] = inverse_data_transform(self.config, pred_x0)
                        show_pred_diffs[t] = (pred_x0 - x0).abs() / 2.

            show_img_dict[f'ckpt_{ckpt_id}'] = show_pred_x0s
            show_img_dict[f'ckpt_{ckpt_id}_diff'] = show_pred_diffs

            err_mean = np.mean(errs)
            print(err_mean)

        for show_tag, img_dict in show_img_dict.items():
            for t, img in img_dict.items():
                t = t + 1
                t = int(t)
                png_path = os.path.join(out_png_dir, f'{show_tag}_{t}.png')
                tvu.save_image(img[0], png_path)


    def get_x0_prediction_along_trajectory(self, x0, ts, ckpt_id, out_png_dir):
        os.makedirs(out_png_dir, exist_ok=True)
        print(f'Save to {out_png_dir}')

        x0 = data_transform(self.config, x0)
        x0 = torch.from_numpy(x0).float()
        b = self.betas.cpu()

        show_pred_x0s = {}
        show_pred_diffs = {}
        model = self._load_model(ckpt_id)
        model.eval()

        e = torch.randn_like(x0)
        for t_idx in tqdm.tqdm(ts, total=len(ts)):
            t = torch.tensor([t_idx])
            a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
            x = x0 * a.sqrt() + e * (1.0 - a).sqrt()

            seq = range(0, t_idx)
            _, x0_preds = generalized_steps(x.to(self.device), seq, model, self.betas)
            x0_pred = x0_preds[-1][0]
            show_pred_x0s[t_idx] = inverse_data_transform(self.config, x0_pred)
            show_pred_diffs[t_idx] = (x0_pred - x0).abs() / 2.

        for show_tag, img_dict in zip(
                [f'ckpt_{ckpt_id}', f'ckpt_{ckpt_id}_diff'], [show_pred_x0s, show_pred_diffs]):
            for t, img in img_dict.items():
                t = t + 1
                t = int(t)
                png_path = os.path.join(out_png_dir, f'{show_tag}_{t}.png')
                tvu.save_image(img[0], png_path)

    def get_inpainting_x0_prediction(self, x0_gt, mask_gt, n_steps, n_resample, ckpt_id, n_output_steps, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f'Save to {output_dir}')

        x0_gt = data_transform(self.config, x0_gt)
        x0_gt = torch.from_numpy(x0_gt).float()
        mask_gt = torch.from_numpy(mask_gt).int()

        x = torch.randn_like(x0_gt)

        model = self._load_model(ckpt_id)
        model.eval()

        step_skip = self.num_timesteps // n_steps
        seq = range(0, self.num_timesteps, step_skip)

        xs, x0_preds = generalized_inpainting_steps(
            x.to(self.device),
            x0_gt.to(self.device),
            mask_gt.to(self.device),
            seq,
            model,
            self.betas,
            n_resample,
            eta=0.0
        )

        # print(len(xs))
        n_output_skip = n_steps // n_output_steps
        output_idx_list = list(range(0, n_steps, n_output_skip))
        output_idx_list.append(len(xs) - 1)
        for x_idx in tqdm.tqdm(output_idx_list, total=len(output_idx_list)):
            x = xs[x_idx].cpu()
            x[mask_gt == 1] = x0_gt[mask_gt == 1][:]
            x = inverse_data_transform(self.config, x)

            t = n_steps - x_idx
            tvu.save_image(
                x[0],
                os.path.join(output_dir, f't_{t}.png')
            )

        x0_gt = inverse_data_transform(self.config, x0_gt)
        tvu.save_image(
            x0_gt[0],
            os.path.join(output_dir, 'x0_gt.png')
        )
        diff_gt = (x0_gt - inverse_data_transform(self.config, xs[-1])).abs()
        tvu.save_image(
            diff_gt[0],
            os.path.join(output_dir, 'diff.png')
        )