import os
import numpy as np
import torch
from runners.runner_utils import get_beta_schedule
from models.diffusion import Model
from functions.denoising import generalized_repaint_steps, conditional_inpainting_steps


class InpaintingSampleUtils:
    def __init__(self, config, device=None):
        self.config = config

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

        self.model = None

    def load_model(self, ckpt_path):
        model = Model(self.config)
        print(f'Load {ckpt_path}')
        states = torch.load(
            ckpt_path,
            map_location=self.device
        )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        self.model = model
        self.model.eval()

    def run_inference(self, x0_gt, mask_gt, n_steps):
        """
        x0_gt should be in range [-1, 1]
        """
        x0_gt = torch.from_numpy(x0_gt).float()
        mask_gt = torch.from_numpy(mask_gt).int()

        step_skip = self.num_timesteps // n_steps
        seq = range(0, self.num_timesteps, step_skip)
        x = torch.randn_like(x0_gt)

        # xs, _ = generalized_repaint_steps(
        #     x.to(self.device),
        #     x0_gt.to(self.device),
        #     mask_gt.to(self.device),
        #     seq,
        #     self.model,
        #     self.betas,
        #     n_resample,
        #     eta=0.0
        # )
        xs, _ = conditional_inpainting_steps(
            x.to(self.device),
            x0_gt.to(self.device),
            mask_gt.to(self.device),
            seq,
            self.model,
            self.betas,
            self.config.model.invalid_region_val,
            eta=0.0
        )

        x0_pred = xs[-1].cpu().numpy()

        return x0_pred
