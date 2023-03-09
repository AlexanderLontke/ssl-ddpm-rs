from functools import partial

import numpy as np
from numpy import random

import torch
from torch import nn

# Util
from remote_sensing_ddpm.model.diffusion_process.util import extract_into_tensor, DiffusionTarget

# Beta Schedule
from remote_sensing_ddpm.model.diffusion_process.beta_schedule import make_beta_schedule


class DDPM:
    def __init__(
        self,
        diffusion_model: nn.Module,
        diffusion_target: DiffusionTarget,
        schedule_type: str,
        beta_schedule_steps: int,
        beta_schedule_linear_start: float,
        beta_schedule_linear_end: float,
        device: str,
    ):
        self.device = torch.device(device)
        self.diffusion_model = diffusion_model
        self.diffusion_target = diffusion_target

        # Helper function
        to_torch = partial(
            torch.tensor, dtype=torch.float32, device=self.device, requires_grad=False
        )

        # Fix beta schedule
        self.beta_schedule_steps = beta_schedule_steps
        self.betas = to_torch(
            make_beta_schedule(
                schedule=schedule_type,
                n_timestep=beta_schedule_steps,
                linear_start=beta_schedule_linear_start,
                linear_end=beta_schedule_linear_end,
            )
        )

        # Cache often used values
        self.alphas = to_torch(1.0 - self.betas)
        self.alphas_cumprod = to_torch(np.cumprod(self.alphas))
        self.sqrt_alphas_cumprod = to_torch(np.sqrt(self.alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1 - self.alphas_cumprod))

        # Setup loss
        self.loss = nn.MSELoss()

    def training_step(self, x_0):
        t = torch.randint(0, self.beta_schedule_steps, (x_0.shape[0],), device=self.device).long()
        return self.p_losses(
            x_0=x_0,
            t=t,
        )

    def p_losses(self, x_0, t):
        """
        Calculates the variational lower bound loss of p_{\theta}(x_{t-1}|x_{t}) based on the simplified
        difference between the distribution q(x_{t}|x_{t+1}) and p_{\theta}(x_{t}|x_{t+1})
        :param x_0: sample without any noise
        :param t: timestep for which the difference of distributions should be calculated
        :return: Simplified loss for the KL_divergence of q and p_{\theta}
        """
        noised_x, noise = self.q_sample(
            x_0=x_0,
            t=t,
        )
        model_x = self.diffusion_model(noised_x, t)

        # Determine target
        if self.diffusion_target == DiffusionTarget.X_0:
            target = x_0
        elif self.diffusion_target == DiffusionTarget.EPS:
            target = noise
        else:
            raise NotImplementedError(f"Diffusion target {self.diffusion_target} not supported")

        loss_t_simple = self.loss(model_x, target)
        return loss_t_simple

    def q_sample(self, x_0, t):
        """
        Samples x_t from the distribution q(x_t|x_{t-1}) given x_0 and t, made possible through the
        re-parametrization trick
        :param x_0: sample without any noise
        :param t: timestep for which a noised sample x_t should be created
        :return: x_t and noise used to generate it
        """
        epsilon_noise = torch.randn_like(x_0)
        noised_x = (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
            * epsilon_noise
        )
        return noised_x, epsilon_noise
