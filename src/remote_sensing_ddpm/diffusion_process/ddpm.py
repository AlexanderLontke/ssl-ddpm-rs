from typing import Optional
from functools import partial

import numpy as np

from tqdm import tqdm

import torch
from torch import nn
import pytorch_lightning as pl

# Util
from remote_sensing_ddpm.constants import (
    TRAINING_LOSS_METRIC_KEY
)
from remote_sensing_ddpm.diffusion_process.util import (
    extract_into_tensor,
)
from remote_sensing_ddpm.constants import (
    DiffusionTarget,
)
from remote_sensing_ddpm.u_net_model.util import default

# Beta Schedule
from remote_sensing_ddpm.diffusion_process.beta_schedule import make_beta_schedule


class LitDDPM(pl.LightningModule):
    def __init__(
        self,
        p_theta_model: nn.Module,
        diffusion_target: DiffusionTarget,
        schedule_type: str,
        beta_schedule_steps: int,
        beta_schedule_linear_start: float,
        beta_schedule_linear_end: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.diffusion_model = p_theta_model
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

        # Cache values often used for approximating p_{\theta}(x_{t-1}|x_{t})
        self.alphas = to_torch(1.0 - self.betas)
        self.alphas_cumprod = to_torch(np.cumprod(self.alphas))
        self.sqrt_alphas_cumprod = to_torch(np.sqrt(self.alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1 - self.alphas_cumprod))
        # Cache values often used to sample from p_{\theta}(x_{t-1}|x_{t})
        self.sqrt_alphas = to_torch(np.sqrt(self.alphas))
        self.sqrt_betas = to_torch(np.sqrt(self.betas))

        # Setup loss
        self.loss = nn.MSELoss()

    # Methods relating to approximating p_{\theta}(x_{t-1}|x_{t})
    def training_step(self, x_0):
        t = torch.randint(
            0, self.beta_schedule_steps, (x_0.shape[0],), device=self.device
        ).long()
        loss = self.p_loss(x_0=x_0, t=t)
        self.log(
            TRAINING_LOSS_METRIC_KEY, loss, prog_bar=True, on_epoch=True,
        )
        return loss

    def p_loss(self, x_0, t):
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
            raise NotImplementedError(
                f"Diffusion target {self.diffusion_target} not supported"
            )

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

    # Methods related to sampling from the distribution p_{\theta}(x_{t-1}|x_{t})
    @torch.no_grad()
    def p_sample(self, x_t, t):
        """
        Method implementing one DDPM sampling step
        :param x_t: sample at current timestep
        :param t: denotes current timestep
        :return: x at timestep t-1
        """
        z = torch.rand((1,)) if t > 1 else 0
        model_estimation = self.diffusion_model(x_t, t)
        # Note: 1 - alpha_t = beta_t
        x_t_minus_one = (
            1.0
            / self.sqrt_alphas[t]
            * (
                x_t
                - (self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t])
                * model_estimation
            )
            + self.sqrt_betas[t] * z
        )
        return x_t_minus_one

    @torch.no_grad()
    def p_sample_loop(self, shape, starting_noise: Optional[torch.Tensor] = None):
        x_t = default(starting_noise, lambda: torch.randn(shape, device=self.device))
        for t in tqdm(
            reversed(range(self.beta_schedule_steps)),
            desc="DDPM sampling:",
            total=self.beta_schedule_steps,
        ):
            x_t = self.p_sample(
                x_t=x_t,
                t=t,
            )
        return x_t
