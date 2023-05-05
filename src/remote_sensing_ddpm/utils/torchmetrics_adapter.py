import torch
from torch import nn

from torchmetrics.metric import Metric


class TorchmetricsAdapter(nn.Module):
    def __init__(
        self,
        torchmetrics_module: Metric,
        apply_argmax: bool,
        device: str,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.device = torch.device(device)
        self.torchmetrics_module = torchmetrics_module.to(self.device)
        self.apply_argmax = apply_argmax

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        if self.apply_argmax:
            inputs = torch.argmax(inputs, dim=1)
        return self.torchmetrics_module(inputs, targets)
