import torch
from torch import nn


class TorchMinMaxScaler(nn.Module):
    def __init__(
        self,
        minimum_value: float,
        maximum_value: float,
        interval_min: float = -1.0,
        interval_max: float = 1.0,
    ):
        super().__init__()
        # Store Scaler values
        self.minimum_value = minimum_value
        self.maximum_value = maximum_value
        self.interval_min = interval_min
        self.interval_max = interval_max

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        return ((x - self.minimum_value) / (self.maximum_value - self.minimum_value)) * (
            self.interval_max - self.interval_min
        ) + self.interval_min
