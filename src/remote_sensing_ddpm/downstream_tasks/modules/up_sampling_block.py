from torch import nn


class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size:int, stride:int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
        )
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, batch):
        return self.relu(self.batch_norm(self.conv(batch)))