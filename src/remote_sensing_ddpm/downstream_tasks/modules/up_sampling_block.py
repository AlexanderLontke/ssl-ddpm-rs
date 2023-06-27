from torch import nn


class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size:int, stride:int):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
        )
        # self.conv1 = nn.Conv2d(
        #     in_channels=out_channels, out_channels=out_channels, padding=1, kernel_size=3,
        # )
        # self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels)
        # self.conv2 = nn.Conv2d(
        #     in_channels=out_channels, out_channels=out_channels, padding=1, kernel_size=3,
        # )
        # self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, batch):
        # x = self.relu(self.batch_norm1(self.conv1(self.up_conv(batch))))
        # return self.relu(self.batch_norm2(self.conv2(x)))
        return self.relu(self.up_conv(batch))
