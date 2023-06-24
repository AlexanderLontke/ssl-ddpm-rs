from torch import nn

class MockFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        data_key,
        padding,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.data_key = data_key
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
    def forward(self, batch):
        output = self.conv_layer(
            batch[self.data_key]
        )
        return output