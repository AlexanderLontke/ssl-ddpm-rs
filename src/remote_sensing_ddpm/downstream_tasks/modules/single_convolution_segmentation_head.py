import torch
from torch import nn

def get_kernel_size(
    in_channels: int, out_channels: int, stride: int = 1, padding: int = 0
) -> int:
    # output_size = [(input_size−K+2P)/S]+1
    # output_size - 1 = (input_size−K+2P)/S | *S
    # ((output_size - 1)S) = input_size−K+2P |+K | - ((output_size - 1)S)
    # K = input_size + 2P - ((output_size - 1)S)
    return int(in_channels + 2 * padding - (out_channels - 1) * stride)


class SingleConvolutionSegmentationHead(nn.Module):
    def __init__(self, feature_map_channels: int, output_map_channels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.downstream_layer = nn.Conv2d(
            in_channels=feature_map_channels,
            kernel_size=get_kernel_size(
                in_channels=feature_map_channels, out_channels=output_map_channels
            ),
            out_channels=output_map_channels,
        )

    def forward(self, batch):
        output_map = self.downstream_layer(batch)
        return torch.argmax(output_map, dim=1)
