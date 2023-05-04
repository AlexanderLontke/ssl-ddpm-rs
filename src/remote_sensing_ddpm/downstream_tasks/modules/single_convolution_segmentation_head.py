import torch
from torch import nn

def get_kernel_size(
    input_size: int, output_size: int, stride: int = 1, padding: int = 0
) -> int:
    # output_size = [(input_size−K+2P)/S]+1
    # output_size - 1 = (input_size−K+2P)/S | *S
    # ((output_size - 1)S) = input_size−K+2P |+K | - ((output_size - 1)S)
    # K = input_size + 2P - ((output_size - 1)S)
    return int(input_size + 2 * padding - (output_size - 1) * stride)


class SingleConvolutionSegmentationHead(nn.Module):
    def __init__(self, input_size: int, output_size: int, feature_map_channels: int, output_map_channels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.downstream_layer = nn.Conv2d(
            in_channels=feature_map_channels,
            kernel_size=get_kernel_size(
                input_size=input_size, output_size=output_size
            ),
            out_channels=output_map_channels,
        )

    def forward(self, batch):
        return self.downstream_layer(batch)
