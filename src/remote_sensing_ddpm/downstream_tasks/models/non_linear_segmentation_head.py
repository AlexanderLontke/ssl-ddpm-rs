from typing import List
import torch
from torch import nn

# Remote sensing ddpm
from remote_sensing_ddpm.downstream_tasks.modules.up_sampling_block import UpSamplingBlock


class NonLinearSegmentationHead(nn.Module):
    def __init__(self, output_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bottleneck_decoder = UpSamplingBlock(
            in_channels=1024, out_channels=1024, kernel_size=2, stride=2,  # 8x8
        )
        self.fourth_level_decoder = UpSamplingBlock(
            in_channels=2048, out_channels=512, kernel_size=2, stride=2,  # 16x16
        )
        self.third_level_decoder = UpSamplingBlock(
            in_channels=1024, out_channels=256, kernel_size=2, stride=2,  # 32x32
        )
        self.second_level_decoder = UpSamplingBlock(
            in_channels=512, out_channels=128, kernel_size=2, stride=2,  # 64x64
        )
        self.output_projection = nn.Conv2d(
            in_channels=256, out_channels=output_classes, padding=1, kernel_size=3,  # 128x128
        )

    def forward(self, all_decoder_levels: torch.Tensor, ):
        assert len(all_decoder_levels) == 5, f"Expected five feature maps but got {len(all_decoder_levels)}"
        bottle_neck, fourth_level, third_level, second_level, first_level = all_decoder_levels
        # Bottleneck: shape [1024, 8, 8]
        x = torch.cat([self.bottleneck_decoder(bottle_neck),  fourth_level], dim=1)
        print("Shape 1", x.shape)
        # Fourth level decoder: shape [1024, 16, 16]
        x = torch.cat([self.fourth_level_decoder(x), third_level], dim=1)
        print("Shape 2", x.shape)
        # Third level decoder: shape [512, 32, 32]
        x = torch.cat([self.third_level_decoder(x), second_level], dim=1)
        print("Shape 3", x.shape)
        # Second level decoder: shape [256, 64, 64]
        x = torch.cat([self.second_level_decoder(x), first_level], dim=1)
        print("Shape 4", x.shape)
        # First level decoder: shape [128, 128, 128]
        # Project into target dimensions
        return self.output_projection(x)
