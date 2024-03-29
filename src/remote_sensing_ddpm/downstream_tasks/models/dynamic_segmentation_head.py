from typing import List
import torch
from torch import nn

# Remote sensing ddpm
from remote_sensing_ddpm.downstream_tasks.modules.up_sampling_block import UpSamplingBlock


class DynamicSegmentationHead(nn.Module):
    def __init__(self,output_channels: int,  u_net_channels: List[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(u_net_channels) > 0
        modules = []
        for i in range(len(u_net_channels)-1):
            modules.append(
                UpSamplingBlock(
                    in_channels=u_net_channels[i] * (1 + int(i != 0)),
                    out_channels=u_net_channels[i+1],
                    kernel_size=2,
                    stride=2,
                )
            )
        self.up_convolutions = nn.ModuleList(modules=modules)
        self.output_projection = nn.Conv2d(
            in_channels=u_net_channels[-1]*2, out_channels=output_channels, padding=0, kernel_size=9,
        )

    def forward(self, all_decoder_levels: torch.Tensor):
        assert len(all_decoder_levels) == 5, f"Expected five feature maps but got {len(all_decoder_levels)}"

        x = all_decoder_levels[0]
        for i, module in enumerate(self.up_convolutions):
            x = torch.cat([module(x), all_decoder_levels[i+1]], dim=1)

        # Project into target dimensions
        return self.output_projection(x)
