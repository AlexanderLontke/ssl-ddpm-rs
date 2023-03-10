# Change detection head
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d
from remote_sensing_ddpm.evaluation.baselines.ddpm_cd.cd_modules.psp import _PSPModule
from remote_sensing_ddpm.evaluation.baselines.ddpm_cd.cd_modules.se import (
    ChannelSpatialSELayer,
)


def get_in_channels(feat_scales, inner_channel, channel_multiplier):
    """
    Get the number of input layers to the change detection head.
    """
    in_channels = 0
    for scale in feat_scales:
        if scale < 3:  # 256 x 256
            in_channels += inner_channel * channel_multiplier[0]
        elif scale < 6:  # 128 x 128
            in_channels += inner_channel * channel_multiplier[1]
        elif scale < 9:  # 64 x 64
            in_channels += inner_channel * channel_multiplier[2]
        elif scale < 12:  # 32 x 32
            in_channels += inner_channel * channel_multiplier[3]
        elif scale < 15:  # 16 x 16
            in_channels += inner_channel * channel_multiplier[4]
        else:
            print("Unbounded number for feat_scales. 0<=feat_scales<=14")
    return in_channels


class AttentionBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU(),
            ChannelSpatialSELayer(num_channels=dim_out, reduction_ratio=2),
        )

    def forward(self, x):
        return self.block(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, time_steps):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim * len(time_steps), dim, 1) if len(time_steps) > 1 else None,
            nn.ReLU() if len(time_steps) > 1 else None,
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class ClassificationHead(nn.Module):
    """
    Change detection head (version 2).
    """

    def __init__(
        self,
        feat_scales,
        out_channels=2,
        inner_channel=None,
        channel_multiplier=None,
        img_size=256,
        time_steps=None,
        use_diffusion=True,
        **kwargs,
    ):
        super(ClassificationHead, self).__init__()

        # Define the parameters of the change detection head
        feat_scales.sort(reverse=True)
        self.feat_scales = feat_scales
        self.in_channels = get_in_channels(
            feat_scales, inner_channel, channel_multiplier
        )
        self.img_size = img_size
        self.time_steps = time_steps
        self.use_diffusion = use_diffusion

        # Convolutional layers before parsing to difference head
        if self.use_diffusion:
            self.decoder = nn.ModuleList()
            for i in tqdm(
                range(0, len(self.feat_scales)),
                desc="Instantiating classification blocks",
            ):
                dim = get_in_channels(
                    [self.feat_scales[i]], inner_channel, channel_multiplier
                )

                self.decoder.append(Block(dim=dim, dim_out=dim, time_steps=time_steps))

                if i != len(self.feat_scales) - 1:
                    dim_out = get_in_channels(
                        [self.feat_scales[i + 1]], inner_channel, channel_multiplier
                    )
                    self.decoder.append(AttentionBlock(dim=dim, dim_out=dim_out))
        else:
            self.decoder = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        # Final classification head
        # Conv Layers
        self.clfr_stg1 = nn.Conv2d(128, 64, kernel_size=5, padding=0)
        self.clfr_stg2 = nn.Conv2d(64, 32, kernel_size=4, padding=0)
        self.clfr_stg3 = nn.Conv2d(32, 16, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(4, 2)
        self.relu = nn.ReLU()
        # Fully connected Layers
        self.fc1 = nn.Linear(16 * (27 ** 2), 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, out_channels)

    def forward(self, feats):
        # Decoder
        lvl = 0
        if self.use_diffusion:
            for layer in self.decoder:
                if isinstance(layer, Block):
                    f = feats[0][self.feat_scales[lvl]]
                    for i in range(1, len(self.time_steps)):
                        f = torch.cat((f, feats[i][self.feat_scales[lvl]]), dim=1)

                    layer_output = layer(f)
                    # Add skip connection
                    if lvl != 0:
                        layer_output = layer_output + x
                    lvl += 1
                else:
                    layer_output = layer(layer_output)
                    x = F.interpolate(layer_output, scale_factor=2, mode="bilinear")
        else:
            x = self.decoder(feats)
        # Spatial Attention Output-Shape: [batch_size, 128, 256, 256]
        # Classifier
        # After Conv1: 252x252xclfr_emb_dim
        x = self.pool(self.relu(self.clfr_stg1(x)))
        # After Pool1: 125x125x64
        # After Conv2: 122x122x32
        x = self.pool(self.relu(self.clfr_stg2(x)))
        # After Pool2: 60x60x32
        # After Conv3: 56x56x16
        x = self.pool(self.relu(self.clfr_stg3(x)))
        # After Pool2: 27x27x16
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
