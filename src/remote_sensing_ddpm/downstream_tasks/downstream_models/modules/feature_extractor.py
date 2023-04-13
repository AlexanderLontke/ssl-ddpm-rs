from typing import List, Union
from enum import Enum

import torch
from torch import nn
import pytorch_lightning as pl


class FeatureSection(Enum):
    ENCODER = "encoder"
    MIDDLE = "middle"
    DECODER = "decoder"
    BOTH = "both"


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        diffusion_pl_module: pl.LightningModule,
        checkpoint_path: str,
        feature_section: Union[str, FeatureSection],
        feature_levels: List[int],
        vectorize_output: bool,
        t: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.diffusion_pl_module = diffusion_pl_module.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            p_theta_model=diffusion_pl_module.p_theta_model,
            **kwargs,
        )

        self.feature_section = FeatureSection(feature_section)
        self.feature_levels = feature_levels
        self.vectorize_output = vectorize_output
        self.t = torch.Tensor(t).to(diffusion_pl_module.device)

    @torch.no_grad()
    def forward(self, x, *args, **kwargs):
        # Add noise to sample...
        x_noisy = self.diffusion_pl_module.q_sample(x_0=x, t=self.t)
        # ... and pass it through the model
        encoder_features, middle_features, decoder_features = self.p_theta_model(
            x_noisy, time=self.t, feat_need=True, *args, **kwargs
        )

        # Select side of the model to get the features from
        if self.feature_section == FeatureSection.ENCODER:
            features_all_levels = encoder_features
            del decoder_features, middle_features
        elif self.feature_section == FeatureSection.MIDDLE:
            features_all_levels = middle_features
            del encoder_features, decoder_features
        elif self.feature_section == FeatureSection.DECODER:
            features_all_levels = decoder_features
            del encoder_features, middle_features
        elif self.feature_section == FeatureSection.BOTH:
            features_all_levels = torch.cat([encoder_features, decoder_features])
            del middle_features
        else:
            raise NotImplementedError(
                f"Feature section: {self.feature_section} is not implemented"
            )
        final_features = []
        for level in self.feature_levels:
            final_features.append(features_all_levels[level])

        if self.vectorize_output:
            # TODO maybe add pooling?
            final_features = [features.flatten() for features in final_features]
            final_features = torch.concat(final_features)
        return final_features
