from typing import Any, Dict, List, Union
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
        data_key: str,
        diffusion_pl_module: pl.LightningModule,
        checkpoint_path: str,
        map_location: str,
        feature_section: Union[str, FeatureSection],
        feature_levels: List[int],
        vectorize_output: bool,
        t: int,
        p_theta_model_kwargs: Dict[str, Any],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_key = data_key
        self.diffusion_pl_module = diffusion_pl_module.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            p_theta_model=diffusion_pl_module.p_theta_model,
            map_location=map_location,
        )

        self.feature_section = FeatureSection(feature_section)
        self.feature_levels = feature_levels
        self.vectorize_output = vectorize_output
        self.t = t
        self.p_theta_model_kwargs = p_theta_model_kwargs

    @torch.no_grad()
    def forward(self, batch):
        # Get data sample
        x_0 = batch[self.data_key]

        # Determine any further required inputs from the data set
        model_kwargs = self.diffusion_pl_module.get_p_theta_model_kwargs_from_batch(batch=batch)

        # Add noise to sample...
        batch_size, *_ = x_0.shape
        t = torch.full(size=(batch_size, ), fill_value=self.t, device=x_0.device)
        x_noisy, _ = self.diffusion_pl_module.q_sample(x_0=x_0, t=t)


        # ... and pass it through the model
        encoder_features, middle_features, decoder_features = self.diffusion_pl_module.p_theta_model(
            x_noisy, t, **self.p_theta_model_kwargs, **model_kwargs
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
            final_features = [features.flatten(start_dim=1) for features in final_features]
            final_features = torch.concat(final_features)
        return final_features
