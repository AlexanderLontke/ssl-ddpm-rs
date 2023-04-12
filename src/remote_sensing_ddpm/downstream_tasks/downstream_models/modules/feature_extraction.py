from typing import List, Union
from enum import Enum

import torch
from torch import nn


class FeatureSection(Enum):
    ENCODER = "encoder"
    DECODER = "decoder"
    BOTH = "both"


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        representation_model: nn.Module,
        feature_section: Union[str, FeatureSection],
        feature_levels: List[int],
        vectorize_output: bool,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.representation_model = representation_model
        self.feature_section = FeatureSection(feature_section)
        self.feature_levels = feature_levels
        self.vectorize_output = vectorize_output

    def forward(self, x, *args, **kwargs):
        encoder_features, decoder_features = self.representation_model(
            x, *args, **kwargs
        )

        # Select side of the model to get the features from
        if self.feature_section == FeatureSection.ENCODER:
            features_all_levels = encoder_features
            del decoder_features
        elif self.feature_section == FeatureSection.DECODER:
            features_all_levels = decoder_features
            del encoder_features
        elif self.feature_section == FeatureSection.BOTH:
            features_all_levels = torch.cat([encoder_features, decoder_features])
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
