from abc import ABC
from torch import nn


class DownstreamTaskModel(nn.Module):
    def __init__(
        self, feature_extractor: nn.Module, downstream_layer: nn.Module, freeze_fe: bool = True, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Setup feature extractor
        self.feature_extractor = feature_extractor
        if freeze_fe:
            self.feature_extractor.eval()
            self.feature_extractor.requires_grad_(requires_grad=False)

        # Setup trainable layers
        self.downstream_layer = downstream_layer

    def forward(self, batch, *args, **kwargs):
        features = self.feature_extractor(batch, *args, **kwargs)
        return self.downstream_layer(features)

    def get_parameters_to_optimize(self):
        return self.downstream_layer.parameters()
