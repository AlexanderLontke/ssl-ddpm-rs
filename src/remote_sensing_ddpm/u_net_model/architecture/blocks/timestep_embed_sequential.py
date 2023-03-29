from torch import nn

from remote_sensing_ddpm.u_net_model.architecture.blocks.timestep_block import (
    TimestepBlock,
)
from remote_sensing_ddpm.u_net_model.architecture.blocks.spatial_transformer import (
    SpatialTransformer,
)


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return
