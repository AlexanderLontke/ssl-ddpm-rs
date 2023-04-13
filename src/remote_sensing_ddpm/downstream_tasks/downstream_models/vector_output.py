from torch import nn

from remote_sensing_ddpm.downstream_tasks.downstream_models.downstream_task_model import (
    DownstreamTaskModel,
)


class VectorOutputDownstreamTask(DownstreamTaskModel):
    def __init__(self, input_size: int, output_size: int, *args, **kwargs):
        downstream_layer = nn.Linear(in_features=input_size, out_features=output_size,)
        super().__init__(downstream_layer=downstream_layer, *args, **kwargs)
