from torch import nn

from remote_sensing_ddpm.downstream_tasks.downstream_models.downstream_task_model import (
    DownstreamTaskModel,
)


class ClassificationDownstreamTask(DownstreamTaskModel):
    def __init__(self, input_size: int, output_size: int, *args, **kwargs):
        downstream_layer = nn.Sequential(
                nn.Linear(in_features=input_size, out_features=output_size,),
                nn.BatchNorm1d(num_features=output_size, eps=1e-5,),
        )
        super().__init__(downstream_layer=downstream_layer, *args, **kwargs)
