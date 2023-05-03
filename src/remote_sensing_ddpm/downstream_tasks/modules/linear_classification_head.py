from torch import nn


class LinearClassificationHead(nn.Module):
    def __init__(self, input_size: int, output_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.downstream_layer = nn.Sequential(
                nn.Linear(in_features=input_size, out_features=output_size,),
                nn.BatchNorm1d(num_features=output_size, eps=1e-5,),
        )

    def forward(self, batch):
        return self.downstream_layer(batch)
