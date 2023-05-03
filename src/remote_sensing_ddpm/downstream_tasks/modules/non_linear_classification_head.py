from torch import nn


class NonLinearClassificationHead(nn.Module):
    def __init__(self, input_size: int, output_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.downstream_layer = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=input_size / (2 ** 4)),
            nn.ReLU(),
            nn.Linear(in_features=input_size / (2 ** 4), out_features=input_size / (2 ** 8)),
            nn.ReLU(),
            nn.Linear(in_features=input_size / (2 ** 8), out_features=output_size),
            nn.Sigmoid(),
        )

    def forward(self, batch):
        return self.downstream_layer(batch)
