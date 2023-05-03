import numpy as np

from torch import nn


class NonLinearClassificationHead(nn.Module):
    def __init__(self, input_size: int, output_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Calculate layer sizes
        log_factor = np.log2(input_size)
        output_layer1 = int(input_size / log_factor)
        output_layer2 = int(input_size / (log_factor ** 2))

        # Instantiate head
        self.downstream_layer = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=output_layer1),
            nn.ReLU(),
            nn.Linear(in_features=output_layer1, out_features=output_layer2),
            nn.ReLU(),
            nn.Linear(in_features=output_layer2, out_features=output_size),
            nn.Sigmoid(),
        )

    def forward(self, batch):
        return self.downstream_layer(batch)
