from torch import nn


class MLPEmbedder(nn.Module):
    def __init__(self, input_size: int, embedding_size: int, dropout_p: float = 0.5, depth: int = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert depth > 1, "Mus have at least two layers"
        # The First layer needs to map from input dimension
        layers = [
            nn.Linear(in_features=input_size, out_features=embedding_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
        ]
        for _ in range(depth-2):
            layers.append(nn.Linear(in_features=embedding_size, out_features=embedding_size))
            layers.append(nn.ReLU()),
            layers.append(nn.Dropout(p=dropout_p))

        # Last layer has no non linearity
        layers.append(nn.Linear(in_features=embedding_size, out_features=embedding_size))
        layers.append(nn.Dropout(p=dropout_p))
        self.layers = nn.Sequential(*layers)

    def forward(self, batch):
        return self.layers(batch)
