import numpy as np
from torch import nn

from dual_computational_systems.models.layers import DistributedLinear


class DistributedNetwork(nn.Module):
    def __init__(
            self,
            channels=1,
            input_dim=1024,
            base_channels=64,
            dropout=0.0,
            sparsity=0.975,
            categories_out=10,
        ):
        super().__init__()
        self.categories_out = categories_out

        hidden_dim1 = int((channels * (np.sqrt(input_dim) ** 4) * (base_channels / 4)) / (input_dim * channels))
        hidden_dim2 = 24 * base_channels

        self.features = nn.Sequential(*[
            DistributedLinear(channels * input_dim, hidden_dim1, sparsity=sparsity, bias=False),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
        ])
        self.classifier = nn.Sequential(*[
            nn.Linear(hidden_dim1, hidden_dim2, bias=False),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout1d(dropout) if dropout else nn.Identity(),
            nn.Linear(hidden_dim2, categories_out, bias=False),
        ])

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.features(x)
        return self.classifier(x)
