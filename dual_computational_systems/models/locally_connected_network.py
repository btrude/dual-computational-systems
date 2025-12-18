from torch import nn
from torch.nn.modules.utils import _pair as pair

from dual_computational_systems.models.layers import LocallyConnected2d


class LocallyConnectedNetwork(nn.Module):
    def __init__(
        self,
        channels,
        base_channels,
        img_size=32,
        kernel_size=5,
        stride=2,
        padding=2,
        dropout=0.0,
        categories_out=10,
    ):
        super().__init__()
        self.categories_out = categories_out

        self.img_size = pair(img_size) if isinstance(img_size, int) else img_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.categories_out = categories_out

        self.features = nn.Sequential(
            LocallyConnected2d(
                channels,
                base_channels,
                self.img_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
        )

        out_width = self.img_size[0] + 2 * self.padding - self.kernel_size
        out_width = int(out_width / self.stride) + 1
        out_height = self.img_size[1] + 2 * self.padding - self.kernel_size
        out_height = int(out_height / self.stride) + 1
        fc_input_dim = base_channels * out_width * out_height

        hidden_dim2 = 24 * base_channels
        self.classifier = nn.Sequential(*[
            nn.Linear(fc_input_dim, hidden_dim2, bias=False),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout1d(dropout) if dropout else nn.Identity(),
            nn.Linear(hidden_dim2, categories_out, bias=False),
        ])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)
