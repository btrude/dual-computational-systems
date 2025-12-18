import torch as t
import torch.nn as nn
from torch.nn.modules.utils import _pair as pair


class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, **kwargs):
        super(LocallyConnected2d, self).__init__()

        self.in_size = in_size
        self.kernel_size = pair(
            kwargs["kernel_size"] if "kernel_size" in kwargs else 3
        )
        self.padding = pair(kwargs["padding"] if "padding" in kwargs else 0)
        self.stride = pair(kwargs["stride"] if "stride" in kwargs else 1)
        self.num_classes = kwargs["num_classes"] if "num_classes" in kwargs else 10
        self.register_parameter("bias", None)

        output_size = self._calculate_out_size(self.in_size)

        self.weight = nn.Parameter(
            t.randn(
                1,
                out_channels,
                in_channels,
                output_size[0],
                output_size[1],
                self.kernel_size[0] * self.kernel_size[1],
            )
        )

    def forward(self, x):
        kh, kw = self.kernel_size
        dh, dw = self.stride
        padding_2dim = self.padding + self.padding

        x = nn.functional.pad(x, padding_2dim)
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        return (x.unsqueeze(1) * self.weight).sum([2, -1])

    def _calculate_out_size(self, in_size):
        out_width = in_size[0] + 2 * self.padding[0] - self.kernel_size[0]
        out_width = int(out_width / self.stride[0]) + 1
        out_height = in_size[1] + 2 * self.padding[1] - self.kernel_size[1]
        out_height = int(out_height / self.stride[1]) + 1
        return (out_width, out_height)
