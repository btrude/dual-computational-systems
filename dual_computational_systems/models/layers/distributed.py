import torch as t
from torch import nn


class DistributedLinear(nn.Linear):
    def __init__(self, in_features, out_features, sparsity=0.975, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)

        n_units = in_features * out_features
        zero_indices = t.randperm(int(n_units * sparsity))

        self.register_buffer("mask", t.ones(self.weight.size()))
        self.weight.data.flatten()[zero_indices] = 0.0
        self.mask.data.flatten()[zero_indices] = 0.0

    def forward(self, input):
        masked_weight = self.weight * self.mask
        return nn.functional.linear(input, masked_weight, self.bias)
