from .base import BaseModule
import torch.nn as nn
import numpy as np
from ..utils import linear


class FC(BaseModule):
    """A fully connected layer"""

    def __init__(self, inp_shape, fc_size, fc_count=1, batch_norm=False,
                 activation="relu"):
        super().__init__()

        self.layers = nn.ModuleList()
        sz = np.prod(inp_shape)
        self.flat_size = sz

        # Setup Dense/Linear Layers
        for _ in range(fc_count):
            slayer = nn.ModuleList([linear(sz, fc_size)])
            sz = fc_size
            if batch_norm:
                slayer.append(nn.BatchNorm1d(sz))

            self.layers.append(slayer)
        self.out_shape = (sz,)
        self.activation = getattr(nn.functional, activation)

    def forward(self, x, **kwargs):
        x = x.view(-1, self.flat_size)

        for layer in self.layers:
            for slayer in layer:
                x = slayer(x)
            x = self.activation(x)
        return x

    def is_cuda(self):
        return self.layers[0][0].weight.is_cuda \
            if hasattr(self.layers[0][0], "weight") \
            else self.layers[0][0].module.weight.is_cuda
