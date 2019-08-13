import torch

from .base import BaseModule
from ..utils import conv_calc_out_size, conv2d

import logging


class CNN(BaseModule):
    def __init__(self, inp_shape, layers, scale=1.0/255.0):
        """Initializes the CNN module

        Args:
            inp_shape: The input shape to the conv-block in the format:
                (channels, height, width)
            layers: Layer configuration for the CNN module. A list of
                dictionaries where each dictionary contains:
                filters/kernel/stride
            scale: Whether to scale the input and by how much. Typically the
                input is a visual input of type uint8 in range (0,255) in which
                case we scale it by 1/255 and convert to float() before
                processing it (It's faster to this on the GPU, and also reduces
                GPU data transfer sizes by 4x)
        """
        super().__init__()
        self.scale = scale
        self.layers = torch.nn.ModuleList()
        out_filters = inp_shape[0]
        sz = inp_shape[1:]
        assert(len(sz) == 2)
        for layer in layers:
            # Create the conv layer
            filters, kernel, stride = \
                layer['filters'], layer['kernel'], layer['stride']
            self.layers += [conv2d(out_filters, filters, kernel, stride)]
            # Calculate the output dimensions of the layer
            sz = conv_calc_out_size(sz, kernel, stride)
            out_filters = filters

        # The final output shape of this conv module (channels,w,h)
        self.out_shape = (out_filters, sz[0], sz[1])

    def forward(self, x, **kwargs):
        if self.scale:
            x = x.float() * self.scale

        for layer in self.layers:
            x = layer(x)
            x = torch.nn.functional.relu(x)
        return x

    def is_cuda(self):
        return self.layers[0].weight.is_cuda
