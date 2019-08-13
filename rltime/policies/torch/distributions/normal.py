import torch
import gym
import numpy as np

from .distribution_layer import DistributionLayer
from rltime.models.torch.utils import linear


class Normal(DistributionLayer):
    """A normal distribution for multiple continuous action values.

    Note there is no bounding of the action high/low values here
    """

    def __init__(self, action_space, input_size):
        super().__init__(action_space, input_size)
        assert(isinstance(action_space, gym.spaces.Box))
        self.actions_shape = action_space.shape

        flat_size = np.prod(action_space.shape)

        self.mean_layer = linear(input_size, flat_size)
        self.logstd = torch.nn.Parameter(torch.zeros(flat_size))

    def forward(self, x):
        batch_size = x.shape[0]
        mean = self.mean_layer(x)
        std = self.logstd.exp().expand_as(mean)

        return torch.distributions.normal.Normal(
            mean.view(batch_size, *self.actions_shape),
            std.view(batch_size, *self.actions_shape)
        )
