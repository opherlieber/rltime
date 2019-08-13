import torch
import gym

from .distribution_layer import DistributionLayer
from rltime.models.torch.utils import linear


class Categorical(DistributionLayer):
    """Categorical distribution for a discrete action space

    For choosing a single action out of all the possible 'n' actions
    """
    def __init__(self, action_space, input_size):
        super().__init__(action_space, input_size)
        assert(isinstance(action_space, gym.spaces.Discrete))
        self.logits_layer = linear(input_size, action_space.n)

    def forward(self, x):
        logits = self.logits_layer(x)
        return torch.distributions.categorical.Categorical(logits=logits)
