import torch


class DistributionLayer(torch.nn.Module):
    """A distribution layer for action selection (e.g. for actor-critic)"""

    def __init__(self, action_space, input_size):
        """Initializes the distribution layer for the given action space and
        input_size (i.e. the output size of the model)
        """
        super().__init__()

    def forward(self, x):
        """Returns the relevant pytorch distribution output for input x,
        which can be used for action selection and distribution data
        """
        raise NotImplementedError
