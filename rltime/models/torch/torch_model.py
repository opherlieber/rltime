import torch
import gym
import numpy as np
from .utils import make_tensor


class TorchModel(torch.nn.Module):
    """Base class for all pytorch models"""

    def __init__(self, observation_space):
        """Initializes the model with the given observation space

        Currently supported observation spaces are:
        - Box spaces
        - A tuple of box spaces, where the 1st one is the 'main' observation,
          and the rest contain additional 1D vectors of linear features for
          the model which are fed to one of the non-convolutional layers
          (Usually the RNN layer)
        """
        super().__init__()
        # When using multiple actors each with it's own CPU copy of the model,
        # we need to limit them to be single-threaded otherwise they slow each
        # other down. This should not effect training time if training is on
        # the GPU
        torch.set_num_threads(1)
        self._setup_inputs(observation_space)

    def _setup_inputs(self, obs_space):
        """Sets up the input sizes based on the given observation space"""
        assert(isinstance(obs_space, (gym.spaces.Box, gym.spaces.Tuple))), \
            "TorchModel currently only supports Box or Tuple as the " \
            "observation space"
        if isinstance(obs_space, gym.spaces.Box):
            # Basic case of just a single box space, no extra input features
            self.main_input_shape = obs_space.shape
            self.extra_input_shape = None
        else:
            # For now we just support the basic case where all spaces are Box
            # (i.e. no nested tuple spaces), and only the 1st space is the main
            # space, while the rest of the spaces are 1D extra feature vectors
            assert(np.all([
                    isinstance(space, gym.spaces.Box)
                    for space in obs_space.spaces])), \
                "TorchModel only supports tuples of boxes as the observation "\
                "space"
            # TODO: Support multiple main spaces and nested tuples??
            assert(np.all(
                [len(space.shape) == 1 for space in obs_space.spaces[1:]])), \
                "TorchModel currently only supports 1D box spaces for the " \
                " non-main observation space"
            self.main_input_shape = obs_space.spaces[0].shape
            self.extra_input_shape = (
                np.sum([space.shape for space in obs_space.spaces[1:]]),)

    def _get_inputs(self, inp):
        """Returns the the input separated into the 'main input' and the
        'extra inputs' (If applicable, i.e. if it's a tuple observation space)
        """
        if not isinstance(inp, tuple):
            return (inp, None)
        else:
            return (inp[0], torch.cat(inp[1:], dim=-1))

    def is_recurrent(self):
        raise NotImplementedError

    def set_layer_preprocessor(self, layer_index, preprocessor):
        raise NotImplementedError
