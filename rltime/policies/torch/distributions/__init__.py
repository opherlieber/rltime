import gym
from .categorical import Categorical
from .normal import Normal


def get_dist_layer_class(action_space):
    """Returns the distribution layer class matching the given action space"""
    if isinstance(action_space, gym.spaces.Discrete):
        return Categorical
    elif isinstance(action_space, gym.spaces.Box):
        return Normal
    else:
        raise ValueError(
            "Unsupported action_space, there is no distribution class "
            f"available for: {type(action_space)}")
