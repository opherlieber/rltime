from .a2c import A2C
from .ppo import PPO
from .dqn import DQN
from .iqn import IQN
from .dist_dqn import DistDQN


def get_types():
    return {
        "a2c": A2C,
        "ppo": PPO,
        "dqn": DQN,
        "dist_dqn": DistDQN,
        "iqn": IQN,
    }
