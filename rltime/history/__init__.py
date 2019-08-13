from .online_history import OnlineHistoryBuffer
from .replay_history import ReplayHistoryBuffer
from .prioritized_replay_history import PrioritizedReplayHistoryBuffer


def get_types():
    return {
        "online": OnlineHistoryBuffer,
        "replay": ReplayHistoryBuffer,
        "prioritized_replay": PrioritizedReplayHistoryBuffer
    }