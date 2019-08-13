from .epsilon_greedy import EpsilonGreedyExplorationManager


def get_types():
    return {
        "epsilon_greedy": EpsilonGreedyExplorationManager,
    }
