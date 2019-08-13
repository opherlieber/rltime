from .torch.sequential import SequentialModel

from .torch import modules


def get_types():
    return {
        "sequential": SequentialModel
    }
