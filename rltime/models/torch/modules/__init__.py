from .fc import FC
from .cnn import CNN
from .lstm import LSTM


def get_types():
    return {
        "fc": FC,
        "cnn": CNN,
        "lstm": LSTM
    }
