import torch.nn as nn


class BaseModule(nn.Module):
    def get_state(self, initials):
        """Gets additional state information from the module which is needed
        for it's execution (For example RNN hidden states)"""
        return {}

    @staticmethod
    def is_recurrent():
        """Returns whether this is a recurrent module or not"""
        return False
