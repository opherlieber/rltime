import torch
import torch.nn as nn
import io
import numpy as np
import os

from ..policy import Policy
from rltime.models.torch.utils import make_tensor
from rltime.general.utils import deep_stack, deep_apply
from rltime.general.backend import StateStore


def torch_get_raw(source):
    """Returns a raw/binary serialization of a pytorch object(Basically does
    torch.save() on it but to a RAM buffer instead of a file)
    """
    f = io.BytesIO()
    torch.save(source, f)
    data = f.getvalue()
    f.close()
    return data


def torch_load_cpu(data):
    """Loads a serialized pytorch object from torch_get_raw() with CPU
    placement"""
    f = io.BytesIO(data)
    ret = torch.load(f, map_location=lambda storage, loc: storage)
    f.close()
    torch.set_num_threads(1)
    return ret


class TorchPolicy(nn.Module, Policy):
    """Base class for pytorch policies"""
    def __init__(self, model_config, observation_space):
        super().__init__()
        # Create the model
        self.model = self._create_model_from_config(
            model_config, observation_space)
        self.is_cuda = self.model.is_cuda

    @classmethod
    def create(cls, *args, cuda="auto", **kwargs):
        """Creates a torch policy with the given args+kwargs

        Args:
            cuda: configures the cuda-placement of the policy. 'auto' means to
                use the default cuda device if available otherwise the cpu.
                Can also be True/False or a specific cuda device for example
                "cuda:1"
        """
        policy = cls(*args, **kwargs)
        if cuda == "auto":
            cuda = torch.cuda.is_available()
        if cuda:
            policy = policy.to(torch.device("cuda" if cuda is True else cuda))

        return policy

    def copy_from(self, source, factor=1.0):
        """Copys parameters from the given source policy to this policy

        Optionally use a factor<1.0 for a partial/gradual copy as done
        in some older RL papers
        """
        for source, dest in zip(source.parameters(), self.parameters()):
            dest.data.copy_(source.data * factor + dest.data * (1.0 - factor))

    def get_grad_norm(self):
        """Calculates the global grad-norm (L2-variant) of the policy/model
        parameters"""
        total_norm = 0
        for p in self.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def is_recurrent(self):
        return self.model.is_recurrent()

    def make_input_state(self, inp, initials):
        return self.model.make_input_state(inp, initials)

    def make_tensor(self, x, non_blocking=False):
        """Makes a tensor out of the given input, on the policies device"""
        return make_tensor(x, device="cpu" if not self.is_cuda() else "cuda",
                           non_blocking=non_blocking)

    def get_creator(self, cuda=False):
        # Returns a lambda that generates a copy of this policy
        # NOTE: cuda option not supported ATM
        data = torch_get_raw(self)
        return lambda: torch_load_cpu(data)

    def get_state(self):
        return torch_get_raw(self.state_dict())

    def load_state(self, state):
        self.load_state_dict(torch_load_cpu(state))

    def get_state_store(self, is_async_storage):
        # We can use CUDA for the state store unless we are on windows and
        # the storage is in a separate process (Windows doesn't support
        # MP shared CUDA tensors)
        if self.is_cuda() and (not is_async_storage or os.name != 'nt'):
            device = "cuda"
        else:
            device = "cpu"
        return StateStore(device)
