import torch
import torch.nn as nn
import numpy as np


def init_weight(weight, mode="xavier"):
    """Initializes the given weight"""
    if mode == "default":
        # Leave the default pytorch initialization
        pass
    elif mode == "xavier":
        # variance-scaling weight initialization using 'fan_in' mode
        # For linear layers (FC/LSTM weights) fan_in will be the input size
        # For conv2D it will be in_channels*kernel_w*kernel_h
        fan_in = np.prod(weight.shape[1:])
        stdev = (1./fan_in)**0.5
        nn.init.uniform_(weight, -stdev, stdev)
    else:
        assert(False), f"Invalid init_weight mode: '{mode}'"


def init_layer(layer):
    """Default layer weight initialization"""
    layer.bias.data.zero_()
    init_weight(layer.weight)


def tupilate(source, val):
    """Broadcasts val to be a tuple of same size as source"""
    if isinstance(val, tuple):
        assert(len(val) == len(source))
        return val
    else:
        return (val,)*len(source)


def conv_calc_out_size(sz, kernel_size, stride, padding=0, dilation=1):
    """Calculates the convolution output size based on input size and conv
    arguments"""
    if isinstance(sz, tuple):
        return tuple(
            [
                conv_calc_out_size(*val)
                for val in zip(
                    sz, tupilate(sz, kernel_size), tupilate(sz, stride),
                    tupilate(sz, padding), tupilate(sz, dilation))
            ]
        )
    return int((sz + 2*padding - dilation*(kernel_size-1)-1)/stride+1)


def conv2d(*args, **kwargs):
    """Makes a conv2d layer with given torch arguments and default init"""
    ret = nn.Conv2d(*args, **kwargs)
    init_layer(ret)
    return ret


def linear(*args, **kwargs):
    """Makes an FC layer with given torch arguments and default init"""
    ret = nn.Linear(*args, **kwargs)
    init_layer(ret)
    return ret


def set_lr(optimizer, lr):
    """Sets the LR for a pytorch optimizer"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def repeat_interleave(input, repeats, dim=0):
    """Repeats a tensor along the given axis using interleaved repeating
    (Instead of tiling)

    This is only available from torch 1.1 so we use a custom implementation if
    not available
    """
    if hasattr(input, "repeat_interleave"):
        return input.repeat_interleave(repeats, dim=dim)
    else:
        total_dims = len(input.shape)
        if dim < 0:
            dim += total_dims
            assert(dim >= 0)
        # Interleaved repeating is the same as a repeating/tiling on the next
        # dimension and merging back to the requested dimension
        res = input.unsqueeze(dim+1).repeat(
            (1,)*(dim+1) + (repeats,) + (1,)*(total_dims - dim - 1))
        return res.view(
            input.shape[:dim] + (
                input.shape[dim] * repeats,) + input.shape[dim+1:])


def make_tensor(x, device, non_blocking=False):
    """Takes an input and make a pytorch tensor out of it

    - If the input is a list/tuple or dictionary, performs the operation
      recursively, leaving the same list/tuple/dict struct
    - All numpy arrays are converted to float32 torch tensors except for uint8
      which is assumed to be image data and kept as uint8
    - The tensor is created on the requested device, if already a tensor only
      ensures its on the requested device
    """

    if isinstance(x, (list, tuple)):
        res = [make_tensor(val, device, non_blocking) for val in x]
        return type(x)(res)
    elif isinstance(x, dict):
        return {
            key: make_tensor(val, device, non_blocking)
            for key, val in x.items()
        }
    elif x is None:
        return None
    else:
        if not isinstance(x, torch.Tensor):
            if x.dtype != np.dtype('uint8') and x.dtype != np.dtype("float32"):
                assert(isinstance(x, np.ndarray))
                # Array case
                x = x.astype('float32')
            x = torch.from_numpy(x)
        return x.to(device, non_blocking=non_blocking)
