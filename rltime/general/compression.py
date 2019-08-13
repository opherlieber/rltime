import numpy as np

from .object_wrapper import ObjectWrapper
from .utils import deep_apply

try:
    import lz4.frame
except ModuleNotFoundError:
    pass


class CompressedNumpyArray():
    """Wrapper for compressing a numpy array"""
    def __init__(self, arr):
        self._dtype = arr.dtype
        self._shape = arr.shape
        self._data = lz4.frame.compress(arr)

    def get(self):
        data = lz4.frame.decompress(self._data, return_bytearray=True)
        return np.frombuffer(data, dtype=self._dtype).reshape(self._shape)

    def __array__(self):
        return self.get()


class CompressedStateWrapper(ObjectWrapper):
    """Wraps a policy input state while compressing the relevant data

    By default, only uint8 (e.g. image data) is compressed. Non-uint8 is
    usually not worthwhile to compress (For example RNN hidden states)
    """
    def __init__(self, state, only_uint8=True):
        def pack(x):
            if x.dtype == "uint8" or not only_uint8:
                return CompressedNumpyArray(x)
            else:
                return x
        self._data = deep_apply(state, pack)

    def get_object(self):
        def unpack(x):
            if isinstance(x, CompressedNumpyArray):
                return x.get()
            else:
                return x
        return deep_apply(self._data, unpack)
