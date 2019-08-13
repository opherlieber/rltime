"""Contains backend specific code

Currently it's only pytorch so everything is just thrown in here but if in the
future TF2.0 is added, this will move to a backend directory and import from
the respective backend
"""
import torch
import numpy as np
from .utils import deep_stack, deep_apply
from rltime.models.torch.utils import make_tensor
import itertools
from collections import deque
import pickle

_backends_config = {
    "torch": {
        "channel_axis": 0
    }
}


def get_backend_name():
    return "torch"


def get_backend_config():
    return _backends_config[get_backend_name()]


def get_channel_axis():
    return get_backend_config()["channel_axis"]


class TensorBufferPool():
    """Implements a tensor buffer pool with the requested size and dtype

    This allows frequent operations such as stacking to avoid memory
    allocations.
    Using the 'pinned' option can also allow the stacked tensors to be loaded
    to the GPU non-blocking (Though need to be careful of corruptions to the
    shared buffer in this case)
    """
    def __init__(self, size, dtype, allow_wrap=False, pinned=False):
        self._size = size
        self._arr = torch.zeros(size, dtype=dtype, pin_memory=pinned)
        self._next = 0
        self._allow_wrap = allow_wrap

    def get(self, shape):
        sz = np.prod(shape)
        if self._next + sz > self._size:
            if not self._allow_wrap:
                raise RuntimeError("TensorBufferPool Overflow")
            self.reset()
        res = self._arr[self._next:self._next+sz].view(shape)
        self._next += sz
        return res

    def reset(self):
        self._next = 0


class StateStore():
    """Implements a state-store/loader for pytorch

    This handles loading/stacking of state data for training to the requested
    device.

    In particular it allows history buffers to stack and load data
    to the GPU in a separate process using shared CUDA tensors (Not supported
    on windows)

    This uses a resident 'buffer pool' as the destination for stacking
    state data, to improve stacking performance.
    This costs 0.5GB of RAM. In case or large observations/batches you may get
    a "TensorBufferPool Overflow" error and need to increase the size.

    It's possible to extened this to use the buffer pool on 'pinned' memory
    and load non-blocking to the GPU, but this introduces too many
    synchronization requirements to avoid corruption, and doesn't help enough
    to justify it.
    """
    def __init__(self, device, pool_size=256*1024*1024):
        self._device = device
        self._pool_size = pool_size

    def init_for_use(self):
        self._history = deque(maxlen=3)
        # Use the stacking pool only if it's a temp buffer (i.e. if
        # it's immediately loaded to GPU after stacking)
        if str(self._device) == "cpu":
            self._use_pool = False
        else:
            self._use_pool = True
            # This is for benchmark scripts to avoid getting initial overheads
            torch.cuda.synchronize()
            # Setup pinned buffer pools for uint8 and float data which are the
            # common types for state data (Other types will do regular
            # stacking)
            self.float32_pool = TensorBufferPool(
                self._pool_size // 4, dtype=torch.float32)
            self.uint8_pool = TensorBufferPool(
                self._pool_size, dtype=torch.uint8)

    def store(self, state):
        # TODO: Possibly add support for storing state data on the GPU
        # (Though this doesn't seem to help much and is not realistic for 1M+
        # replay buffers). For now it works best to leave state-data as-is
        # (i.e. numpy) until it needs to be stacked for training
        return state

    def _stack_pool(self, arrs):
        """Stacks the list of arrays using the buffer pool as destination"""
        dtype = arrs[0].dtype if \
            isinstance(arrs[0], (np.ndarray, torch.Tensor)) else None
        if dtype == "uint8":
            pool = self.uint8_pool
        elif dtype == "float32":
            pool = self.float32_pool
        else:
            # Unsupported type just do regular stacking with memory allocation
            return np.stack(arrs)
        # Calc the final stacked shape and get a buffer of that size from the
        # buffer pool
        shape = (len(arrs),) + arrs[0].shape
        dest = pool.get(shape)
        # stack the numpy arrays to the destination (dest.numpy() should
        # be an immediate view on the tensor ram)
        if isinstance(arrs[0], torch.Tensor):
            torch.stack(arrs, out=dest)
        else:
            np.stack(arrs, out=dest.numpy())

        return dest

    def stack(self, states):
        """Stacks a list of states using deep stacking"""
        # Reset the pools for stacking
        if self._use_pool:
            self.uint8_pool.reset()
            self.float32_pool.reset()

        # It's faster to stack first to system memory and then load a big chunk
        # to GPU, then to load many small chunks and then stack on the GPU
        states = deep_stack(
            states, op=self._stack_pool if self._use_pool else np.stack)
        res = make_tensor(states, device=self._device, non_blocking=False)

        # Not sure why this is needed but the process sometimes gets stuck
        # without this when doing shared multi-processing CUDA tensors.
        # We save the last 3 generated buffers
        self._history.append(res)
        return res


class SampleBatch():
    pass


class SharedSampleBatch(SampleBatch):
    """Wrapper for batch of actor samples using shared memory

    This wraps a list of samples received from actors, moving state-data
    to shared-memory using pytorch tensor memory-sharing. This allows the
    batch to quickly be sent around using multiprocessing queues (other than
    the one-time overhead of creating this wrapper).

    This helps only if the samples are received on a sub-process, and even more
    if the history buffer is also in a sub-process

    This usually requires increasing the shared file descriptor limit on linux:
        ulimit -n 65536

    The batch should eventually be unpacked using unpack() once the state
    data needs to be used (i.e. in the history buffer). Until then all data
    except state-data can be used as usual.
    """
    def __init__(self, samples):
        states = [sample['next_state'] for sample in samples]
        self._states = deep_stack(states)
        # Batch all state-data to a single shared tensor
        self._states = deep_apply(
            self._states, lambda x: torch.from_numpy(x).share_memory_())
        # Leave the rest of the fields as-is without state data
        self._len = len(samples)
        self._samples = [{**sample, "next_state": None} for sample in samples]
        self._packed = True

    def __iter__(self):
        return self._samples.__iter__()

    def __len__(self):
        return self._len

    def unpack(self):
        if not self._packed:
            return

        # TODO: the copy() reduces performance but removes the shared memory
        # descriptor limit requirement
        # np_states = deep_apply(self._states, lambda x:x.numpy().copy())
        np_states = deep_apply(self._states, lambda x: x.numpy())

        for i, sample in enumerate(self._samples):
            sample["next_state"] = deep_apply(np_states, lambda x: x[i])
        self._packed = False


class SharedSampleList():
    """Implements a list of SharedSampleBatch's which can be appended/extended

    The resulting object can be used as if it was a single larger batch list,
    even though it contains multiple batches under the hood
    """
    def __init__(self):
        self._batches = []
        self._len = 0

    def unpack(self):
        for batch in self._batches:
            batch.unpack()

    def __iter__(self):
        return itertools.chain(*[batch.__iter__() for batch in self._batches])

    def append(self, batch):
        assert(isinstance(batch, SampleBatch))
        self._batches.append(batch)
        self._len += len(batch)

    def extend(self, batch_list):
        assert(isinstance(batch_list, SharedSampleList))
        self._batches.extend(batch_list._batches)
        self._len += batch_list._len

    def __len__(self):
        return self._len
