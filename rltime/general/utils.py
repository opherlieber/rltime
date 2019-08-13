import numpy as np
import cloudpickle
import struct
import importlib
from .allowed_modules import _allowed_modules


def import_by_full_name(full_name):
    """Imports a type by it's full packaged name, ie package1.package2.type

    For security reasons only python modules listed in allowed_modules.py are
    allowed to be imported this way.
    """
    components = full_name.split(".")
    assert(components[0] in _allowed_modules), \
        f"Can't import {full_name} by string as it's not in the " \
        f"allowed_modules list. If it's your module and/or you know it's " \
        f"safe please add it to allowed_modules.py. Currently allowed " \
        f"modules are:{_allowed_modules}"

    module = importlib.import_module(".".join(components[:-1]))
    return getattr(module, components[-1])


def deep_stack(x, op=np.stack, args={}, base_type=np.ndarray):
    """Performs a deep stacking of a list/tuple of items.

    If it's a list of basic items (According to base_type) just stacks them
    If it's a list of lists or list of dictionaries it returns a
    list/dictionary accordingly with corresponding items stacked
    For example: [[a,b],[c,d],[e,f]] will return [stack(a,c,e),stack(b,d,f)]
                 [{'x':a1,'state':s1},{'x':a2,'state':s2}] will return:
                 {'x':stack(a1,a2),'state':stack(s1,s2)}
    Each item in the list should have same structure (Same length of each
    sublist, same keys of each dictionary etc.. and the corresponding items
    should be stackable recursively by these same rules)
    """
    if isinstance(x[0], base_type):
        return op(x, **args)
    elif isinstance(x[0], (list, tuple)):
        return type(x[0])([
            deep_stack([item[i] for item in x], op, args, base_type)
            for i in range(len(x[0]))])
    elif isinstance(x[0],dict):
        ret = {}
        for key in x[0]:
            ret[key] = deep_stack(
                [item[key] for item in x], op, args, base_type)
        return ret
    elif x[0] is None:
        return None
    else:
        return op([item for item in x], **args)


def deep_apply(x, f):
    """Performs a deep application of method f() on all objects in 'x'

    Recursively enters any list/tuple/dict.
    Non-destructive, returns a new object with the same structure
    """
    if isinstance(x, (list, tuple)):
        return type(x)([deep_apply(item, f) for item in x])
    elif isinstance(x, dict):
        return {key: deep_apply(val, f) for key, val in x.items()}
    elif x is None:
        return None
    else:
        return f(x)


def deep_dictionary_update(dest, source):
    """Performs a deep dictionary update of source to dest (in-place!)"""
    assert(isinstance(dest, dict))
    assert(isinstance(source, dict))
    for key, val in source.items():
        if isinstance(val, dict):
            if key not in dest:
                dest[key] = {}
            deep_dictionary_update(dest[key], source[key])
        else:
            dest[key] = val


def anneal_value(base_value, progress, anneal_mode, default_target=0.0):
    """Anneals a base_value across a period of [0,1]

    Args:
        base_value: The initial value at time=0
        progress: The current progress of the period in [0,1] (Greater than 1
            is treated as 1)
        anneal_mode: How to anneal the value. False/None means not to anneal,
            True means anneal to the <default_target> value, and a float value
            means anneal to this specific value
    """

    assert(progress >= 0)
    progress = min(progress, 1.0)
    if anneal_mode is False or anneal_mode is None:
        return base_value
    else:
        target = default_target if anneal_mode is True else float(anneal_mode)
        return base_value + (target - base_value)*progress


def type_to_string(tp):
    if hasattr(tp, "__name__"):
        return tp.__name__
    else:
        return str(tp)


def tcp_send_object(sock, obj, compress=False, pre_pickled=False):
    """Sends any python object over TCP using cloud-pickle with optional LZ4
    compression. Returns True if sent, False if connection closed"""
    data = cloudpickle.dumps(obj) if not pre_pickled else obj
    if compress:
        import lz4.frame
        data = lz4.frame.compress(data)

    # Send metadata to receiver: Size of the data buffer and whether
    # compression is enabled
    sock.send(struct.pack("II",len(data), 1 if compress else 0))
    sent = sock.send(data)
    if not sent:
        return False
    # Assumed either connection closed and sent=0, or the full thing was sent?
    # Maybe not if XFR stopped in the middle??
    assert(sent == len(data))
    return True


def tcp_recv_object(sock, chunk_size=1048576):
    """Receives a python object sent by 'tcp_send_object' over TCP. Returns the
    received object, or None if connection closed"""
    metadata = sock.recv(8)
    if len(metadata) != 8:  # Connection closed
        return None
    sz, compressed = struct.unpack("II", metadata)

    buffer = bytearray(sz)
    buffer_view = memoryview(buffer)
    offset = 0
    while sz > 0:
        amount_get = min(chunk_size, sz)
        amount_read = sock.recv_into(buffer_view[offset:], amount_get)
        if not amount_read:
            return None  # Connection closed

        sz -= amount_read
        offset += amount_read

    if compressed:
        import lz4.frame
        buffer = lz4.frame.decompress(buffer)
    return cloudpickle.loads(buffer)
