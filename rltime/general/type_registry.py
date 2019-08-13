"""This module manages a registry of common types/classes/functions which can
be referenced by a string-name, split into groups"""
from .singleton import Singleton


class TypeRegistry(dict, metaclass=Singleton):
    """The singleton type registry, which maps common type strings to their
    respctive classes"""

    def __init__(self):
        """Initialize all the builtin types"""

        import rltime.training
        self['trainers'] = rltime.training.get_types()

        import rltime.models
        self['models'] = rltime.models.get_types()

        from rltime.models import modules
        self['modules'] = modules.get_types()

        import rltime.history
        self['history'] = rltime.history.get_types()

        import rltime.exploration
        self["exploration"] = rltime.exploration.get_types()


def get_registered_type(group, ref):
    registry = TypeRegistry()
    if not isinstance(ref, str):
        return ref
    elif group not in registry:
        raise TypeError(f"No types registered for group '{group}'")
    elif ref not in registry[group]:
        raise TypeError(
            f"No type '{ref}' registered in group '{group}', available types "
            f"in this group are: {list(registry[group].keys())}")
    return registry[group][ref]
