"""Functions for handling loading JSON configuration files, including
referencing/nesting other json-files/python-types"""
import rltime
import json
import os
import re
import importlib
from gym.wrappers.monitoring.video_recorder import ImageEncoder

from .utils import import_by_full_name, deep_dictionary_update


class ConfigException(Exception):
    pass


def resolve_file_path(file_path):
    """Resolves a json-file path for loading

    If the file doesn't exist we check also relative to the common configs
    directory, this allows referencing common config json files from config
    files which are outside the main package.
    """
    if not os.path.isfile(file_path):
        # Allow loading config files relative to rltime/configs directory
        base_path = os.path.dirname(rltime.__file__)
        rel_file_path = os.path.join(base_path, "configs", file_path)
        if os.path.isfile(rel_file_path):
            return rel_file_path
    return file_path


def load_config(file_path):
    file_path = resolve_file_path(file_path)
    with open(file_path, "r") as in_data:
        return json.load(
            in_data,
            object_pairs_hook=lambda obj:
                json_parse_pairs(obj, os.path.dirname(file_path)))


def load_nested_json(json_file, nested_key=None):
    res = load_config(json_file)
    if nested_key is not None:
        try:
            for key in nested_key.split("->"):
                res = res[key]
        except KeyError:
            raise ConfigException(
                f"Could not find nested key '{nested_key}' "
                f"in '{json_file}'")
    return res


def parse_ref(ref):
    """Parses a reference loader name and arguments

    For example "@json('aaa','bbb')" will return: "json", ["aaa","bbb"]
    """
    if ref[:1] != "@":
        raise ConfigException("Invalid config ref: "+str(ref))
    try:
        parts = re.split(r"@|\s*\(\s*'|'\s*,\s*'|'\s*\)\s*", ref)
        return parts[1], parts[2:-1]
    except ValueError:
        raise ConfigException(f"Failed to parse reference '{ref}'")


def load_ref(ref, base_path):
    try:
        org_path = os.getcwd()
        if base_path:
            os.chdir(base_path)

        ref_name, ref_args = parse_ref(ref)
        ref_loaders = {
            "json": load_nested_json,
            "python": import_by_full_name
        }
        if ref_name not in ref_loaders:
            raise ConfigException(
                f"Unknnown reference type: {ref_name}, "
                f"available ones are: {list(ref_loaders.keys())}")
        return ref_loaders[ref_name](*ref_args)
    except FileNotFoundError:
        raise ConfigException(
            f"Could not load referenced file '{ref}'. "
            f"Make sure it's in your path or reachable from '{base_path}'")
    finally:
        os.chdir(org_path)


def load_refs_if_needed(val, base_path):
    if isinstance(val, str) and val[:1] == "@":
        return load_ref(val, base_path)
    elif isinstance(val, list):
        return [load_refs_if_needed(item, base_path) for item in val]
    return val


def json_parse_pairs(obj, base_path):
    res = {}
    for key, val in obj:
        if key[:1] == "_":
            # Treat all keys starting with '_' as comments
            continue
        val = load_refs_if_needed(val, base_path)

        if key == "**":
            # Inline dictionary embed
            res = {**res, **val}
        elif key == "***":
            # Deep inline dictionary embed
            deep_dictionary_update(res, val)
        else:
            res[key] = val

    return res
