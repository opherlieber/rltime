"""This file implements the logging functionality"""
import datetime
import os
import json
import numpy as np
import logging
import pickle
import datetime
from .config import load_config


class LoggerException(Exception):
    pass


class LoggerBase():
    def log_config(self, config):
        raise NotImplementedError

    def log_result(self, type, result, step):
        raise NotImplementedError

    def save_checkpoint(self, data, step):
        raise NotImplementedError

    def get_checkpoint(self):
        raise NotImplementedError


def json_custom_type(obj):
    """Custom json types for encoding

    Python objects converted to "@python('path_to_object')", numpy types
    converted to numbers
    """
    if callable(obj):
        return f"@python('{obj.__module__}.{obj.__name__}')"
    elif isinstance(obj, (np.int32, np.int64, np.float32, np.float64)):
        return obj.item()
    elif isinstance(obj, datetime.datetime):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        # This shouldn't usually happen, we won't store the data just indicate
        # it was an ndarray
        return "<ndarray>"
    else:
        print(obj)
        raise TypeError("Don't know how to json encode type: "+str(type(obj)))


def make_json(source, pretty):
    """ Converts input value to json """
    return json.dumps(
        source, sort_keys=True, indent=4 if pretty else None,
        default=json_custom_type)


class TensorboardLogger:
    def __init__(self, logdir):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=logdir)
        except (ModuleNotFoundError, ImportError) as e:
            logging.getLogger().warning(
                "Tensorboard not available, make sure you have installed: "
                "torch 1.1.0+, tensorboard 1.14.0+ and Pillow: "+str(e))
            self.writer = None

    def close(self):
        if not self.writer:
            return
        self.writer.close()

    def log_scalar(self, tag, value, global_step):
        if not self.writer:
            return
        self.writer.add_scalar(
            tag=tag, scalar_value=value, global_step=global_step)

    def log_dict(self, data, global_step, prefix=""):
        """Logs all scalars from a dictionary

        The tag will be the dictionary key and deep-keys will be combined
        using '.'
        """
        if not self.writer:
            return
        for key, val in data.items():
            if isinstance(val, dict):
                self.log_dict(val, global_step, prefix=prefix+key+".")
            elif isinstance(val, list):
                for i, val in enumerate(val):
                    # TODO: What if it's a list of dicts?
                    self.log_scalar(prefix + key+f"_{i}", val, global_step)
            else:
                self.log_scalar(prefix + key, val, global_step)


class DirectoryLogger():

    @staticmethod
    def _make_date_string():
        return datetime.datetime.today().strftime("%Y%m%d_%H%M%S")

    def _configure_logging(self):
        """Configures python logging to 'log.txt' file in the log directory,
        and to the console"""
        format = '%(asctime)s:%(levelname)s:%(process)d:%(message)s'
        logging.basicConfig(format=format,
                            filename=os.path.join(self.path, "log.txt"),
                            filemode='a',
                            level=logging.DEBUG)
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(format))
        logging.getLogger().addHandler(console)

    @classmethod
    def create_new(cls, base_path, name, **kwargs):
        """Initialize a new directory-logger in the given base-path.

        Will be logged to a new sub-directory in base_path with name including
        the data+time and given 'name' parameter
        """
        path = os.path.join(base_path, f"{cls._make_date_string()}_{name}")
        print("Logging to:", path)
        if os.path.exists(path):
            raise LoggerException(f"Log path exists, won't overwrite")
        os.makedirs(path)
        return cls(path, **kwargs)

    def __init__(self, path, use_logging=True, tensorboard=True):
        """Initializes a directory-logger in the given path.

        New loggers should init using 'DirectoryLogger.create_new()' to create
        the dated sub-directory
        """
        self.path = path

        # Init logging
        if use_logging:
            self._configure_logging()

        # Init TF logging
        if tensorboard:
            self.tensorboard_logger = TensorboardLogger(self.path)
        else:
            self.tensorboard_logger = None

    def log_config(self, config):
        """Logs the configuration to config.json, overwrites if exists (Should
        usually be called only once)"""
        text = make_json(config, True)
        with open(os.path.join(self.path, "config.json"), "w") as out:
            out.write(text)
        print("Config:", text)

    def get_config(self):
        return load_config(os.path.join(self.path, "config.json"))

    def log_result(self, type, result, step):
        """Logs a result dictionary as a new row in the respective output json
        file (<type>.json)"""
        with open(os.path.join(self.path, f"{type}.json"), "a") as out:
            out.write(make_json(result, False)+"\n")
        if self.tensorboard_logger:
            self.tensorboard_logger.log_dict(result, step)

        # Print result also to screen, using separate line per root-key
        if step is not None:
            print(f"\nStep {step}:")
        print("\n".join([
            f"  {key}: {make_json(item, False)}"
            for key, item in sorted(result.items())
        ]))

    def save_checkpoint(self, data, step):
        """Saves checkpoint data for the given step

        Currently we only support saving the last checkpoint and overwriting
        previous.
        """
        checkpoint = {"step": step, "data": data}
        pickle.dump(
            checkpoint, open(os.path.join(self.path, "checkpoint.p"), "wb"))

    def get_checkpoint(self):
        """Gets the last checkpoint data

        Returns: The global step for the checkpoint, and the data
        """
        cp = pickle.load(open(os.path.join(self.path, "checkpoint.p"), "rb"))
        return cp['step'], cp['data']
