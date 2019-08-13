import argparse
import os
import json

from rltime.general.config import load_config
from rltime.general.type_registry import get_registered_type
from rltime.acting.create import create_actors_from_config
from rltime.general.loggers import DirectoryLogger
from rltime.general.utils import type_to_string, deep_dictionary_update
from rltime.env_wrappers.common import make_env_creator
import rltime.training
from rltime.general.config_template import validate_config

""" Entry point for training a policy

Usage options:
    - Trigger from command line ('python -u train.py' or
      'python -um rltime.train') with relevant arguments (see parse_args(), at
      least the json config file)
    - Call train_from_config with the configuration, either a python dictionary
      or path to json file
    - Call train() with the relevant configuration sections
"""


def train(log_dir, log_name, acting_config, env, model_config, training_config,
          policy_args={}, env_args={}):
    # Create the logger
    full_name = type_to_string(env)+"_" + \
        type_to_string(training_config["type"]) + \
        (f"_{log_name}" if log_name else "")
    logger = DirectoryLogger.create_new(
        base_path=log_dir, name=full_name)

    # Log the full config to the output directory
    logger.log_config({
        "acting": acting_config,
        "env": env,
        "env_args": env_args,
        "model": model_config,
        "policy_args": policy_args,
        "training": training_config
    })

    # Make the env-creation function
    env_creator = make_env_creator(env, **env_args)

    # Create the actors
    actors = create_actors_from_config(env_creator, acting_config)

    # Create the specified training class with the specified arguments
    assert("type" in training_config), \
        "No 'type' specified in the training config"
    training_cls = get_registered_type("trainers", training_config['type'])
    trainer = training_cls(
        logger=logger, actors=actors, model_config=model_config,
        policy_args=policy_args)

    # Train with the specified training arguments
    trainer.train(**training_config.get("args", {}))

    # Cleanup Actors
    actors.close()


def train_from_config(log_dir, log_name,
                      config, env=None, num_envs=None, conf_update=None):
    """Trains the given configuration

    args:
        log_dir: Base directory where to place training results (Each result
            gets a separate sub-directory)
        log_name: Custom text to add to the log directory (In addition to
            date/time and ENV name)
        config: The config to run, either a python dictionary or path to json
            config
        env: The environment to run. If None the ENV must be specified in the
            config. If not None it overrides the ENV from the config if exists
        num_envs: The amount of ENVs to run. If None the number must be
            configured in the config, if not None overrides the amount from the
            config
        conf_update: Update the config with this dictionary (deep update),
            usefull for changing a single/small amount of hyperparameters with
            a baseline config
    """
    if not isinstance(config, dict):
        config = load_config(config)
    validate_config(config)

    acting_config = config.get("acting", {})
    assert(acting_config or num_envs), \
        "Must either specify an 'acting' section in the config or specify "\
        "the 'num_envs' argument"
    if num_envs:
        acting_config['actor_envs'] = num_envs
    env = env or config.get("env", None)
    assert(env), "Must specify an 'env' either in the config or as an argument"

    model_config = config.get("model")
    assert(model_config), "Must specify a 'model' section in the config"

    train_config = config.get("training", None)
    assert(train_config), "Must specify a 'training' section in the config"

    if conf_update:
        config = dict(config)  # Avoid changing the passed config
        deep_dictionary_update(config, conf_update)

    train(
        log_dir, log_name, acting_config, env, model_config, train_config,
        config.get("policy_args", {}), config.get("env_args", {}))


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'config', type=str, help="The training configuration JSON file")
    parser.add_argument(
        '--num-envs', type=int,
        help="Optionally specify the number of envs to run (Overrides the "
             "config value if exists), required if not in the config file")
    parser.add_argument(
        '--env', type=str,
        help="Optionally specify the ENV to run (Overrides the config value "
             "if exists), required if not in the config file")
    parser.add_argument(
        '--log-dir', type=str, default=os.path.expanduser("~/rltime_logs"),
        help="Base directory where to place training results (Each result "
             "gets a separate sub-directory")
    parser.add_argument(
        '--log-name', type=str,
        help="Custom text to add to the log directory (In addition to "
             "date/time and ENV name")
    parser.add_argument(
        '--conf-update', type=str,
        help="Optional JSON dictionary string to deep-update the config with")

    return parser.parse_args()


def main():
    args = parse_args()

    conf_update = None if not args.conf_update \
        else json.loads(args.conf_update)
    train_from_config(
        log_dir=args.log_dir, log_name=args.log_name, config=args.config,
        env=args.env, num_envs=args.num_envs, conf_update=conf_update)

if __name__ == '__main__':
    main()
