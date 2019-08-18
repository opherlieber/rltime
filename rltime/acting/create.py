from .actor import Actor


def create_actors_from_config(env_creator, config):
    """Creates the actor(s) from the given acting config"""

    # Arguments for each actor
    actor_args = dict(
        env_creator=env_creator,
        num_envs=config['actor_envs'],
        exploration_config=config.get("exploration", None),
        **config.get("extra_args", {})
    )

    # The base actor class to use, usually Actor by default but can choose a
    # different one in the config, for example AsyncActor
    actor_cls = config.get("actor_cls", Actor)

    if "pool" not in config:
        # No actor-pool, just create a single local vectorized actor
        return actor_cls(**actor_args)
    else:
        # Create a pool of actors
        pool_cls = config['pool'].get("type")
        pool_args = config['pool'].get("args", {})
        return pool_cls(
            **pool_args, actor_cls=actor_cls, actor_args=actor_args)
