from .config import ConfigException

# The expected train configuration structure (Not all fields are required, but
# no additional ones are allowed)
_template = {
    "acting": {
        "actor_envs": int,
        "actor_cls": object,
        "exploration": object,
        "extra_args": dict,
        "pool": {
            "type": object,
            "args": dict
        }
    },
    "env": object,
    "env_args": dict,
    "model": {
        "type": object,
        "args": dict
    },
    "policy_args": dict,
    "training": {
        "type": object,
        "args": dict
    }
}


def validate_config(config, template=_template, prefix=""):
    for key, val in config.items():
        if key not in template:
            raise ConfigException(f"Unexpected config key: '{prefix+key}'")
        template_val = template[key]
        expected_type = dict if isinstance(template_val, dict) \
            else template_val
        if not isinstance(val, expected_type):
            raise ConfigException(
                f"Expected type '{expected_type}'"
                f" for config key: '{prefix+key}'")
        if isinstance(template_val, dict):
            validate_config(val, template_val, prefix+key+"->")
