# Extended JSON Syntax
When loading JSON config files for training there is extended support for special directives to make it easier to organize common configurations.

## Referencing other JSON files
```json
"@json('<json_file>')"
```
This is useful for organizing common configurations such as environment parameters, models, modules etc., and referencing them from multiple configuration files, for example to use a shared model configuration:
```json
{
    "model": "@json('models/nature_cnn_lstm512_fc512.json')",
}
```
The file will be searched for in the standard OS search path. In addition files can be referenced relative to the 'rltime/configs' directory, to allow reuse of existing configurations present there without explicitly pointing to it.

## Referencing python types
```json
"@python('<full python package path with attribute>')"
```
This allows to reference any accessible python attribute, for example for custom training classes, modules, ENVs or ENV wrappers:
```json
{
    "env_args": {
        "wrappers": [
            {
                "type": "@python('rltime.env_wrappers.atari.wrap_atari')",
            }
        ]
    }
}
```
The argument should be the full module package path ending with the attribute defined in that module (For example a python class or method)

## Embedding dictionaries
Dictionaries can also be embedded by using "\*\*" as the key, which will perform the equivalent of the python "\*\*" operator at that key location. This can be useful for taking an existing configuration and then changing only certain sections:
```json
{
    "**": "@json('atari_ppo.json')",
    "env": "BreakoutNoFrameskip-v4",
}
```
Here the '\*\*' is replaced with the contents/keys of the referenced JSON file (Including shallow replace of already existing keys if there were any beforehand).

The "\*\*\*" key does a similar operation, except it does a 'deep update' of the given dictionary, leaving the existing structure and only replacing/adding values existing in the given dictionary. This can be useful for updating only certain hyperparameters which are not at the top level:
```json
{
    "**": "@json('atari_ppo.json')",
    "***": {
        "training": {
            "args": {
                "lr": 1e-4,
            }
        }
    }
}
```
Here we take the 'atari_ppo.json' config and only change the 'lr' value, leaving all other training args as they are.

## Comments
JSON doesn't natively support comments, however commenting can be done by starting the name of the key with an underscore. Any key beginning with an underscore will be ignored when the configuration is loaded:
```json
{
    "**": "@json('atari_ppo.json')",
    "_note": "Changing only the LR to 1e-4",
    "***": {
        "training": {
            "args": {
                "lr": 1e-4,
            }
        }
    }
}
```

## Output Config
When training, the parsed config is written to the log directory as 'config.json'. This config includes the final values after resolving all JSON references and embeddings (except for '@python' references which remain).