"""This file configures allowed modules for textual types in json files

To avoid safety issues with python references in JSON files (E.g. a malicious
JSON config can be configured to import and execute fatal 'os' code),
only modules listed here will be allowed to be referenced from config files and
imported by their string name

In any case though please be wary with and review external config files before
using them, like you would with any external python code from unknown sources
"""

_allowed_modules = ["rltime", "gym", "retro", "gym_ple"]
