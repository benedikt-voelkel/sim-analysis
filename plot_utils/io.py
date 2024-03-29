"""
Methods to: manage input/output
"""

import os
import yaml
from plot_utils.logger import get_logger

def parse_yaml(filepath):
    """
    Parse a YAML file and return dictionary
    Args:
        filepath: Path to the YAML file to be parsed.
    """
    if not os.path.isfile(filepath):
        get_logger().critical("YAML file %s does not exist.", filepath)
    with open(filepath) as f:
        return yaml.safe_load(f)


def dump_yaml_from_dict(to_yaml, path):
    path = os.path.expanduser(path)
    with open(path, "w") as stream:
        yaml.safe_dump(to_yaml, stream, default_flow_style=False, allow_unicode=False)


def checkdir(path):
    """
    Check for existence of directory and create if not existing
    """
    if not os.path.exists(path):
        os.makedirs(path)

def print_dict(to_be_printed, indent=0, skip=None):
    for key, value in to_be_printed.items():
        if isinstance(skip, list) and key in skip:
            continue
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            print_dict(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))
