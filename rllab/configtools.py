import json
import numpy as np
from json_numpy import default, object_hook
import logging
from dataclasses import asdict

__version__ = 0.012

logger = logging.getLogger()


class ConfigMethods:
    @staticmethod
    def save_config(config, path_filename):
        if isinstance(config, dict):
            with open(path_filename, 'w') as f:
                json.dump(config, f, default=default)
        else:
            with open(path_filename, 'w') as f:
                json.dump(asdict(config), f)
        logger.debug(f"Config saved to {path_filename}")

    @staticmethod
    def load_config(path_filename):
        with open(path_filename, 'r') as f:
            data = json.load(f, object_hook=object_hook)
        logger.debug(f"Config loaded from {path_filename}")
        return data

