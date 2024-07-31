import json
import logging
from dataclasses import asdict

__version__ = 0.001

logger = logging.getLogger()


class ConfigMethods:
    @staticmethod
    def save_config(config, path_filename):
        with open(path_filename, 'w') as f:
            if isinstance(config, dict):
                try:
                    json.dump(config, f)
                except TypeError:
                    logger.debug(f"Error: object not serializable")
            else:
                json.dump(asdict(config), f)
        logger.debug(f"Config saved to {path_filename}")

    @staticmethod
    def load_config(path_filename):
        with open(path_filename, 'r') as f:
            data = json.load(f)
        logger.debug(f"Config loaded from {path_filename}")
        return data

