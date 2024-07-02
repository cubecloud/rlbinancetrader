import logging
from rllab.configtools import ConfigMethods
from dataclasses import dataclass, field, asdict


def create_dirs() -> dict:
    dirs = dict(training=None,
                evaluation=None,
                )
    return dirs


@dataclass(init=True)
class LABConfig:
    ID_NAME: str = str()
    EXPERIMENT_PATH: dict = str()
    EXP_ID: str = str()
    DIRS: dict = field(default_factory=create_dirs, init=False)


__version__ = 0.001

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('test_configtools.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

lb_cfg = LABConfig()
# lb_cfg.ID_NAME = 'NEW_ID'

# setattr(lb_cfg, "TEST_ID", "test_ID")
print(asdict(lb_cfg))

cfg = ConfigMethods(lb_cfg)

cfg.save_config(f'test_config.json')
cfg.load_config(f'test_config.json')
# setattr(lb_cfg, "TEST_ID", "test_ID")
print(asdict(lb_cfg))
