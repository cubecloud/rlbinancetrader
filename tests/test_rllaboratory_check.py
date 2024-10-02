import logging
import datetime
import numpy as np
from dbbinance.fetcher.constants import Constants
from rllab.rllaboratory import LabBase
from binanceenv.bienv import BinanceEnvCash
from stable_baselines3 import A2C, PPO, DDPG, DQN, TD3, SAC
from multiprocessing import freeze_support

__version__ = 0.0019

logger = logging.getLogger()

if __name__ == '__main__':
    freeze_support()

    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('test_rllaboratory.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger('numba').setLevel(logging.INFO)
    logging.getLogger('gymnasium').setLevel(logging.INFO)
    logging.getLogger('LoadDbIndicators').setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-0110-000853/MaskablePPO_BinanceEnvCash_60000000_cfg.json'

    rllab = LabBase.load_agent(json_cfg)
    # rllab.test_agent(filename='best_model', verbose=1)
    # rllab.test_agent(filename=750_000, n_tests=15, verbose=1)
    """ Sell action reward """
    # rllab.backtesting_agent(filename='best_model', render_mode='human', n_tests=10, verbose=1, seed=443,
    #                         use_period='test')
    rllab.backtesting_agent(filename=59_000_000, render_mode='human', n_tests=10, verbose=1, seed=1443,
                            use_period='test')
    # rllab.evaluate_agent(0)
