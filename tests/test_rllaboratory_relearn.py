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
    logging.getLogger('LoadDbIndicators').setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    json_cfg = './save/BinanceEnvCash/PPO/exp-1309-164239/PPO_BinanceEnvCash_33000000_cfg.json'

    rllab = LabBase.load_agent(json_cfg)
    # rllab.test_agent(filename='best_model', verbose=1)
    # rllab.test_agent(filename='SAC_BinanceEnvCash_7000000_chkp_2700000_steps', verbose=1)
    """ Sell action reward """
    # rllab.backtesting_agent(filename='best_model', render_mode='human', n_tests=10, verbose=1)
    rllab.loaded_learn(
        filename=33_000_000,
        # filename='best_model',
        # render_mode='human',
        reset_num_timesteps=True,
        # total_timesteps=33_000_000,
        # env_kwargs_update={'gamma': 0.999},
        verbose=1)

    # rllab.backtesting_agent(filename=18_600_000, render_mode='human', n_tests=10, verbose=1)
    # rllab.evaluate_agent(0)
