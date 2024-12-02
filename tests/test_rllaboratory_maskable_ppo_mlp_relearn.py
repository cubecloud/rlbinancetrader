import logging
import datetime
import numpy as np
from dbbinance.fetcher.constants import Constants
from rllab.rllaboratory import LabBase
from binanceenv.bienv import BinanceEnvCash
from stable_baselines3 import A2C, PPO, DDPG, DQN, TD3, SAC
from multiprocessing import freeze_support
import warnings

__version__ = 0.0028

logger = logging.getLogger()

if __name__ == '__main__':
    freeze_support()

    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('test_rllab_ppo.log')
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

    json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-3011-174850/MaskablePPO_BinanceEnvCash_900000000_cfg.json'

    rllab = LabBase.load_agent(json_cfg)
    # rllab.test_agent(filename='best_model', verbose=1)
    # rllab.test_agent(filename='SAC_BinanceEnvCash_7000000_chkp_2700000_steps', verbose=1)
    """ Sell action reward """
    # rllab.backtesting_agent(filename='best_model', render_mode='human', n_tests=10, verbose=1)
    _start_datetime = '2023-07-20 01:00:00'
    _end_datetime = '2024-11-01 01:00:00'
    _timeframe = '15m'
    _discretization = '15m'
    total_timesteps = 900_000_000

    agents_n_env = 3780
    learning_start = (3780 * 300 * 3)

    data_processor_kwargs = dict(start_datetime=_start_datetime,
                                 end_datetime=_end_datetime,
                                 timeframe=_timeframe,
                                 discretization=_discretization,
                                 symbol_pair='BTCUSDT',
                                 market='spot',
                                 minimum_train_size=0.0267,
                                 maximum_train_size=0.031,
                                 minimum_test_size=0.168,
                                 maximum_test_size=0.188,
                                 test_size=0.13,
                                 verbose=0,
                                 indicators_sign=True
                                 )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        rllab.loaded_learn(
            # filename=504_630_000,
            filename='best_model',
            reset_num_timesteps=True,
            total_timesteps=total_timesteps,
            env_kwargs_update={
                'data_processor_kwargs': data_processor_kwargs,
                'stable_cache_data_n': 3780 * 2,
                'reuse_data_prob': 1.0,
                'verbose': 0,
                'render_mode': 'human',
            },

            agent_kwargs_update={
                'n_steps': 300,
                'batch_size': 31500,
                'n_epochs': 10,
                'stats_window_size': 25,
                'ent_coef': 0.01,
                'gamma': 0.92,
                'learning_rate': {'CoSheduller': dict(warmup=learning_start,
                                                      learning_rate=4.5e-6,
                                                      min_learning_rate=3.5e-6,
                                                      total_epochs=total_timesteps,
                                                      epsilon=1)
                                  },
                'seed': 511,
            },
            env_wrapper='labsubproc',
            n_envs=agents_n_env,
            eval_freq=300,
            verbose=0,
        )

    # rllab.backtesting_agent(filename=18_600_000, render_mode='human', n_tests=10, verbose=1)
    # rllab.evaluate_agent(0)
