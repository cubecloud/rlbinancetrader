import sys

sys.path.insert(0, '/home/cubecloud/Python/projects/rlbinancetrader')

import logging
import datetime
import numpy as np
from dbbinance.fetcher.constants import Constants
from rllab.rllaboratory import LabBase
from binanceenv.bienv import BinanceEnvCash
from stable_baselines3 import A2C, PPO, DDPG, DQN, TD3, SAC
from multiprocessing import freeze_support, get_logger
import warnings

__version__ = 0.0038

logger = get_logger()

if __name__ == '__main__':
    freeze_support()

    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('test_rllab_mask_ppo_relearn.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger('numba').setLevel(logging.INFO)
    logging.getLogger('LoadDbIndicators').setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-0912-140611/MaskablePPO_BinanceEnvCash_900000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-1212-030411/MaskablePPO_BinanceEnvCash_900000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-1612-220439/MaskablePPO_BinanceEnvCash_1200000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-2512-004141/MaskablePPO_BinanceEnvCash_1200000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-2912-004826/MaskablePPO_BinanceEnvCash_1200000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-3012-164947/MaskablePPO_BinanceEnvCash_1200000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-3112-205905/MaskablePPO_BinanceEnvCash_1200000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-0201-015059/MaskablePPO_BinanceEnvCash_1200000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-0801-194931/MaskablePPO_BinanceEnvCash_1200000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-1212-030411/MaskablePPO_BinanceEnvCash_900000000_cfg.json'
    json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-1101-120227/MaskablePPO_BinanceEnvCash_1200000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-1501-010030/MaskablePPO_BinanceEnvCash_1500000000_cfg.json'

    rllab = LabBase.load_agent(json_cfg)
    # rllab.test_agent(filename='best_model', verbose=1)
    # rllab.test_agent(filename='SAC_BinanceEnvCash_7000000_chkp_2700000_steps', verbose=1)
    """ Sell action reward """
    # rllab.backtesting_agent(filename='best_model', render_mode='human', n_tests=10, verbose=1)
    # _start_datetime = '2022-12-31 23:00:00'
    # _start_datetime = '2023-01-20 01:00:00'
    # _start_datetime = '2023-07-20 01:00:00'
    # _start_datetime = '2023-08-20 01:00:00'
    # _end_datetime = '2024-07-30 01:00:00'
    # _end_datetime = '2024-08-30 01:00:00'
    # _end_datetime = '2024-11-20 01:00:00'

    # _end_datetime = '2024-12-10 01:00:00'
    # _end_datetime = '2024-12-10 01:00:00'

    _start_datetime = '2023-07-20 01:00:00'
    _end_datetime = '2024-07-30 01:00:00'

    # _start_datetime = '2023-03-20 01:00:00'
    # _end_datetime = '2024-10-30 01:00:00'

    _timeframe = '15m'
    _discretization = '15m'
    total_timesteps = 3_000_000_000

    agents_n_env = 5000
    n_steps = 300
    warmup_timesteps = (agents_n_env * n_steps) * 300 * 3

    # learning_start = (agents_n_env * n_steps * 3)

    # data_processor_kwargs = dict(start_datetime=_start_datetime,
    #                              end_datetime=_end_datetime,
    #                              timeframe=_timeframe,
    #                              discretization=_discretization,
    #                              symbol_pair='BTCUSDT',
    #                              market='spot',
    #                              minimum_train_size=0.0267,
    #                              maximum_train_size=0.031,
    #                              minimum_test_size=0.168,
    #                              maximum_test_size=0.185,
    #                              test_size=0.13,
    #                              verbose=0,
    #                              indicators_sign=True
    #                              )
    data_processor_kwargs = dict(start_datetime=_start_datetime,
                                 end_datetime=_end_datetime,
                                 timeframe=_timeframe,
                                 discretization=_discretization,
                                 symbol_pair='BTCUSDT',
                                 market='spot',
                                 minimum_train_size=940,
                                 maximum_train_size=990,
                                 minimum_test_size=940,
                                 maximum_test_size=990,
                                 test_size=0.13,
                                 verbose=1,
                                 indicators_sign=True
                                 )
    # data_processor_kwargs = dict(start_datetime=_start_datetime,
    #                              end_datetime=_end_datetime,
    #                              timeframe=_timeframe,
    #                              discretization=_discretization,
    #                              symbol_pair='BTCUSDT',
    #                              market='spot',
    #                              minimum_train_size=550,
    #                              maximum_train_size=650,
    #                              minimum_test_size=550,
    #                              maximum_test_size=650,
    #                              test_size=0.1,
    #                              verbose=1,
    #                              indicators_sign=True
    #                              )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        rllab.loaded_learn(
            # filename=101_304_000,
            filename='best_model',
            reset_num_timesteps=True,
            total_timesteps=total_timesteps,
            env_kwargs_update={
                'data_processor_kwargs': data_processor_kwargs,
                'stable_cache_data_n': 10000,
                'reuse_data_prob': 1.0,
                'verbose': 0,
                'render_mode': 'human',
                'gamma': 0.92,
            },

            agent_kwargs_update={
                'n_steps': n_steps,
                'batch_size': 60000,
                'n_epochs': 10,
                'stats_window_size': 25,
                'ent_coef': 0.01,
                'clip_range': 0.2,
                'clip_range_vf': 0.2,
                'gamma': 0.92,
                'learning_rate': {'CoSheduller': dict(warmup=warmup_timesteps,
                                                      stable_warmup=True,
                                                      floor_learning_rate=1e-7,
                                                      min_learning_rate=1e-6,
                                                      learning_rate=2.5e-6,
                                                      total_epochs=total_timesteps,
                                                      epsilon=1,
                                                      pre_warmup_coef=0.3333)
                                  },
                'seed': 543,
            },
            env_wrapper='labsubproc',
            env_wrapper_kwargs_update={'use_threads': False},
            n_envs=agents_n_env,
            n_eval_episodes=50,
            eval_freq=n_steps,
            verbose=0,
        )

    # rllab.backtesting_agent(filename=18_600_000, render_mode='human', n_tests=10, verbose=1)
    # rllab.evaluate_agent(0)
