import logging
import datetime
import numpy as np
from dbbinance.fetcher.constants import Constants
from rllab.rllaboratory import LabBase
from binanceenv.bienv import BinanceEnvCash
from stable_baselines3 import A2C, PPO, DDPG, DQN, TD3, SAC
from multiprocessing import freeze_support
from stable_baselines3.common.buffers import RolloutBuffer

__version__ = 0.0020

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

    _timeframe = '15m'
    _discretization = '15m'
    _gap_period = '5d'

    # _start_datetime = datetime.datetime.strptime('2023-07-20 01:00:00', Constants.default_datetime_format)
    _start_datetime = '2023-07-20 01:00:00'
    # _start_datetime = datetime.datetime.strptime('2024-03-01 01:00:00', Constants.default_datetime_format)

    # _end_datetime = datetime.datetime.strptime('2024-07-30 01:00:00', Constants.default_datetime_format)
    # _end_datetime = '2024-07-30 01:00:00'
    # _end_datetime = '2024-11-01 01:00:00'
    _end_datetime = '2024-12-11 01:00:00'

    data_processor_kwargs = dict(start_datetime=_start_datetime,
                                 end_datetime=_end_datetime,
                                 timeframe=_timeframe,
                                 discretization=_discretization,
                                 symbol_pair='BTCUSDT',
                                 market='spot',
                                 minimum_train_size=0.0267,
                                 maximum_train_size=0.031,
                                 minimum_test_size=0.258,
                                 maximum_test_size=0.278,
                                 # minimum_test_size=0.7,
                                 # maximum_test_size=0.9,
                                 test_size=0.05,
                                 verbose=0,
                                 indicators_sign=True
                                 )

    # data_processor_kwargs = dict(start_datetime=_start_datetime,
    #                              end_datetime=_end_datetime,
    #                              timeframe=_timeframe,
    #                              discretization=_discretization,
    #                              symbol_pair='BTCUSDT',
    #                              market='spot',
    #                              minimum_train_size=0.0267,
    #                              maximum_train_size=0.031,
    #                              minimum_test_size=0.2,
    #                              maximum_test_size=0.3,
    #                              test_size=0.2,
    #                              verbose=0,
    #                              indicators_sign=True
    #                              )

    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-1011-230816/MaskablePPO_BinanceEnvCash_300000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-2311-021257/MaskablePPO_BinanceEnvCash_300000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-2411-231048/MaskablePPO_BinanceEnvCash_900000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-2911-102136/MaskablePPO_BinanceEnvCash_900000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-3011-174850/MaskablePPO_BinanceEnvCash_900000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-0112-232354/MaskablePPO_BinanceEnvCash_900000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-0212-222359/MaskablePPO_BinanceEnvCash_900000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-0312-120029/MaskablePPO_BinanceEnvCash_900000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-0812-012052/MaskablePPO_BinanceEnvCash_900000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-0912-140611/MaskablePPO_BinanceEnvCash_900000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-1112-020256/MaskablePPO_BinanceEnvCash_900000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-1212-030411/MaskablePPO_BinanceEnvCash_900000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-1612-220439/MaskablePPO_BinanceEnvCash_1200000000_cfg.json'
    json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-1912-124618/MaskablePPO_BinanceEnvCash_1200000000_cfg.json'


    rllab = LabBase.load_agent(json_cfg)
    # rllab.test_agent(filename='best_model', verbose=1)
    # rllab.test_agent(filename=750_000, n_tests=15, verbose=1)
    """ Sell action reward """
    # rllab.backtesting_agent(filename='best_model', render_mode='human', n_tests=20, verbose=1, seed=443,
    #                         use_period='test')
    # rllab.backtesting_agent(filename=190_512_000, render_mode='human', n_tests=20, verbose=1, seed=443,
    #                         use_period='test')
    rllab.backtesting_agent(filename=281_232_000, render_mode='human', n_tests=20, verbose=1, seed=443,
                            use_period='check',
                            data_processor_kwargs=data_processor_kwargs)
    # rllab.evaluate_agent(0)
