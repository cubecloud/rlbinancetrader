import logging
import datetime
import numpy as np
from dbbinance.fetcher.constants import Constants
from rllab.rllaboratory import LabBase
from binanceenv.bienv import BinanceEnvCash
from stable_baselines3 import A2C, PPO, DDPG, DQN, TD3, SAC
from multiprocessing import freeze_support
from multiprocessing import get_logger
from stable_baselines3.common.buffers import RolloutBuffer

__version__ = 0.025

logger = get_logger()

if __name__ == '__main__':
    freeze_support()

    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler('test_rllaboratory.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s')
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
    # _start_datetime = '2022-12-31 23:00:00'
    _start_datetime = '2023-07-20 01:00:00'
    # _start_datetime = datetime.datetime.strptime('2024-03-01 01:00:00', Constants.default_datetime_format)

    # _end_datetime = datetime.datetime.strptime('2024-07-30 01:00:00', Constants.default_datetime_format)
    # _end_datetime = '2023-07-20 01:00:00'
    # _end_datetime = '2024-07-30 01:00:00'
    # _end_datetime = '2024-11-01 01:00:00'
    _end_datetime = '2024-12-11 01:00:00'

    # data_processor_kwargs = dict(start_datetime=_start_datetime,
    #                              end_datetime=_end_datetime,
    #                              timeframe=_timeframe,
    #                              discretization=_discretization,
    #                              symbol_pair='BTCUSDT',
    #                              market='spot',
    #                              minimum_train_size=0.0267,
    #                              maximum_train_size=0.031,
    #                              minimum_test_size=0.258,
    #                              maximum_test_size=0.278,
    #                              # minimum_test_size=0.7,
    #                              # maximum_test_size=0.9,
    #                              test_size=0.05,
    #                              verbose=0,
    #                              indicators_sign=True
    #                              )

    # data_processor_kwargs = dict(start_datetime=_start_datetime,
    #                              end_datetime=_end_datetime,
    #                              timeframe=_timeframe,
    #                              discretization=_discretization,
    #                              symbol_pair='BTCUSDT',
    #                              market='spot',
    #                              minimum_train_size=500,
    #                              maximum_train_size=600,
    #                              minimum_test_size=500,
    #                              maximum_test_size=600,
    #                              test_size=0.1,
    #                              verbose=0,
    #                              indicators_sign=True
    #                              )

    data_processor_kwargs = dict(start_datetime=_start_datetime,
                                 end_datetime=_end_datetime,
                                 timeframe=_timeframe,
                                 discretization=_discretization,
                                 symbol_pair='BTCUSDT',
                                 market='spot',
                                 minimum_train_size=930,
                                 maximum_train_size=945,
                                 minimum_test_size=915,
                                 maximum_test_size=955,
                                 test_size=0.1,
                                 verbose=1,
                                 indicators_sign=True
                                 )
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
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-1212-030411/MaskablePPO_BinanceEnvCash_900000000_cfg.json'    # -< original
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-1612-220439/MaskablePPO_BinanceEnvCash_1200000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-1912-124618/MaskablePPO_BinanceEnvCash_1200000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-2312-172848/MaskablePPO_BinanceEnvCash_900000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-2512-004141/MaskablePPO_BinanceEnvCash_1200000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-2912-004826/MaskablePPO_BinanceEnvCash_1200000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-3012-164947/MaskablePPO_BinanceEnvCash_1200000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-0201-015059/MaskablePPO_BinanceEnvCash_1200000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-0301-115724/MaskablePPO_BinanceEnvCash_1200000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-0501-081315/MaskablePPO_BinanceEnvCash_1200000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-0901-155926/MaskablePPO_BinanceEnvCash_1200000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-1101-120227/MaskablePPO_BinanceEnvCash_1200000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-1401-040250/MaskablePPO_BinanceEnvCash_1800000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-1501-010030/MaskablePPO_BinanceEnvCash_1500000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-1701-200906/MaskablePPO_BinanceEnvCash_3000000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-0702-102640/MaskablePPO_BinanceEnvCash_3000000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-1702-175422/MaskablePPO_BinanceEnvCash_1500000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-1802-073821/MaskablePPO_BinanceEnvCash_1500000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-2802-124403/MaskablePPO_BinanceEnvCash_1500000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/MaskablePPO/exp-2802-192401/MaskablePPO_BinanceEnvCash_1500000000_cfg.json'
    # json_cfg = './save/BinanceEnvCash/RecurrentPPO/exp-1203-114139/RecurrentPPO_BinanceEnvCash_50000000_cfg.json'
    # json_cfg = '/home/cubecloud/Backup/Experiments/rlbinancetrader/save/BinanceEnvCash/MaskablePPO/exp-2103-093747/MaskablePPO_BinanceEnvCash_3000000000_cfg.json'
    json_cfg = '/home/cubecloud/Backup/Experiments/rlbinancetrader/save/BinanceEnvCash/MaskablePPO/exp-2703-204027/MaskablePPO_BinanceEnvCash_300000000_cfg.json'

    rllab = LabBase.load_agent(json_cfg)
    # rllab = LabBase.load_agent(json_cfg, '/home/cubecloud/Backup/Experiments/rlbinancetrader/save')
    # rllab.test_agent(filename='best_model', verbose=1)
    # rllab.test_agent(filename=750_000, n_tests=15, verbose=1)
    """ Sell action reward """
    rllab.backtesting_agent(filename='best_model', render_mode='human', n_tests=20, verbose=1, seed=443,
                            use_period='test')
    # rllab.backtesting_agent(filename='best_model', render_mode='human', n_tests=20, verbose=1, seed=443,
    #                         use_period='train')
    # rllab.backtesting_agent(filename=16380000, render_mode='human', n_tests=20, verbose=1, seed=443,
    #                         use_period='test')
    # rllab.backtesting_agent(filename='best_model', render_mode='human', n_tests=20, verbose=1, seed=443,
    #                         use_period='check',
    #                         data_processor_kwargs=data_processor_kwargs)
    # rllab.evaluate_agent(0)
