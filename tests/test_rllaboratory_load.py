import logging
import datetime
import numpy as np
from dbbinance.fetcher.constants import Constants
from rllab.rllaboratory import LabBase
from binanceenv.bienv import BinanceEnvCash
from stable_baselines3 import A2C, PPO, DDPG, DQN, TD3, SAC
from multiprocessing import freeze_support

__version__ = 0.0016

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

    # total_timesteps = 16_000_000
    # buffer_size = 1_000_000
    # learning_start = 750_000
    # batch_size = 1024
    #
    # _timeframe = '15m'
    # _discretization = '15m'
    # _gap_period = '5d'
    #
    # _start_datetime = datetime.datetime.strptime('2023-07-20 01:00:00', Constants.default_datetime_format)
    # # _start_datetime = datetime.datetime.strptime('2024-03-01 01:00:00', Constants.default_datetime_format)
    #
    # _end_datetime = datetime.datetime.strptime('2024-07-30 01:00:00', Constants.default_datetime_format)
    #
    # data_processor_kwargs = dict(start_datetime=_start_datetime,
    #                              end_datetime=_end_datetime,
    #                              timeframe=_timeframe,
    #                              discretization=_discretization,
    #                              symbol_pair='BTCUSDT',
    #                              market='spot',
    #                              minimum_train_size=0.027,
    #                              maximum_train_size=0.053,
    #                              minimum_test_size=0.35,
    #                              maximum_test_size=0.5,
    #                              test_size=0.13,
    #                              verbose=0,
    #                              )
    #
    # env_box_kwargs = dict(data_processor_kwargs=data_processor_kwargs,
    #                       pnl_stop=-0.9,
    #                       # max_lot_size=0.5,
    #                       verbose=0,
    #                       log_interval=1,
    #                       seed=42,
    #                       target_balance=100_000.,
    #                       target_minimum_trade=100.,
    #                       observation_type='lookback_assets_close_indicators',
    #                       # observation_type='indicators_close',
    #                       stable_cache_data_n=120,
    #                       reuse_data_prob=0.95,
    #                       eval_reuse_prob=0.9999,
    #                       # lookback_window=None,
    #                       lookback_window='2h',
    #                       max_hold_timeframes='30d',
    #                       total_timesteps=total_timesteps,
    #                       eps_start=0.99,
    #                       eps_end=0.01,
    #                       eps_decay=0.2,
    #                       gamma=0.9995,
    #                       invalid_actions=15_000,
    #                       penalty_value=1e-5,
    #                       action_type='box',
    #                       # index_type='target_time',
    #                       index_type='prediction_time',
    #                       )
    #
    # sac_policy_kwargs = dict(
    #     features_extractor_class='MlpExtractorNN',
    #     features_extractor_kwargs=dict(features_dim=256),
    #     share_features_extractor=True,
    #     activation_fn='LeakyReLU',
    #     # net_arch=net_arch,
    # )
    #
    # sac_kwargs = dict(policy="MlpPolicy",
    #                   buffer_size=buffer_size,
    #                   learning_starts=learning_start,
    #                   policy_kwargs=sac_policy_kwargs,
    #                   batch_size=batch_size,
    #                   stats_window_size=100,
    #                   ent_coef='auto_0.0001',
    #                   learning_rate={'CoSheduller': dict(warmup=learning_start,
    #                                                      learning_rate=2e-4,
    #                                                      min_learning_rate=1e-5,
    #                                                      total_epochs=total_timesteps,
    #                                                      epsilon=100)},
    #                   action_noise={'NormalActionNoise': dict(mean=5e-1 * np.ones(3),
    #                                                           sigma=4.99e-1 * np.ones(3))
    #                                 },
    #                   train_freq=(2, 'step'),
    #                   target_update_interval=5,  # update target network every 10 _gradient_ steps
    #                   device="auto",
    #                   verbose=1)
    #
    # # rllab = LabBase(
    # #     env_cls=[BinanceEnvCash],
    # #     env_kwargs=[env_box_kwargs],
    # #     agents_cls=[SAC],
    # #     agents_kwargs=[sac_kwargs],
    # #     agents_n_env=[4],
    # #     env_wrapper='dummy',
    # #     total_timesteps=total_timesteps,
    # #     checkpoint_num=int(total_timesteps // 100_000),
    # #     n_eval_episodes=50,
    # #     log_interval=200,
    # #     eval_freq=25_000,
    # #     experiment_path='/home/cubecloud/Python/projects/rlbinancetrader/tests/save',
    # #     deterministic=False,
    # #     verbose=0
    # # )

    json_cfg = './save/BinanceEnvCash/SAC/exp-0609-144101/SAC_BinanceEnvCash_16000000_cfg.json'

    rllab = LabBase.load_agent(json_cfg)
    # rllab.test_agent(filename='best_model', verbose=1)
    # rllab.test_agent(filename='SAC_BinanceEnvCash_7000000_chkp_2700000_steps', verbose=1)
    """ Sell action reward """
    # rllab.backtesting_agent(filename='best_model', render_mode='human', n_tests=5, verbose=1)
    rllab.backtesting_agent(filename=3_700_000, render_mode='human', n_tests=5, verbose=1)
    # rllab.evaluate_agent(0)
