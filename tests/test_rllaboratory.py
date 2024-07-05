import logging
import datetime
import gc

from dateutil.relativedelta import relativedelta
from dbbinance.fetcher.datautils import get_timedelta_kwargs
from dbbinance.fetcher.datafetcher import ceil_time, floor_time
from dbbinance.fetcher.constants import Constants

from stable_baselines3 import A2C, PPO, DDPG, DQN, TD3, SAC

from binanceenv.bienv import BinanceEnvBase
from rllab.rllaboratory import LabBase
from multiprocessing import freeze_support

__version__ = 0.035

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
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    _timeframe = '15m'
    _discretization = '15m'
    _gap_period = '10d'

    _start_datetime = datetime.datetime.strptime('2023-07-15 01:00:00', Constants.default_datetime_format)
    # _start_datetime = datetime.datetime.strptime('2024-03-01 01:00:00', Constants.default_datetime_format)

    _timedelta_kwargs = get_timedelta_kwargs(_gap_period, current_timeframe=_timeframe)
    _end_datetime = floor_time(datetime.datetime.utcnow(), '1m')
    _end_datetime = _end_datetime - relativedelta(**_timedelta_kwargs)

    data_processor_kwargs = dict(start_datetime=_start_datetime,
                                 end_datetime=_end_datetime,
                                 timeframe=_timeframe,
                                 discretization=_discretization,
                                 symbol_pair='BTCUSDT',
                                 market='spot',
                                 minimum_train_size=0.05,
                                 maximum_train_size=0.5,
                                 minimum_test_size=0.5,
                                 maximum_test_size=0.97,
                                 test_size=0.2,
                                 verbose=0,
                                 )

    env_discrete_kwargs = dict(data_processor_kwargs=data_processor_kwargs,
                               pnl_stop=-0.9,
                               # max_lot_size=0.5,
                               verbose=0,
                               log_interval=500,
                               observation_type='indicators_assets',
                               reuse_data_prob=0.5,
                               eval_reuse_prob=0.05,
                               max_hold_timeframes='3d',
                               penalty_value=10,
                               action_type='discrete',
                               )
    env_box_kwargs = dict(data_processor_kwargs=data_processor_kwargs,
                          pnl_stop=-0.9,
                          # max_lot_size=0.5,
                          verbose=1,
                          log_interval=500,
                          observation_type='indicators_assets',
                          reuse_data_prob=0.4,
                          eval_reuse_prob=0.05,
                          max_hold_timeframes='3d',
                          penalty_value=10,
                          action_type='box',
                          )

    env_box1_1_kwargs = dict(data_processor_kwargs=data_processor_kwargs,
                             pnl_stop=-0.9,
                             # max_lot_size=0.5,
                             verbose=1,
                             log_interval=500,
                             observation_type='indicators_assets',
                             reuse_data_prob=0.4,
                             eval_reuse_prob=0.05,
                             max_hold_timeframes='3d',
                             penalty_value=10,
                             action_type='box1_1',
                             )

    env_binbox_kwargs = dict(data_processor_kwargs=data_processor_kwargs,
                             pnl_stop=-0.9,
                             # max_lot_size=0.5,
                             verbose=0,
                             log_interval=500,
                             observation_type='indicators',
                             reuse_data_prob=0.4,
                             eval_reuse_prob=0.05,
                             max_hold_timeframes='3d',
                             penalty_value=10,
                             action_type='binbox',
                             )

    dqn_kwargs = dict(policy="MlpPolicy",
                      batch_size=64,
                      stats_window_size=100_000,
                      exploration_fraction=0.95,
                      exploration_final_eps=0.05,
                      exploration_initial_eps=1.0,
                      buffer_size=200_000,
                      learning_starts=200_001,
                      # train_freq=(10, 'step'),
                      # train_freq=(1, 'episode'),
                      device='cpu',
                      verbose=1)

    ppo_kwargs = dict(policy="MlpPolicy",
                      device='cpu',
                      verbose=1)

    ddpg_kwargs = dict(policy="MlpPolicy",
                       device='cpu',
                       verbose=1)

    a2c_kwargs = dict(policy="MlpPolicy",
                      device="cpu",
                      verbose=1)

    td3_kwargs = dict(policy="MlpPolicy",
                      device="cpu",
                      verbose=1)

    sac_kwargs = dict(policy="MlpPolicy",
                      device="cpu",
                      verbose=1)

    rllab = LabBase(
        env_cls=[BinanceEnvBase, BinanceEnvBase, BinanceEnvBase],
        env_kwargs=[env_discrete_kwargs, env_box1_1_kwargs, env_box1_1_kwargs,],
        agents_cls=[DQN, PPO, DDPG],
        agents_kwargs=[dqn_kwargs, ppo_kwargs, ddpg_kwargs],
        total_timesteps=10_000_000,
        checkpoint_num=40,
        n_eval_episodes=20,
        eval_freq=200_000,
        experiment_path='/home/cubecloud/Python/projects/rlbinancetrader/tests/save')

    rllab.learn()

    del rllab
    gc.collect()

    rllab = LabBase(
        env_cls=[BinanceEnvBase, BinanceEnvBase, BinanceEnvBase],
        env_kwargs=[env_box1_1_kwargs, env_box1_1_kwargs, env_box1_1_kwargs],
        agents_cls=[A2C, TD3, SAC],
        agents_kwargs=[a2c_kwargs, td3_kwargs, sac_kwargs],
        total_timesteps=10_000_000,
        checkpoint_num=40,
        n_eval_episodes=20,
        eval_freq=200_000,
        experiment_path='/home/cubecloud/Python/projects/rlbinancetrader/tests/save')

    rllab.learn()

