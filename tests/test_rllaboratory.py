import logging
import datetime
import gc
import numpy as np
from dateutil.relativedelta import relativedelta
from dbbinance.fetcher.datautils import get_timedelta_kwargs
from dbbinance.fetcher.datafetcher import ceil_time, floor_time
from dbbinance.fetcher.constants import Constants

from stable_baselines3 import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.envs import BitFlippingEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from stable_baselines3 import A2C, PPO, DDPG, DQN, TD3, SAC
# from torch.nn import Tanh, Softmax, LeakyReLU, ReLU
from binanceenv.bienv import BinanceEnvBase
from rllab.rllaboratory import LabBase
from multiprocessing import freeze_support

__version__ = 0.041

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
    _gap_period = '5d'

    _start_datetime = datetime.datetime.strptime('2023-07-20 01:00:00', Constants.default_datetime_format)
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
                                 minimum_train_size=0.08,
                                 maximum_train_size=0.42,
                                 minimum_test_size=0.8,
                                 maximum_test_size=0.99,
                                 test_size=0.1,
                                 verbose=0,
                                 )

    env_discrete_kwargs = dict(data_processor_kwargs=data_processor_kwargs,
                               pnl_stop=-0.20,
                               # max_lot_size=0.5,
                               verbose=1,
                               log_interval=500,
                               observation_type='indicators_assets',
                               reuse_data_prob=0.5,
                               eval_reuse_prob=0.95,
                               max_hold_timeframes='5d',
                               total_timesteps=5_000_000,
                               invalid_actions=60,
                               penalty_value=7,
                               action_type='discrete',
                               )
    env_box_kwargs = dict(data_processor_kwargs=data_processor_kwargs,
                          pnl_stop=-0.6,
                          # max_lot_size=0.5,
                          verbose=1,
                          log_interval=100,
                          observation_type='indicators_assets',
                          reuse_data_prob=0.5,
                          eval_reuse_prob=0.95,
                          max_hold_timeframes='5d',
                          total_timesteps=5_000_000,
                          invalid_actions=60,
                          penalty_value=7,
                          action_type='box',
                          )

    env_box1_1_kwargs = dict(data_processor_kwargs=data_processor_kwargs,
                             pnl_stop=-0.5,
                             # max_lot_size=0.5,
                             verbose=1,
                             log_interval=1,
                             observation_type='assets_close_indicators_action_masks',
                             # observation_type='indicators_assets',
                             reuse_data_prob=0.99,
                             eval_reuse_prob=0.99,
                             max_hold_timeframes='2d',
                             total_timesteps=25_000_000,
                             gamma=0.99,
                             invalid_actions=200,
                             penalty_value=1e-5,
                             action_type='box1_1',
                             )

    env_binbox_kwargs = dict(data_processor_kwargs=data_processor_kwargs,
                             pnl_stop=-0.5,
                             # max_lot_size=0.5,
                             verbose=1,
                             log_interval=100,
                             observation_type='idx_assets_close_indicators_action_masks',
                             reuse_data_prob=0.97,
                             eval_reuse_prob=0.99,
                             max_hold_timeframes='2d',
                             total_timesteps=8_000_000,
                             gamma=0.99,
                             invalid_actions=200,
                             penalty_value=1e-5,
                             action_type='binbox',
                             )

    dqn_kwargs = dict(policy="MlpPolicy",
                      batch_size=64,
                      stats_window_size=50000,
                      exploration_fraction=0.95,
                      exploration_final_eps=0.05,
                      exploration_initial_eps=1.0,
                      buffer_size=100_000,
                      learning_starts=100_001,
                      # train_freq=(10, 'step'),
                      # train_freq=(1, 'episode'),
                      device='cpu',
                      verbose=1)

    ppo_policy_kwargs = dict(net_arch=dict(pi=[32, 32],
                                           vf=[64, 64]))

    ppo_kwargs = dict(policy="MlpPolicy",
                      policy_kwargs=ppo_policy_kwargs,
                      batch_size=32,
                      stats_window_size=10000,
                      normalize_advantage=True,
                      use_sde=True,
                      sde_sample_freq=10,
                      device='cpu',
                      verbose=1)

    ddpg_kwargs = dict(policy="MlpPolicy",
                       stats_window_size=10000,
                       device='cpu',
                       verbose=1)

    a2c_policy_kwargs = dict(net_arch=dict(pi=[32, 32],
                                           vf=[64, 64]))

    a2c_kwargs = dict(policy="MlpPolicy",
                      policy_kwargs=a2c_policy_kwargs,
                      stats_window_size=10_000,
                      use_rms_prop=False,
                      use_sde=True,
                      sde_sample_freq=10,
                      device="auto",
                      verbose=1)

    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(3), sigma=5e-2 * np.ones(3), dt=1e-2)
    # action_noise = NormalActionNoise(mean=np.zeros(3), sigma=0.1 * np.ones(3))
    # td3_policy_kwargs = dict(net_arch=dict(pi=[27, 9, 3, 9],
    #                                        qf=[128, 32, 8, 32]))
    # td3_policy_kwargs = dict(net_arch=dict(pi=[64, 32, 16],
    #                                        qf=[256, 128, 64]))
    td3_kwargs = dict(policy="MlpPolicy",
                      buffer_size=500_000,
                      # policy_kwargs=td3_policy_kwargs,
                      batch_size=256,
                      learning_starts=500_000,
                      stats_window_size=100,
                      learning_rate=0.0003,
                      action_noise=action_noise,
                      train_freq=(100, 'step'),
                      device="auto",
                      verbose=1)

    sac_kwargs = dict(policy="MlpPolicy",
                      buffer_size=500_000,
                      batch_size=256,
                      learning_starts=50_000,
                      stats_window_size=100,
                      action_noise=action_noise,
                      device="auto",
                      verbose=1)

    # rllab = LabBase(
    #     env_cls=[BinanceEnvBase, BinanceEnvBase, BinanceEnvBase],
    #     env_kwargs=[env_box1_1_kwargs, env_box1_1_kwargs, env_discrete_kwargs],
    #     agents_cls=[PPO, DDPG, DQN],
    #     agents_kwargs=[ppo_kwargs, ddpg_kwargs, dqn_kwargs],
    #     agents_n_env=[3, 3, 3],
    #     total_timesteps=5_000_000,
    #     checkpoint_num=50,
    #     n_eval_episodes=20,
    #     eval_freq=100_000,
    #     experiment_path='/home/cubecloud/Python/projects/rlbinancetrader/tests/save')
    #
    # rllab.learn()
    #
    # del rllab
    # gc.collect()

    # rllab = LabBase(
    #     env_cls=[BinanceEnvBase, BinanceEnvBase, BinanceEnvBase],
    #     env_kwargs=[env_box1_1_kwargs, env_box1_1_kwargs, env_box1_1_kwargs],
    #     agents_cls=[TD3, SAC, A2C],
    #     agents_kwargs=[td3_kwargs, sac_kwargs, a2c_kwargs],
    #     agents_n_env=[1, 1, 2],
    #     total_timesteps=5_000_000,
    #     checkpoint_num=100,
    #     n_eval_episodes=20,
    #     eval_freq=50_000,
    #     experiment_path='/home/cubecloud/Python/projects/rlbinancetrader/tests/save')

    rllab = LabBase(
        env_cls=[BinanceEnvBase],
        env_kwargs=[env_box1_1_kwargs],
        agents_cls=[TD3],
        agents_kwargs=[td3_kwargs],
        agents_n_env=[1],
        env_wrapper='dummy',
        total_timesteps=25_000_000,
        checkpoint_num=80,
        n_eval_episodes=20,
        eval_freq=100_000,
        experiment_path='/home/cubecloud/Python/projects/rlbinancetrader/tests/save')

    rllab.learn()
