import sys
sys.path.insert(0, '/home/cubecloud/Python/projects/rlbinancetrader')
import logging
# import datetime
# import gc
import numpy as np
from dbbinance.fetcher.datautils import get_timeframe_bins
# from dateutil.relativedelta import relativedelta
# from dbbinance.fetcher.datautils import get_timedelta_kwargs
# from dbbinance.fetcher.datafetcher import ceil_time, floor_time
# from dbbinance.fetcher.constants import Constants
#
# from stable_baselines3 import HerReplayBuffer
# from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
# from stable_baselines3.common.envs import BitFlippingEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from stable_baselines3 import A2C, PPO, DDPG, DQN, TD3, SAC
from sb3_contrib import MaskablePPO
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
# from torch.nn import Tanh, Softmax, LeakyReLU, ReLU
# from binanceenv.bienv import BinanceEnvBase
from binanceenv.bienv import BinanceEnvCash

from customnn.mlpextractor import MlpExtractorNN
from rllab.rllaboratory import LabBase
# from rllab.labcosheduller import CoSheduller
from multiprocessing import freeze_support
import warnings

# import torch

__version__ = 0.105

logger = logging.getLogger()

if __name__ == '__main__':
    freeze_support()

    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('test_rllab_mask_ppo.log')
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
    _end_datetime = '2024-07-30 01:00:00'
    # _timedelta_kwargs = get_timedelta_kwargs(_gap_period, current_timeframe=_timeframe)
    # _end_datetime = floor_time(datetime.datetime.utcnow(), '1m')
    #
    # _end_datetime = _end_datetime - relativedelta(**_timedelta_kwargs)

    agents_n_env = 2520
    total_timesteps = 300_000_000
    # buffer_size = 1_500_000
    learning_start = 2520 * 100
    # batch_size = 660 * agents_n_env
    lookback_window = '12h'
    seed = 531

    data_processor_kwargs = dict(start_datetime=_start_datetime,
                                 end_datetime=_end_datetime,
                                 timeframe=_timeframe,
                                 discretization=_discretization,
                                 symbol_pair='BTCUSDT',
                                 market='spot',
                                 minimum_train_size=0.0267,
                                 maximum_train_size=0.03,
                                 minimum_test_size=0.168,
                                 maximum_test_size=0.188,
                                 test_size=0.13,
                                 verbose=0,
                                 indicators_sign=True
                                 )

    env_discrete_kwargs = dict(data_processor_kwargs=data_processor_kwargs,
                               pnl_stop=-0.9,
                               verbose=0,
                               log_interval=1,
                               seed=seed,
                               target_balance=5_000.,
                               target_minimum_trade=100.,
                               target_maximum_trade=500.,
                               target_scale_decay=100_000,
                               # observation_type='lookback_dict',
                               # observation_type='assets_close_indicators',
                               observation_type='lookback_assets_close_indicators_action_ret',
                               # observation_type='indicators_close',
                               stable_cache_data_n=3150,  # 630*5 = 3150
                               reuse_data_prob=1.0,
                               eval_reuse_prob=1.0,
                               # lookback_window=None,
                               lookback_window=lookback_window,
                               max_hold_timeframes='24h',
                               total_timesteps=total_timesteps,
                               # eps_start=0.99,
                               # eps_end=0.01,
                               # eps_decay=0.2,
                               # gamma=0.995,
                               # gamma=0.99,    # not used! in agent model
                               # reduced by 10 (from 0.999) to have less reward backpropagation for 15 min 24h*4 = 96 timesteps
                               # invalid_actions=15_000,
                               penalty_value=1e-6,
                               action_type='discrete',
                               index_type='target_time',
                               # index_type='prediction_time',
                               render_mode='human',
                               )

    features_dim = 256
    ppo_policy_kwargs = dict(
        features_extractor_class='LSTMExtractorNN',
        features_extractor_kwargs=dict(features_dim=features_dim,
                                       activation_fn='Swish'),
        share_features_extractor=True,
        net_arch=[features_dim, 256, 128],
    )

    ppo_kwargs = dict(
        policy="MlpPolicy",
        # policy="MultiInputPolicy",
        policy_kwargs=ppo_policy_kwargs,
        n_steps=250,
        batch_size=25200,
        n_epochs=10,
        stats_window_size=25,
        ent_coef=0.01,
        normalize_advantage=False,
        clip_range=0.2,
        learning_rate={'CoSheduller': dict(warmup=learning_start,
                                           learning_rate=1e-3,
                                           min_learning_rate=5e-4,
                                           total_epochs=total_timesteps,
                                           epsilon=1)
                       },
        # lookback window (timesteps) / 100 -> 12h * 4 = 48
        gamma=0.99,
        device='auto',
        seed=seed,
        verbose=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        rllab = LabBase(
            env_cls=[BinanceEnvCash],
            env_kwargs=[env_discrete_kwargs],
            agents_cls=[MaskablePPO],
            agents_kwargs=[ppo_kwargs],
            agents_n_env=[agents_n_env],
            env_wrapper='labsubproc',
            total_timesteps=total_timesteps,
            checkpoint_num=250,
            n_eval_episodes=50,
            log_interval=1,
            eval_freq=250,
            experiment_path='/home/cubecloud/Python/projects/rlbinancetrader/tests/save',
            deterministic=False,
            verbose=0,
            seed=seed,
        )

        rllab.learn()

