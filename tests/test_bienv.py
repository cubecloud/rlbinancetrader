import logging
import datetime
import numpy as np
import pandas as pd

# from datetime import timezone

from dateutil.relativedelta import relativedelta
from dbbinance.fetcher.datautils import get_timedelta_kwargs
from dbbinance.fetcher.datafetcher import ceil_time, floor_time
from dbbinance.fetcher.constants import Constants

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C, PPO, DDPG, DQN
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from binanceenv.bienv import BinanceEnvBase

__version__ = 0.03

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('test_bienv.log')
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
_gap_period = '8w'

_start_datetime = datetime.datetime.strptime('2023-07-15 01:00:00', Constants.default_datetime_format)

_timedelta_kwargs = get_timedelta_kwargs(_gap_period, current_timeframe=_timeframe)
_end_datetime = floor_time(datetime.datetime.utcnow(), '1m')
_end_datetime = _end_datetime - relativedelta(**_timedelta_kwargs)

data_processor_kwargs = dict(start_datetime=_start_datetime,
                             end_datetime=_end_datetime,
                             timeframe=_timeframe,
                             discretization=_discretization,
                             symbol_pair='BTCUSDT',
                             market='spot',
                             minimum_test_size=0.05,
                             )

env_kwargs = dict(data_processor_kwargs=data_processor_kwargs,
                  pnl_stop=-0.9,
                  verbose=1,
                  log_interval=500,
                  observation_type='indicators')

# env = BinanceEnvBase(data_processor_kwargs, pnl_stop=-0.9, verbose=1, log_interval=500,
#                      observation_type='indicators')

# check_env(env)  # It will check your custom environment and output additional warnings if needed
# env.action_space.seed(42)
# observation, info = env.reset(seed=42)
# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# model = A2C("MlpPolicy", env, n_steps=5, learning_rate=3e-4, stats_window_size=500, verbose=1)
# model = PPO("MlpPolicy", env, stats_window_size=500, verbose=2)
# dqn_kwargs = dict(policy="MlpPolicy",
#                   batch_size=64,
#                   exploration_fraction=0.001,
#                   buffer_size=50_000,
#                   learning_starts=50_001,
#                   train_freq=(1, 'episode'),
#                   tau=0.3,
#                   seed=42,
#                   verbose=1)
#
# model = DQN(env=env, **dqn_kwargs)
vec_env = make_vec_env(BinanceEnvBase, n_envs=1, seed=0, env_kwargs=env_kwargs)

model = DQN("MlpPolicy", vec_env, batch_size=32, exploration_fraction=0.01, buffer_size=70_000, learning_starts=70_001,
            verbose=1)
# model = DDPG("MlpPolicy", env, verbose=2)
timesteps = 1_000_000
model.learn(total_timesteps=timesteps, log_interval=1000, progress_bar=False)
model.save(path=f'/home/cubecloud/Python/projects/rlbinancetrader/tests/save/dqn_{timesteps}_model_{BinanceEnvBase.__class__}{env_kwargs["observation_type"]}')
del model

# env = BinanceEnvBase(data_processor_kwargs, pnl_stop=-0.9, verbose=1, observation_type='indicators')

model = DQN.load(env=vec_env, path=f'/home/cubecloud/Python/projects/rlbinancetrader/tests/save/dqn_{timesteps}_model_{BinanceEnvBase.__class__}{env_kwargs["observation_type"]}')

logger.info(f"{__name__}: Testing learned model")

# check_env(env)  # It will check your custom environment and output additional warnings if needed
# env.action_space.seed(42)

# vec_env = model.get_env()
for _ in range(10):
    observation, info = vec_env.reset()
    done = False
    while not done:
        action, _state = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = vec_env.step(action)
        if terminated or truncated:
            observation, info = vec_env.reset()
            break
vec_env.close()
