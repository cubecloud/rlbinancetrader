import logging
import datetime

from dateutil.relativedelta import relativedelta
from dbbinance.fetcher.datautils import get_timedelta_kwargs
from dbbinance.fetcher.datafetcher import ceil_time, floor_time
from dbbinance.fetcher.constants import Constants

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

from binanceenv.bienv import BinanceEnvBase

__version__ = 0.031

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('test_dqn_model.log')
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
                             minimum_test_size=0.05
                             )

logger.info(f"{__name__}: Testing learned model on same period as trained {_start_datetime} - {_end_datetime}")
env_kwargs = dict(data_processor_kwargs=data_processor_kwargs,
                  pnl_stop=-0.9,
                  verbose=1,
                  log_interval=500,
                  observation_type='indicators')
vec_env = make_vec_env(BinanceEnvBase, n_envs=1, seed=0, env_kwargs=env_kwargs)
# env = BinanceEnvBase(data_processor_kwargs, pnl_stop=-0.9, verbose=1, observation_type='indicators')

# check_env(env)  # It will check your custom environment and output additional warnings if needed
model = DQN.load(env=vec_env,
                 path=f'/home/cubecloud/Python/projects/rlbinancetrader/tests/save/BinanceEnvBase/DQN/exp-0207-025354/training/DQN_BinanceEnvBase_2000000_chkp_1920000_steps.zip')

# vec_env = model.get_env()
for _ in range(10):
    observation = vec_env.reset()
    done = False
    while not done:
        action, _state = model.predict(observation, deterministic=True)
        observation, reward, terminated, info = vec_env.step(action)
        if terminated:
            observation = vec_env.reset()
            break
vec_env.close()

_timedelta_kwargs = get_timedelta_kwargs('8w', current_timeframe=_timeframe)
_start_datetime = floor_time(datetime.datetime.utcnow(), '1m')
_start_datetime = _start_datetime - relativedelta(**_timedelta_kwargs)

_timedelta_kwargs = get_timedelta_kwargs('1d', current_timeframe=_timeframe)
_end_datetime = floor_time(datetime.datetime.utcnow(), '1m')
_end_datetime = _end_datetime - relativedelta(**_timedelta_kwargs)

data_processor_kwargs = dict(start_datetime=_start_datetime,
                             end_datetime=_end_datetime,
                             timeframe=_timeframe,
                             discretization=_discretization,
                             symbol_pair='BTCUSDT',
                             market='spot',
                             minimum_test_size=0.65
                             )

logger.info(f"{__name__}: Testing learned model on NEW period {_start_datetime} - {_end_datetime}")

env = BinanceEnvBase(data_processor_kwargs, pnl_stop=-0.9, verbose=1, observation_type='indicators')

check_env(env)  # It will check your custom environment and output additional warnings if needed
model = DQN.load(env=env,
                 path=f'/home/cubecloud/Python/projects/rlbinancetrader/tests/save/BinanceEnvBase/DQN/exp-0207-025354/training/DQN_BinanceEnvBase_2000000_chkp_1920000_steps.zip')
# vec_env = model.get_env()
for _ in range(10):
    observation = vec_env.reset()
    done = False
    while not done:
        action, _state = model.predict(observation, deterministic=True)
        observation, reward, terminated, info = vec_env.step(action)
        if terminated:
            observation = vec_env.reset()
            break
vec_env.close()
