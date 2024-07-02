import logging
import datetime

from dateutil.relativedelta import relativedelta
from dbbinance.fetcher.datautils import get_timedelta_kwargs
from dbbinance.fetcher.datafetcher import ceil_time, floor_time
from dbbinance.fetcher.constants import Constants

from stable_baselines3 import A2C, PPO, DDPG, DQN

from binanceenv.bienv import BinanceEnvBase
from rllab.rllaboratory import LabBase

__version__ = 0.01

logger = logging.getLogger()

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
_gap_period = '1w'

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
                             maximum_train_size=0.4,
                             minimum_test_size=0.15,
                             maximum_test_size=0.7,
                             test_size=0.2,
                             verbose=0,
                             )

env_kwargs = dict(data_processor_kwargs=data_processor_kwargs,
                  pnl_stop=-0.9,
                  max_lot_size=0.06,
                  verbose=0,
                  log_interval=1,
                  observation_type='indicators',
                  # action_type='box',
                  action_type='discrete',
                  )

dqn_kwargs = dict(policy="MlpPolicy",
                  batch_size=64,
                  stats_window_size=1000,
                  exploration_fraction=0.5,
                  exploration_final_eps=0.05,
                  exploration_initial_eps=1.0,
                  buffer_size=50_000,
                  learning_starts=10_001,
                  train_freq=(10, 'step'),
                  # train_freq=(1, 'episode'),
                  device='cuda',
                  verbose=1)

ppo_kwargs = dict(policy="MlpPolicy",
                  verbose=1)

rllab = LabBase(env_cls=BinanceEnvBase,
                env_kwargs=env_kwargs,
                agents_cls=[DQN, ],
                agents_kwargs=[dqn_kwargs, ],
                # agents_cls=[PPO, ],
                # agents_kwargs=[ppo_kwargs, ],
                total_timesteps=1_000_000,
                checkpoint_num=25,
                eval_freq=100_000,
                experiment_path='/home/cubecloud/Python/projects/rlbinancetrader/tests/save')

rllab.learn()
