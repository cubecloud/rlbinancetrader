import logging
import datetime
import numpy as np
import pandas as pd
from datetime import timezone

from dateutil.relativedelta import relativedelta
from dbbinance.fetcher.datautils import get_timedelta_kwargs
from dbbinance.fetcher.datafetcher import ceil_time, floor_time
from dbbinance.fetcher.datafetcher import DataFetcher
from dbbinance.fetcher.constants import Constants

from dbbinance.config.configpostgresql import ConfigPostgreSQL
from datawizard.dataprocessor import IndicatorProcessor, ProcessorBase
from indicators import LoadDbIndicators
from rllab.labtools import round_up
import multiprocessing as mp

__version__ = 0.021

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('test_dataprocessor.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

_timeframe = '15m'
_discretization = '15m'

_start_datetime = '2023-07-20 01:00:00'
_end_datetime = '2024-07-30 01:00:00'

data_processor_kwargs = dict(start_datetime=_start_datetime,
                             end_datetime=_end_datetime,
                             timeframe=_timeframe,
                             discretization=_discretization,
                             symbol_pair='BTCUSDT',
                             market='spot',
                             minimum_train_size=0.0227,
                             maximum_train_size=0.0237,
                             minimum_test_size=0.148,
                             maximum_test_size=0.17,
                             test_size=0.13,
                             verbose=1,
                             )

dp = IndicatorProcessor(**data_processor_kwargs)
n_episodes = 630
episodes_start_end_lst = dp.get_n_episodes_start_end_lst(index_type='target_time',
                                                         period_type='train',
                                                         n_episodes=n_episodes)
# episodes_start_end_lst = list(set(episodes_start_end_lst))
print(f'Length = {len(episodes_start_end_lst)}')

for ix, (start, end) in enumerate(episodes_start_end_lst):
    print(f'{ix}: {start} - {end}')

n_envs = 15

if n_envs == 'auto':
    n_envs = max(1, min(n_episodes, mp.cpu_count() - 1))
    if len(episodes_start_end_lst) < n_envs and n_envs > 1:
        n_envs = len(episodes_start_end_lst)

n_episodes_per_env = int(round_up(max(1., len(episodes_start_end_lst) / n_envs), 0))

env_start_end_lst: list = []

indices = np.arange(0, len(episodes_start_end_lst) + 1, n_episodes_per_env)
for idx in indices:
    ep_start_end = episodes_start_end_lst[idx: min(idx + n_episodes_per_env, len(episodes_start_end_lst)+1)]
    env_start_end_lst.append(ep_start_end)

print(env_start_end_lst)

# print(len(episodes_start_end_lst))
