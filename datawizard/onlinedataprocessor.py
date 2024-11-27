import random
import sys
import logging

import pandas as pd

from datetime import timezone, datetime
from dateutil.relativedelta import relativedelta
from typing import Union, List, Tuple

from dbbinance.fetcher import check_convert_to_datetime
from dbbinance.fetcher.getfetcher import get_datafetcher
from dbbinance.fetcher.datafetcher import DataFetcher
from dbbinance.fetcher.datautils import convert_timeframe_to_freq
from dbbinance.fetcher.datautils import get_timedelta_kwargs
from dbbinance.fetcher.constants import Constants

from indicators import LoadDbIndicators
import multiprocessing as mp

__version__ = 0.091


logger = logging.getLogger()

mp_count = mp.Value('i', 0)


class OnlineProcessorBase:
    count = mp_count

    def __init__(self,
                 timeframe,
                 discretization,
                 symbol_pair='BTCUSDT',
                 market='spot',
                 host=None,
                 database=None,
                 user=None,
                 password=None,
                 verbose: int = 0,
                 seed=42):

        with OnlineProcessorBase.count.get_lock():
            OnlineProcessorBase.count.value += 1

        self.idnum = int(OnlineProcessorBase.count.value)
        self.timeframe = timeframe
        self.discretization = discretization
        self.market = market
        self.symbol_pair = symbol_pair

        self.verbose = verbose
        self.seed = seed
        self.fetcher = get_datafetcher(host=host, database=database, user=user, password=password)

    def __del__(self):
        with OnlineProcessorBase.count.get_lock():
            OnlineProcessorBase.count.value -= 1

    def get_ohlcv_and_indicators(self, start_datetime, end_datetime, index_type='target_time'):
        pass