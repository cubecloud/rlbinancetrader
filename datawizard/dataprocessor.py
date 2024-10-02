import random
import sys
import logging

import pandas as pd

from datetime import timezone, datetime
from dateutil.relativedelta import relativedelta
from typing import Union, List, Tuple

from dbbinance.fetcher import check_convert_to_datetime
from dbbinance.fetcher.getfetcher import get_datafetcher
from dbbinance.fetcher.datautils import convert_timeframe_to_freq
from dbbinance.fetcher.datautils import get_timedelta_kwargs
from dbbinance.fetcher.constants import Constants

from indicators import LoadDbIndicators

__version__ = 0.08

from rllab.labtools import round_up

# from tests.test_dataprocessor import episodes_lst

logger = logging.getLogger()


class ProcessorBase:

    def __init__(self, start_datetime, end_datetime, timeframe, discretization,
                 symbol_pair='BTCUSDT',
                 market='spot',
                 minimum_train_size: float = 0.05,
                 maximum_train_size: float = 0.4,
                 minimum_test_size: float = 0.15,
                 maximum_test_size: float = 0.7,
                 test_size: float = 0.2,
                 verbose: int = 0,
                 seed=42):
        self.start_datetime = check_convert_to_datetime(start_datetime, utc_aware=False)
        self.end_datetime = check_convert_to_datetime(end_datetime, utc_aware=False)
        self.timeframe = timeframe
        self.discretization = discretization
        self.market = market
        self.symbol_pair = symbol_pair

        self.__initial_minimum_train_size = minimum_train_size
        self.__initial_minimum_test_size = minimum_test_size
        self.__initial_maximum_train_size = maximum_train_size
        self.__initial_maximum_test_size = maximum_test_size
        self.verbose = verbose
        self.fetcher = get_datafetcher()
        self.seed = seed

        """ get all data from start_datetime until end_datetime to fill datafetcher with RAW data cache """
        self.ohlcv_df = self.get_ohlcv_df(self.start_datetime, self.end_datetime, self.symbol_pair, self.market)
        self.all_period_timeframes = pd.date_range(start=self.start_datetime, end=self.end_datetime,
                                                   freq=convert_timeframe_to_freq(self.timeframe)).to_series().to_list()

        self.train_timeframes_num = int(len(self.all_period_timeframes) * (1 - test_size))
        self.test_timeframes_num = len(self.all_period_timeframes) - self.train_timeframes_num

        msg = (f"TRAIN pool timeframes: {self.train_timeframes_num}, "
               f"Pool period: {self.all_period_timeframes[0]} - {self.all_period_timeframes[self.train_timeframes_num - 1]}")
        logger.info(msg)
        msg = (f"TEST pool timeframes: {self.test_timeframes_num}, "
               f"Pool period: {self.all_period_timeframes[self.train_timeframes_num]} - {self.all_period_timeframes[-1]}")
        logger.info(msg)
        assert self.train_timeframes_num >= 50, f'Error: train_timeframes_num is to LOW {self.train_timeframes_num}'
        assert self.test_timeframes_num >= 50, f'Error: test_timeframes_num is to LOW {self.test_timeframes_num}'
        self.minimum_train_timeframes_num: Union[int, None] = None
        self.maximum_train_timeframes_num: Union[int, None] = None
        self.minimum_test_timeframes_num: Union[int, None] = None
        self.maximum_test_timeframes_num: Union[int, None] = None
        self.train_minute_timeframes_series: Union[pd.DataFrame, None] = None
        self.test_minute_timeframes_series: Union[pd.DataFrame, None] = None

        self.change_train_test_timeframes_num(minimum_train_size, maximum_train_size, minimum_test_size,
                                              maximum_test_size)

    @property
    def initial_minimum_train_size(self):
        return self.__initial_minimum_train_size

    @property
    def initial_minimum_test_size(self):
        return self.__initial_minimum_test_size

    @property
    def initial_maximum_train_size(self):
        return self.__initial_maximum_train_size

    @property
    def initial_maximum_test_size(self):
        return self.__initial_maximum_test_size

    def change_train_test_timeframes_num(self, minimum_train_size: float = 0.05, maximum_train_size: float = 0.4,
                                         minimum_test_size: float = 0.15, maximum_test_size: float = 0.7, ):

        self.change_train_timeframes_num(minimum_train_size, maximum_train_size)
        self.change_test_timeframes_num(minimum_test_size, maximum_test_size)

    def change_train_timeframes_num(self, minimum_train_size: float = 0.05, maximum_train_size: float = 0.4, ):
        self.minimum_train_timeframes_num = int(self.train_timeframes_num * minimum_train_size)
        self.maximum_train_timeframes_num = int(self.train_timeframes_num * maximum_train_size)

        self.train_minute_timeframes_series = pd.date_range(start=self.start_datetime,
                                                            end=self.all_period_timeframes[self.train_timeframes_num],
                                                            freq=convert_timeframe_to_freq('1m')).to_series()

    def change_test_timeframes_num(self, minimum_test_size: float = 0.15, maximum_test_size: float = 0.7, ):
        self.minimum_test_timeframes_num = int(self.test_timeframes_num * minimum_test_size)
        self.maximum_test_timeframes_num = int(self.test_timeframes_num * maximum_test_size)

        self.test_minute_timeframes_series = pd.date_range(start=self.all_period_timeframes[self.train_timeframes_num],
                                                           end=self.end_datetime,
                                                           freq=convert_timeframe_to_freq('1m')).to_series()

    def get_ohlcv_df(self, _start_datetime, _end_datetime, symbol_pair, market):
        _ohlcv_df = self.fetcher.resample_to_timeframe(
            table_name=self.fetcher.prepare_table_name(market=market, symbol_pair=symbol_pair),
            start=_start_datetime.replace(tzinfo=timezone.utc),
            end=_end_datetime.replace(tzinfo=timezone.utc),
            to_timeframe=self.timeframe,
            origin="end",
            use_cols=Constants.ohlcv_cols,
            use_dtypes=Constants.ohlcv_dtypes,
            open_time_index=True,
            cached=True)
        return _ohlcv_df

    def _get_random_start_end(self, minute_timeframes_series):
        return tuple(minute_timeframes_series.sample(n=2, random_state=self.seed))

    def _get_random_start(self, minute_timeframes_series: Union[pd.DataFrame, None]):
        return minute_timeframes_series.sample(n=2, random_state=self.seed)[random.randint(0, 1)]

    def _get_random_period(self, period_type='train'):
        if period_type == 'train':
            minute_timeframes = self.train_minute_timeframes_series
            minimum_timeframes_num = self.minimum_train_timeframes_num
            maximum_timeframes_num = self.maximum_train_timeframes_num
        else:
            minute_timeframes = self.test_minute_timeframes_series
            minimum_timeframes_num = self.minimum_test_timeframes_num
            maximum_timeframes_num = self.maximum_test_timeframes_num

        done = False
        random_start_datetime, random_end_datetime = None, None
        period_timeframes_len = None
        while not done:
            random_start_datetime = self._get_random_start(minute_timeframes)
            timedelta_timeframes = random.randint(minimum_timeframes_num, maximum_timeframes_num)
            timedelta_kwargs = get_timedelta_kwargs(f'{timedelta_timeframes * Constants.binsizes[self.timeframe]}m',
                                                    current_timeframe=self.timeframe)
            random_end_datetime = random_start_datetime + relativedelta(**timedelta_kwargs)
            if random_end_datetime <= minute_timeframes[-1]:
                period_timeframes_len = len(pd.date_range(start=random_start_datetime,
                                                          end=random_end_datetime,
                                                          freq=convert_timeframe_to_freq(self.timeframe)))
                done = True
        if self.verbose:
            msg = (f"{self.__class__.__name__}: Period {period_type.upper()} timeframes: {period_timeframes_len}, "
                   f"Random period: {random_start_datetime} - {random_end_datetime}\n")
            logger.info(msg)
        return random_start_datetime, random_end_datetime

    def _get_avg_period_len(self, period_type):
        if period_type == 'train':
            minimum_timeframes_num = self.minimum_train_timeframes_num
            maximum_timeframes_num = self.maximum_train_timeframes_num
        else:
            minimum_timeframes_num = self.minimum_test_timeframes_num
            maximum_timeframes_num = self.maximum_test_timeframes_num
        avg_period_len = int((minimum_timeframes_num + maximum_timeframes_num) // 2)
        return avg_period_len

    def _prepare_episodes_start_end_lst(self, num_episodes: int, period_type: str = 'train') -> List[tuple]:
        def get_shifted_range(shifted_minute_ix):
            return pd.Series(index=pd.date_range(start=minute_timeframes[shifted_minute_ix], end=minute_timeframes[-1],
                                                 freq=convert_timeframe_to_freq(self.timeframe)), dtype=int)

        if period_type == 'train':
            minute_timeframes = self.train_minute_timeframes_series
            minimum_timeframes_num = self.minimum_train_timeframes_num
            maximum_timeframes_num = self.maximum_train_timeframes_num
        else:
            minute_timeframes = self.test_minute_timeframes_series
            minimum_timeframes_num = self.minimum_test_timeframes_num
            maximum_timeframes_num = self.maximum_test_timeframes_num

        # shifts = min(max(1, int(num_episodes // Constants.binsizes[self.timeframe])),
        #              Constants.binsizes[self.timeframe])

        finished = False
        ix = 0
        selected_period = get_shifted_range(ix)

        """ 
        if q-ty of current_total_timeframes (all minute shifts) greater
        than maximum_total_timeframes_needed (all num_episodes)
            
        """
        current_total_timeframes = selected_period.shape[0] * Constants.binsizes[self.timeframe]
        maximum_total_timeframes_needed = maximum_timeframes_num * num_episodes
        if current_total_timeframes > maximum_total_timeframes_needed:
            n_shifted_episodes = min(num_episodes, int(round_up(selected_period.shape[0] / maximum_timeframes_num, 0)))
            # n_shifted_episodes = selected_period.shape[0] / maximum_timeframes_num
            shifts = round_up(num_episodes / n_shifted_episodes, 0)
        else:
            shifts = Constants.binsizes[self.timeframe]
            n_shifted_episodes = int(round_up(num_episodes / shifts, 0))

        episodes_start_end_lst: list = []
        """ one_shift_start_end_lst to reverse each shift """
        one_shift_start_end_lst: list = []
        while not finished:
            selected_period_episodes_len = int(selected_period.shape[0] / n_shifted_episodes)

            if selected_period_episodes_len < maximum_timeframes_num:
                start_offset = maximum_timeframes_num - selected_period_episodes_len
            else:
                start_offset = 0

            msg = (f"{self.__class__.__name__}: shift = +{ix}: {selected_period_episodes_len} < {maximum_timeframes_num} "
                   f"-> start_offset = +{start_offset}")
            logger.info(msg)

            _end_datetime = selected_period.index[-1]

            for episode_ix in range(n_shifted_episodes):
                done = False
                _start_datetime = None
                while not done:
                    timedelta_timeframes = random.randint(minimum_timeframes_num, maximum_timeframes_num)
                    timedelta_kwargs = get_timedelta_kwargs(
                        f'{timedelta_timeframes * Constants.binsizes[self.timeframe]}m',
                        current_timeframe=self.timeframe)
                    _start_datetime = _end_datetime - relativedelta(**timedelta_kwargs)
                    if selected_period[:_start_datetime].shape[0] >= minimum_timeframes_num:
                        if _start_datetime >= minute_timeframes[0]:
                            done = True
                    else:
                        done = True
                if selected_period[:_start_datetime].shape[0] < minimum_timeframes_num:
                    break

                one_shift_start_end_lst.append((_start_datetime, _end_datetime))
                if start_offset:
                    timedelta_kwargs = get_timedelta_kwargs(
                        f'{start_offset * Constants.binsizes[self.timeframe]}m',
                        current_timeframe=self.timeframe)
                    _end_datetime = _start_datetime + relativedelta(**timedelta_kwargs)
                else:
                    _end_datetime = _start_datetime

            one_shift_start_end_lst.reverse()
            episodes_start_end_lst.extend(one_shift_start_end_lst)
            one_shift_start_end_lst.clear()
            if len(episodes_start_end_lst) < num_episodes - 1:
                selected_period = get_shifted_range(ix)
                ix += 1
                if ix == shifts - 1:
                    n_shifted_episodes = num_episodes - len(episodes_start_end_lst)
            else:
                finished = True

        return episodes_start_end_lst

    def prepare_n_episodes_lst(self, period_type='train', n_episodes: Union[str, int] = 'auto'):
        if period_type == 'train':
            timeframes_num = self.train_timeframes_num
        else:
            timeframes_num = self.test_timeframes_num

        if n_episodes == 'auto':
            avg_period_len = self._get_avg_period_len(period_type)
            """ 
            Calculate number of periods based on timeframes_num and 
            timeframe binsize (qty of minutes in binsize) 
            """
            num_episodes = int((timeframes_num * Constants.binsizes.get(self.timeframe)) // avg_period_len)
        else:
            num_episodes = n_episodes

        return self._prepare_episodes_start_end_lst(num_episodes, period_type)

    def get_random_ohlcv_df(self):
        start_datetime, end_datetime = self._get_random_period()
        logger.debug(
            f"{self.__class__.__name__}: Get OHLCV data with start_datetime - end_datetime': {start_datetime} - {end_datetime}")
        _ohlcv_df = self.get_ohlcv_df(start_datetime, end_datetime, symbol_pair=self.symbol_pair, market=self.market)
        return _ohlcv_df


class IndicatorProcessor(ProcessorBase):
    def __init__(self, start_datetime, end_datetime, timeframe, discretization, symbol_pair='BTCUSDT', market='spot',
                 minimum_train_size: float = 0.05, maximum_train_size: float = 0.4, minimum_test_size: float = 0.15,
                 maximum_test_size: float = 0.7, test_size: float = 0.2, verbose: int = 0, seed=42):
        super().__init__(start_datetime, end_datetime, timeframe, discretization, symbol_pair, market,
                         minimum_train_size, maximum_train_size, minimum_test_size, maximum_test_size, test_size,
                         verbose, seed)
        logger.debug(f"{self.__class__.__name__}: Initialize LoadDbIndicators")
        self.loaded_indicators = LoadDbIndicators(self.start_datetime,
                                                  self.end_datetime,
                                                  symbol_pairs=[symbol_pair, ],
                                                  market='spot',
                                                  timeframe=self.timeframe,
                                                  discretization=self.discretization,
                                                  )
        self.initialized_full_period = False

    def get_ohlcv_and_indicators_sample(self, timedelta='1d', index_type='target_time'):
        start_datetime = self.start_datetime
        _timedelta_kwargs = get_timedelta_kwargs(timedelta, current_timeframe=self.timeframe)
        end_datetime = start_datetime + relativedelta(**_timedelta_kwargs)
        _ohlcv_df, _indicators_df = self.get_ohlcv_and_indicators(start_datetime, end_datetime, index_type)
        return _ohlcv_df, _indicators_df

    def get_indicators_df(self, _start_datetime, _end_datetime, index_type='target_time'):
        logger.debug(f"{self.__class__.__name__}: Set new period': {_start_datetime} - {_end_datetime}")
        self.loaded_indicators.set_new_period(_start_datetime, _end_datetime, index_type)
        logger.debug(
            f"\n{self.__class__.__name__}: Get indicator_df with start_datetime - end_datetime: {_start_datetime} - {_end_datetime}\n")
        _indicators_df = self.loaded_indicators.get_data_df(index_type)
        return _indicators_df

    def check_cache(self, index_type):
        if not self.initialized_full_period:
            logger.debug(
                f"{self.__class__.__name__}: Preload indicator data for full period, "
                f"start_datetime - end_datetime': {self.start_datetime} - {self.end_datetime}\n")
            _ = self.get_indicators_df(self.start_datetime, self.end_datetime, index_type=index_type)
            self.initialized_full_period = True

    def get_ohlcv_and_indicators(self, start_datetime, end_datetime, index_type='target_time'):
        logger.debug(
            f"\n{self.__class__.__name__}: Get OHLCV data with start_datetime - end_datetime: {start_datetime} - {end_datetime}")
        _ohlcv_df = self.get_ohlcv_df(start_datetime, end_datetime, symbol_pair=self.symbol_pair, market=self.market)
        logger.debug(
            f"{self.__class__.__name__}: load indicator data with start_datetime - end_datetime: {start_datetime} - {end_datetime}\n")
        _indicators_df = self.get_indicators_df(start_datetime, end_datetime, index_type=index_type)

        if _ohlcv_df.shape[0] != _indicators_df.shape[0]:
            logger.debug(
                f"{self.__class__.__name__}: ohlcv_df.shape = {_ohlcv_df.shape}, indicators_df.shape = {_indicators_df.shape}")

            logger.debug(f"{self.__class__.__name__}: OHLCV.shape: \n{_ohlcv_df.shape}")
            logger.debug(f"{self.__class__.__name__}: OHLCV data head: \n{_ohlcv_df.head(5).to_string()}\n")
            logger.debug(f"{self.__class__.__name__}: OHLCV data tail: \n{_ohlcv_df.tail(5).to_string()}\n")

            logger.debug(f"{self.__class__.__name__}: Indicators.shape: \n{_indicators_df.shape}")
            logger.debug(f"{self.__class__.__name__}: Indicators data head: \n{_indicators_df.head(5).to_string()}\n")
            logger.debug(f"{self.__class__.__name__}: Indicators data tail: \n{_indicators_df.tail(5).to_string()}\n")

            sys.exit('Error: Check data_processor, length of data is not equal!')

        assert _ohlcv_df.index[0] == _indicators_df.index[0], "Error: ohlcv.index[0] != _indicators_df.index[0]"
        assert _ohlcv_df.index[-1] == _indicators_df.index[-1], "Error: ohlcv.index[-1] != _indicators_df.index[-1]"
        return _ohlcv_df, _indicators_df

    def get_random_ohlcv_and_indicators(self, index_type='target_time', period_type='train'):
        self.check_cache(index_type)

        msg = f"{self.__class__.__name__}: {period_type.upper()} pool timeframes: "
        if period_type == 'train':
            msg = (f"{msg} {self.train_timeframes_num}, "
                   f"Pool period: {self.train_minute_timeframes_series[0]} - {self.train_minute_timeframes_series[-1]}")
        else:
            msg = (f"{msg} {self.test_timeframes_num}, "
                   f"Pool period: {self.test_minute_timeframes_series[0]} - {self.test_minute_timeframes_series[-1]}")

        start_datetime, end_datetime = self._get_random_period(period_type=period_type)
        msg = f"{msg}, Random period: {start_datetime} - {end_datetime}"
        logger.debug(msg)
        _ohlcv_df, _indicators_df = self.get_ohlcv_and_indicators(start_datetime, end_datetime, index_type)
        return _ohlcv_df, _indicators_df

    def get_n_episodes_start_end_lst(self,
                                     index_type='target_time',
                                     period_type='train',
                                     n_episodes: Union[str, int] = 'auto'):
        self.check_cache(index_type)
        start_end_episodes_lst = self.prepare_n_episodes_lst(period_type=period_type, n_episodes=n_episodes)
        return start_end_episodes_lst

    def get_n_episodes_ohlcv_and_indicators(self,
                                            index_type='target_time',
                                            period_type='train',
                                            n_episodes: Union[str, int] = 'auto'):

        episodes_lst: list = []
        msg = f"{self.__class__.__name__}: {period_type.upper()} pool timeframes: "
        if period_type == 'train':
            msg = (f"{msg} {self.train_timeframes_num}, "
                   f"Pool period: {self.train_minute_timeframes_series[0]} - {self.train_minute_timeframes_series[-1]}")
        else:
            msg = (f"{msg} {self.test_timeframes_num}, "
                   f"Pool period: {self.test_minute_timeframes_series[0]} - {self.test_minute_timeframes_series[-1]}")
        logger.debug(msg)

        start_end_episodes_lst = self.get_n_episodes_start_end_lst(index_type, period_type, n_episodes)
        for (start_datetime, end_datetime) in start_end_episodes_lst:
            msg = f"{self.__class__.__name__}: period: {start_datetime} - {end_datetime}"
            logger.debug(msg)
            _ohlcv_df, _indicators_df = self.get_ohlcv_and_indicators(start_datetime, end_datetime, index_type)
            episodes_lst.append((_ohlcv_df, _indicators_df))
        return episodes_lst
