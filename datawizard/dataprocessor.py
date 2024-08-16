import random
import sys
import logging

import pandas as pd
from datetime import timezone
from dateutil.relativedelta import relativedelta

from dbbinance.fetcher.getfetcher import get_datafetcher
from dbbinance.fetcher.datautils import convert_timeframe_to_freq
from dbbinance.fetcher.datautils import get_timedelta_kwargs
from dbbinance.fetcher.constants import Constants

from indicators import LoadDbIndicators

__version__ = 0.05

logger = logging.getLogger()


class ProcessorBase:
    max_timesteps = 365 * Constants.binsizes['1d']

    def __init__(self, start_datetime, end_datetime, timeframe, discretization,
                 symbol_pair='BTCUSDT', market='spot',
                 minimum_train_size: float = 0.05,
                 maximum_train_size: float = 0.4,
                 minimum_test_size: float = 0.15,
                 maximum_test_size: float = 0.7,
                 test_size: float = 0.2,
                 verbose: int = 0):
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.timeframe = timeframe
        self.discretization = discretization
        self.market = market
        self.symbol_pair = symbol_pair
        self.verbose = verbose
        self.fetcher = get_datafetcher()

        self.max_timesteps = self.max_timesteps // Constants.binsizes[self.timeframe]
        """ get all data from start_datetime until end_datetime to fill datafetcher with RAW data cache """
        self.ohlcv_df = self.get_ohlcv_df(self.start_datetime, self.end_datetime, self.symbol_pair, self.market)
        all_period_timeframes = pd.date_range(start=self.start_datetime, end=self.end_datetime,
                                              freq=convert_timeframe_to_freq(self.timeframe)).to_series().to_list()

        self.train_timeframes_num = int(len(all_period_timeframes) * (1 - test_size))
        self.test_timeframes_num = len(all_period_timeframes) - self.train_timeframes_num

        msg = (f"FULL pool timeframes: {self.train_timeframes_num}, "
               f"Pool period: {all_period_timeframes[0]} - {all_period_timeframes[self.train_timeframes_num]}")
        logger.info(msg)
        assert self.train_timeframes_num >= 50, f'Error: train_timeframes_num is to LOW {self.train_timeframes_num}'
        assert self.test_timeframes_num >= 50, f'Error: test_timeframes_num is to LOW {self.test_timeframes_num}'
        self.minimum_train_timeframes_num = int(self.train_timeframes_num * minimum_train_size)
        self.maximum_train_timeframes_num = int(self.train_timeframes_num * maximum_train_size)
        self.minimum_test_timeframes_num = int(self.test_timeframes_num * minimum_test_size)
        self.maximum_test_timeframes_num = int(self.test_timeframes_num * maximum_test_size)

        self.train_minute_timeframes_series = pd.date_range(start=self.start_datetime,
                                                            end=all_period_timeframes[self.train_timeframes_num],
                                                            freq=convert_timeframe_to_freq('1m')).to_series()
        self.test_minute_timeframes_series = pd.date_range(start=all_period_timeframes[self.train_timeframes_num],
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
        return tuple(minute_timeframes_series.sample(n=2))

    def _get_random_start(self, minute_timeframes_series):
        return minute_timeframes_series.sample(n=2)[0]

    def _get_random_period(self, period_type='train'):
        minute_timeframes = self.train_minute_timeframes_series
        minimum_timeframes_num = self.minimum_train_timeframes_num
        maximum_timeframes_num = self.maximum_train_timeframes_num
        if period_type == 'test':
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
                   f"Random period: {random_end_datetime} - {random_start_datetime}\n")
            logger.info(msg)
        return random_start_datetime, random_end_datetime

    def get_random_ohlcv_df(self):
        start_datetime, end_datetime = self._get_random_period()
        logger.debug(
            f"{self.__class__.__name__}: Get OHLCV data with start_datetime - end_datetime': {start_datetime} - {end_datetime}")
        _ohlcv_df = self.get_ohlcv_df(start_datetime, end_datetime, symbol_pair=self.symbol_pair, market=self.market)
        return _ohlcv_df


class IndicatorProcessor(ProcessorBase):
    def __init__(self, start_datetime, end_datetime, timeframe, discretization, symbol_pair='BTCUSDT', market='spot',
                 minimum_train_size: float = 0.05, maximum_train_size: float = 0.4, minimum_test_size: float = 0.15,
                 maximum_test_size: float = 0.7, test_size: float = 0.2, verbose: int = 0):
        super().__init__(start_datetime, end_datetime, timeframe, discretization, symbol_pair, market,
                         minimum_train_size, maximum_train_size, minimum_test_size, maximum_test_size, test_size,
                         verbose)
        logger.debug(f"{self.__class__.__name__}: Initialize LoadDbIndicators")
        self.loaded_indicators = LoadDbIndicators(self.start_datetime,
                                                  self.end_datetime,
                                                  symbol_pairs=[symbol_pair, ],
                                                  market='spot',
                                                  timeframe=self.timeframe,
                                                  discretization=self.discretization,
                                                  )
        self.initialized_full_period = False

    def get_ohlcv_and_indicators_sample(self, index_type='target_time'):
        start_datetime = self.start_datetime
        _timedelta_kwargs = get_timedelta_kwargs('1d', current_timeframe=self.timeframe)
        end_datetime = start_datetime + relativedelta(**_timedelta_kwargs)
        _ohlcv_df, _indicators_df = self.get_ohlcv_and_indicators(start_datetime, end_datetime, index_type)
        return _ohlcv_df, _indicators_df

    def get_indicators_df(self, _start_datetime, _end_datetime, index_type='target_time'):
        logger.debug(f"{self.__class__.__name__}: Set new period': {_start_datetime} - {_end_datetime}")
        self.loaded_indicators.set_new_period(_start_datetime, _end_datetime, index_type)
        logger.debug(
            f"\n{self.__class__.__name__}: Get indicator_df with start_datetime - end_datetime': {_start_datetime} - {_end_datetime}\n")
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
            f"\n{self.__class__.__name__}: Get OHLCV data with start_datetime - end_datetime': {start_datetime} - {end_datetime}")
        _ohlcv_df = self.get_ohlcv_df(start_datetime, end_datetime, symbol_pair=self.symbol_pair, market=self.market)
        logger.debug(
            f"{self.__class__.__name__}: load indicator data with start_datetime - end_datetime': {start_datetime} - {end_datetime}\n")
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
