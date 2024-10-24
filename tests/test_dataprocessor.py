import logging
import datetime
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

__version__ = 0.025

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
_show_period = '4h'
_gap_period = '1d'

# _timedelta_kwargs = get_timedelta_kwargs(_show_period, current_timeframe=_timeframe)
# _end_datetime = floor_time(datetime.datetime.utcnow(), '1m')
#
# _start_datetime = _end_datetime - relativedelta(**_timedelta_kwargs)
# _start_datetime = floor_time(_start_datetime, '1m')

_start_datetime = datetime.datetime.strptime('2023-07-24 12:00:00', Constants.default_datetime_format)

_timedelta_kwargs = get_timedelta_kwargs(_gap_period, current_timeframe=_timeframe)
_end_datetime = floor_time(datetime.datetime.utcnow(), '1m')
_end_datetime = _end_datetime - relativedelta(**_timedelta_kwargs)

msg = (f"{__name__}: Preload indicator data with start_datetime-end_datetime: {_start_datetime} - {_end_datetime}, "
       f"timeframe: {_timeframe}, discretization: {_discretization}")
logger.debug(msg)

dp = ProcessorBase(_start_datetime, _end_datetime, _timeframe, _discretization, symbol_pair='BTCUSDT', market='spot')
ohlcv_df = dp.get_random_ohlcv_df()

logger.debug(f"{__name__}: OHLCV.shape: \n{ohlcv_df.shape}")
logger.debug(f"{__name__}: OHLCV data head: \n{ohlcv_df.head(5).to_string()}\n")
logger.debug(f"{__name__}: OHLCV data tail: \n{ohlcv_df.tail(5).to_string()}\n")

msg = (
    f"{__name__}: Load data with IndicatorProcessor start_datetime-end_datetime: {_start_datetime} - {_end_datetime} "
    f"timeframe: {_timeframe}, discretization: {_discretization}\n")
logger.debug(msg)

""" Test rolling """
rolling_frames = 96
s = ((ohlcv_df['high'] + ohlcv_df['low'] + ohlcv_df['close']) / 3) * ohlcv_df['volume']
s = s.rolling(window=rolling_frames, min_periods=1).sum()
s_df = s / ohlcv_df['volume'].rolling(window=rolling_frames, min_periods=1).sum()

logger.debug(f"{__name__}: Rolling test vwap tail: \n{s_df.tail(5).to_string()}\n")

""" Test indicators """
dp = IndicatorProcessor(_start_datetime, _end_datetime, _timeframe, _discretization, symbol_pair='BTCUSDT',
                        market='spot', indicators_sign=True, )
ohlcv_df, indicators_df = dp.get_random_ohlcv_and_indicators()

logger.debug(f"{__name__}: OHLCV.shape: \n{ohlcv_df.shape}")
logger.debug(f"{__name__}: OHLCV data head: \n{ohlcv_df.head(5).to_string()}\n")
logger.debug(f"{__name__}: OHLCV data tail: \n{ohlcv_df.tail(5).to_string()}\n")

logger.debug(f"{__name__}: Indicators.shape: \n{indicators_df.shape}")
logger.debug(f"{__name__}: Indicators data head: \n{indicators_df.head(5).to_string()}\n")
logger.debug(f"{__name__}: Indicators data tail: \n{indicators_df.tail(5).to_string()}\n")

_timeframe = '5m'
_discretization = '15m'
_gap_period = '1d'

_start_datetime = datetime.datetime.strptime('2023-07-24 12:00:00', Constants.default_datetime_format)

_timedelta_kwargs = get_timedelta_kwargs(_gap_period, current_timeframe=_timeframe)
_end_datetime = floor_time(datetime.datetime.utcnow(), '1m')
_end_datetime = _end_datetime - relativedelta(**_timedelta_kwargs)

msg = (
    f"{__name__}: Load data with IndicatorProcessor start_datetime-end_datetime: {_start_datetime} - {_end_datetime} "
    f"timeframe: {_timeframe}, discretization: {_discretization}\n")
logger.debug(msg)

ohlcv_df, indicators_df = dp.get_random_ohlcv_and_indicators()

logger.debug(f"{__name__}: OHLCV.shape: \n{ohlcv_df.shape}")
logger.debug(f"{__name__}: OHLCV data head: \n{ohlcv_df.head(5).to_string()}\n")
logger.debug(f"{__name__}: OHLCV data tail: \n{ohlcv_df.tail(5).to_string()}\n")

logger.debug(f"{__name__}: Indicators.shape: \n{indicators_df.shape}")
logger.debug(f"{__name__}: Indicators data head: \n{indicators_df.head(5).to_string()}\n")
logger.debug(f"{__name__}: Indicators data tail: \n{indicators_df.tail(5).to_string()}\n")



""" Test episodes start-end list """

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
                             indicators_sign=True,
                             )

dp = IndicatorProcessor(**data_processor_kwargs)

episodes_lst = dp.get_n_episodes_ohlcv_and_indicators(index_type='target_time',
                                                      period_type='train',
                                                      n_episodes=10)
for (episode_ohlcv_df, episode_indicator_df) in episodes_lst:
    logger.debug(f"{__name__}: {episode_ohlcv_df.index[0]} - {episode_ohlcv_df.index[-1]}")
