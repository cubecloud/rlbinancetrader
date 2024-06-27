import datetime
import logging
from dateutil.relativedelta import relativedelta
from dbbinance.fetcher.datautils import get_timedelta_kwargs
from dbbinance.fetcher.datafetcher import ceil_time, floor_time

from indicators import LoadDbIndicators

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('test_loadindicators.log')
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

_timedelta_kwargs = get_timedelta_kwargs(_show_period, current_timeframe=_timeframe)
_end_datetime = ceil_time(datetime.datetime.utcnow(), '1m')
_start_datetime = _end_datetime - relativedelta(**_timedelta_kwargs)
_start_datetime = floor_time(_start_datetime, '1m')
logger.debug(f"Preload indicator data with start_datetime-end_datetime': {_start_datetime} - {_end_datetime}")

loaded_ind = LoadDbIndicators(_start_datetime,
                              _end_datetime,
                              symbol_pairs=['BTCUSDT', ],
                              market='spot',
                              timeframe=_timeframe,
                              discretization=_discretization,
                              )

_df = loaded_ind.get_data_df('prediction_time')
logger.debug(f"Indicator data 'prediction_time': \n{_df.to_string()}\n")
_df = loaded_ind.get_data_df('target_time')
logger.debug(f"Indicator data 'target_time': \n{_df.to_string()}\n")

_timedelta_kwargs = get_timedelta_kwargs(_show_period, current_timeframe=_timeframe)
_end_datetime = ceil_time(datetime.datetime.utcnow(), '1m') - relativedelta(**_timedelta_kwargs)
_start_datetime = _end_datetime - relativedelta(**_timedelta_kwargs)
_start_datetime = floor_time(_start_datetime, '1m')
logger.debug(f"Set end_datetime to the 'show_period' _back_: {_start_datetime} - {_end_datetime}")

loaded_ind = LoadDbIndicators(_start_datetime,
                              _end_datetime,
                              symbol_pairs=['BTCUSDT', ],
                              market='spot',
                              timeframe=_timeframe,
                              discretization=_discretization,
                              )

_df = loaded_ind.get_data_df('prediction_time')
logger.debug(f"Indicator data 'prediction_time': \n{_df.to_string()}\n")
_df = loaded_ind.get_data_df('target_time')
logger.debug(f"Indicator data 'target_time': \n{_df.to_string()}\n")

_timedelta_kwargs = get_timedelta_kwargs(_show_period, current_timeframe=_timeframe)
_end_datetime = ceil_time(datetime.datetime.utcnow(), '1m')
_start_datetime = _end_datetime - relativedelta(**_timedelta_kwargs)
_end_datetime = ceil_time(datetime.datetime.utcnow(), '1m') + relativedelta(**_timedelta_kwargs)
_start_datetime = floor_time(_start_datetime, '1m')

logger.debug(f"Set end_datetime to the 'show_period' in _future_: {_start_datetime} - {_end_datetime}")

loaded_ind = LoadDbIndicators(_start_datetime,
                              _end_datetime,
                              symbol_pairs=['BTCUSDT', ],
                              market='spot',
                              timeframe=_timeframe,
                              discretization=_discretization,
                              )

_df = loaded_ind.get_data_df('prediction_time')
logger.debug(f"Indicator data 'prediction_time': \n{_df.to_string()}\n")
_df = loaded_ind.get_data_df('target_time')
logger.debug(f"Indicator data 'target_time': \n{_df.to_string()}\n")
