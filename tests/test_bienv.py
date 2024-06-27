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

from indicators import LoadDbIndicators

from binanceenv.bienv import IndicatorsObsSpaceContinuous

__version__ = 0.02

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
_end_datetime = floor_time(datetime.datetime.utcnow(), '1m')

_start_datetime = _end_datetime - relativedelta(**_timedelta_kwargs)
_start_datetime = floor_time(_start_datetime, '1m')
logger.debug(f"Preload indicator data with start_datetime-end_datetime': {_start_datetime} - {_end_datetime}")

fetcher = DataFetcher(host=ConfigPostgreSQL.HOST,
                      database=ConfigPostgreSQL.DATABASE,
                      user=ConfigPostgreSQL.USER,
                      password=ConfigPostgreSQL.PASSWORD,
                      binance_api_key='dummy',
                      binance_api_secret='dummy',
                      )

ohlcv_df = fetcher.resample_to_timeframe(table_name="spot_data_btcusdt_1m",
                                         start=_start_datetime.replace(tzinfo=timezone.utc),
                                         end=_end_datetime.replace(tzinfo=timezone.utc),
                                         to_timeframe=_timeframe,
                                         origin="end",
                                         use_cols=Constants.ohlcv_cols,
                                         use_dtypes=Constants.ohlcv_dtypes,
                                         open_time_index=True,
                                         cached=True)

logger.debug(f"OHLCV data head: \n{ohlcv_df.head(5).to_string()}\n")
logger.debug(f"OHLCV data tail: \n{ohlcv_df.tail(5).to_string()}\n")

loaded_ind = LoadDbIndicators(_start_datetime,
                              _end_datetime,
                              symbol_pairs=['BTCUSDT', ],
                              market='spot',
                              timeframe=_timeframe,
                              discretization=_discretization,
                              )

ind_df = loaded_ind.get_data_df('prediction_time')
logger.debug(f"Indicator data 'prediction_time': \n{ind_df.to_string()}\n")

ind_df = loaded_ind.get_data_df('target_time')
logger.debug(f"Indicator data 'target_time': \n{ind_df.to_string()}\n")

_df = pd.concat([ohlcv_df, ind_df], axis=1)
logger.debug(f"OHLCV+indicators': \n{_df.head(100).to_string()}\n")
