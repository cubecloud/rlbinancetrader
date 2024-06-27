
import logging
import datetime
from datetime import timezone

from dbbinance.fetcher.datafetcher import DataUpdater
from dbbinance.fetcher.datafetcher import DataFetcher
from dbbinance.fetcher.datafetcher import Constants
from dbbinance.config.configpostgresql import ConfigPostgreSQL
from dbbinance.config.configbinance import ConfigBinance
from dbbinance.fetcher.datafetcher import floor_time

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C

from binanceenv.bienv import BinanceEnv

__version__ = 0.031

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('rltrader.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logging.getLogger('apscheduler').setLevel(logging.DEBUG)

fetcher = DataFetcher(host=ConfigPostgreSQL.HOST,
                      database=ConfigPostgreSQL.DATABASE,
                      user=ConfigPostgreSQL.USER,
                      password=ConfigPostgreSQL.PASSWORD,
                      binance_api_key='dummy',
                      binance_api_secret='dummy',
                      )

""" Train period - from 01 Aug 2018 until now - 26 weeks """
start_datetime = datetime.datetime.strptime('01 Aug 2018', '%d %b %Y').replace(tzinfo=timezone.utc)
end_datetime = floor_time(datetime.datetime.now(timezone.utc) - datetime.timedelta(weeks=26))
timeframe = '1h'

print(f'Start datetime - end datetime: {start_datetime} - {end_datetime}\n')
_data_df = fetcher.resample_to_timeframe(table_name="spot_data_btcusdt_1m",
                                         start=start_datetime,
                                         end=end_datetime,
                                         to_timeframe=timeframe,
                                         origin="start",
                                         use_cols=Constants.ohlcv_cols,
                                         use_dtypes=Constants.ohlcv_dtypes,
                                         open_time_index=False,
                                         cached=True)

print(_data_df.head(10).to_string())
print('Length:', _data_df.shape[0])
env = BinanceEnv(ohlcv_df=_data_df)
check_env(env)  # It will check your custom environment and output additional warnings if needed
env.action_space.seed(42)
observation, info = env.reset(seed=42)
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    if done:
      obs = vec_env.reset()

env.close()
