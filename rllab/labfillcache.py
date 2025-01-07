import time
from tqdm import tqdm
from typing import Union
from dbbinance.fetcher import MpCacheManager
from dbbinance.fetcher import PERCacheManager
from datawizard.dataprocessor import IndicatorProcessor
import multiprocessing as mp

logger = mp.get_logger()

__version__ = 0.003


def worker_fill_cache(data_processor_obj: Union[IndicatorProcessor],
                      cache_obj: Union[PERCacheManager],
                      env_kwargs: dict,
                      ep_start_end_lst: list):
    def get_new_ohlcv_and_indicators(start_datetime,
                                     end_datetime,
                                     index_type: str = 'target_time'):

        _ohlcv_df, _indicators_df = data_processor_obj.get_ohlcv_and_indicators(start_datetime=start_datetime,
                                                                                end_datetime=end_datetime,
                                                                                index_type=index_type)

        if _ohlcv_df.shape[0] != _indicators_df.shape[0]:
            msg = (f"{__name__}: ohlcv_df.shape = {_ohlcv_df.shape}, "
                   f"indicators_df.shape = {_indicators_df.shape}")
            logger.debug(msg)
            raise "Error: Check data_processor, length of data is not equal!"
        return _ohlcv_df, _indicators_df

    for (_start, _end) in ep_start_end_lst:
        ohlcv_df, indicators_df = get_new_ohlcv_and_indicators(_start,
                                                               _end,
                                                               index_type=env_kwargs['index_type'])
        if tuple((_start, _end)) != tuple((ohlcv_df.index[0], ohlcv_df.index[-1])):
            raise f"Error: ({_start}, {_end} != ({ohlcv_df.index[0]}, {ohlcv_df.index[-1]})"
        cm_key = tuple((ohlcv_df.index[0], ohlcv_df.index[-1]))
        cache_obj.update_cache(key=cm_key, value=(ohlcv_df, indicators_df))


# def mp_fill_cache(env_kwargs: dict, n_envs: Union[str, int] = 'auto', seed: int = 42,
#                   port: Union[int, None] = None, start_host: bool = True):
#     def pbar_updater(cache_obj: Union[MpCacheManager], ):
#         pbar = tqdm(total=env_kwargs['stable_cache_data_n'])
#         sl_time = 1.3
#         while pbar.n < env_kwargs['stable_cache_data_n']:
#             pbar.set_description(f"{env_kwargs['use_period'].upper()} DataFrames loaded")
#             time.sleep(sl_time)
#             pbar.n = len(cache_obj.keys())
#             pbar.refresh()
#         pbar.close()
#
#     def mp_cache_server(port, use_period, start_host):
#
#         if port is None:
#             if use_period == 'train':
#                 cache_kwargs: dict = {'port': 5005, 'max_memory_gb': 6}
#             elif use_period == 'test':
#                 cache_kwargs: dict = {'port': 5006}
#             else:
#                 cache_kwargs: dict = {'port': 5007}
#         else:
#             cache_kwargs = {'port': port}
#
#         if not MpCacheManager.is_server_running(port=port):
#
#             cache_server = MpCacheManager(start_host=start_host,
#                                           unique_name=use_period,
#                                           **cache_kwargs)
#         else:
#             cache_server = MpCacheManager(max_memory_gb=6,
#                                           start_host=False,
#                                           port=port,
#                                           unique_name=use_period)
#         return cache_server
#
#     if env_kwargs['use_period'] == 'train':
#         if port is None:
#             port = 5005
#         if not MpCacheManager.is_server_running(port=port):
#             mp_cache_server = MpCacheManager(max_memory_gb=6,
#                                              start_host=start_host,
#                                              port=port,
#                                              unique_name='train')
#         else:
#             mp_cache_server = MpCacheManager(max_memory_gb=6,
#                                              start_host=False,
#                                              port=port,
#                                              unique_name='train')
#     elif env_kwargs['use_period'] == 'test':
#         if port is None:
#             port = 5006
#         if not MpCacheManager.is_server_running(port=port):
#             mp_cache_server = MpCacheManager(start_host=start_host,
#                                              port=port,
#                                              unique_name=env_kwargs['use_period'])
#         else:
#             mp_cache_server = MpCacheManager(start_host=False,
#                                              port=port,
#                                              unique_name=env_kwargs['use_period'])
#     else:
#         if port is None:
#             port = 5007
#         if not MpCacheManager.is_server_running(port=port):
#             mp_cache_server = MpCacheManager(start_host=start_host,
#                                              port=port,
#                                              unique_name=env_kwargs['use_period'])
#
#     """ Get the list of episodes start - end """
#     env_kwargs['data_processor_kwargs'].update({'seed': seed})
#     dp_obj = IndicatorProcessor(**env_kwargs['data_processor_kwargs'])
#     episodes_start_end_lst = dp_obj.get_n_episodes_start_end_lst(index_type=env_kwargs['index_type'],
#                                                                  period_type=env_kwargs['use_period'],
#                                                                  n_episodes=env_kwargs['stable_cache_data_n'])
#
#     logger.info(f'{self.__class__.__name__}: start-end list contains: #{len(episodes_start_end_lst)}')
#
#     min_start = min(episodes_start_end_lst, key=lambda x: x[0])[0]
#     max_end = max(episodes_start_end_lst, key=lambda x: x[1])[1]
#
#     logger.info(f'{self.__class__.__name__}: min_start / max_end {min_start}/{max_end}')
#
#     if n_envs == 'auto':
#         n_envs = min(len(episodes_start_end_lst), mp.cpu_count() - 2 or 1)
#
#     env_start_end_lst: list = []
#
#     n_episodes_per_env = int(round_up(max(1., len(episodes_start_end_lst) / n_envs), 0))
#     indices = np.arange(0, len(episodes_start_end_lst) + 1, n_episodes_per_env)
#     for idx in indices:
#         ep_start_end = episodes_start_end_lst[idx: min(idx + n_episodes_per_env, len(episodes_start_end_lst) + 1)]
#         env_start_end_lst.append(ep_start_end)
#     """ Get the list of episodes start - end """
#
#     job_lst: list = []
#     for ix, start_end_lst in zip(range(len(env_start_end_lst)), env_start_end_lst):
#         job_lst.append(mp.Process(target=worker_fill_cache,
#                                   args=(dp_obj, mp_cache_server, env_kwargs, start_end_lst),
#                                   name=f'fillcache_{ix}'))
#         job_lst[-1].start()
#
#     job_lst.append(mp.Process(target=pbar_updater, args=(mp_cache_server,), name=f'pbar'))
#     job_lst[-1].start()
#     for job in job_lst:
#         job.join()
#
#     mp_cache_server.update_cache_size()
#     cache_size_mb = round(mp_cache_server.current_memory_usage / (1024 * 1024), 1)
#     logger.info(
#         f'{self.__class__.__name__}: {env_kwargs["use_period"].upper()} cache_size = {cache_size_mb} Mb')
#
#     return mp_cache_server
