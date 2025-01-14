import random
import pandas as pd
from typing import Union, List, Tuple
from dateutil.relativedelta import relativedelta
from itertools import cycle
from multiprocessing import get_logger

from dbbinance.fetcher.getfetcher import get_datafetcher
from dbbinance.fetcher.datautils import convert_timeframe_to_freq
from dbbinance.fetcher.datautils import get_timedelta_kwargs
from dbbinance.fetcher.constants import Constants
from rllab.labtools import round_up

__version__ = 0.098

logger = get_logger()


def generate_shifts(num_shifts: int) -> List[int]:
    def half_list(half: List[int]) -> List[int]:
        result: list = []
        if len(half) <= 2:
            half.reverse()
            result.extend(half)
        else:
            mid = len(half) // 2
            if mid > 0:
                result.append(half[mid])
                result.extend(half_list(list(range(half[0], half[mid]))))
                result.extend(half_list(list(range(half[mid + 1], half[-1] + 1))))
        return result

    shifts_lst: list = [0]
    mid = num_shifts // 2
    shifts_lst.append(mid)
    first_half = half_list(list(range(1, mid)))
    second_half = half_list(list(range(mid + 1, num_shifts)))
    max_len = min(len(first_half), len(second_half))
    for a, b in zip(first_half, second_half):
        shifts_lst.extend([a, b])
    shifts_lst.extend(first_half[max_len:])
    shifts_lst.extend(second_half[max_len:])
    return shifts_lst


def calculate_optimal_offset(timeframe) -> int:
    return max(int(Constants.binsizes['1d'] / Constants.binsizes[timeframe] / 1.5), 1)


def generate_episodes(dates_range: pd.Series, min_timeframes_per_episode: int, max_timeframes_per_episode: int,
                      n_shifted_episodes: int, offset: int) -> List[Tuple]:
    total_timeframes = dates_range.shape[0]
    episodes = set()

    def generate_n_episodes(n_episodes, start_offset: int = -1):
        current_end_index = total_timeframes - 1
        while n_episodes > 0:
            # Random length of current episode
            episode_length = random.randint(min_timeframes_per_episode, max_timeframes_per_episode)

            start_index = current_end_index - episode_length
            if start_index < 0:
                current_end_index = total_timeframes - int(episode_length // random.randint(2, 10))
                continue

            # checking unique tuple
            new_episode = (dates_range.index[start_index], dates_range.index[current_end_index])
            if new_episode not in episodes:
                episodes.add(new_episode)
                n_episodes -= 1

            # setting new end_index
            current_end_index = start_index + episode_length + start_offset
            if current_end_index <= 0:
                current_end_index = total_timeframes - int(
                    random.randint(min_timeframes_per_episode, max_timeframes_per_episode) // random.randint(2, 10))

    def generate_auto_episodes(start_offset: int = -1):
        current_end_index = total_timeframes - 1
        while True:
            # Random length of current episode

            episode_length = random.randint(min_timeframes_per_episode, max_timeframes_per_episode)

            start_index = current_end_index - episode_length
            if start_index < 0:
                break

            # checking unique tuple
            new_episode = (dates_range.index[start_index], dates_range.index[current_end_index])
            if new_episode not in episodes:
                episodes.add(new_episode)

            # setting new end_index
            current_end_index = start_index + episode_length + start_offset
            if current_end_index <= 0:
                break

    if n_shifted_episodes > 0:
        generate_n_episodes(n_shifted_episodes, offset)
    else:
        generate_auto_episodes(offset)

    return sorted(list(episodes))  # Creating sorted list


def prepare_episodes_start_end_lst(num_episodes: int,
                                   minute_timeframes: pd.Series,  # minute timeframe
                                   min_timeframes_per_episode: int,  # number in chose timeframe
                                   max_timeframes_per_episode: int,  # number in chose timeframe
                                   timeframe: str,  # chose timeframe
                                   offset: Union[int, None] = None
                                   ) -> List[Tuple]:
    def get_shifted_range(shifted_minute_ix):
        return pd.Series(index=pd.date_range(start=minute_timeframes.index[shifted_minute_ix],
                                             end=minute_timeframes.index[-1],
                                             freq=convert_timeframe_to_freq(timeframe)),
                         dtype=int)

    shifts = generate_shifts(Constants.binsizes[timeframe])
    circular_shifts = cycle(shifts)

    """
    if q-ty of current_total_timeframes (all minute shifts) greater
    than maximum_total_timeframes_needed (all num_episodes)

    """
    selected_period = get_shifted_range(shifts[0])
    current_total_timeframes = selected_period.shape[0] * len(shifts)  # ~ total timeframes in all shifts
    """ 
    Logic:
    If num_episodes = 0 -> using 'auto' and calculate optimal _offset_ 
    If num_episodes != 0 and _offset_ == None -> using _offset_ from calculations of n_shifted_episodes
    If num_episodes != 0 and _offset_ is not None -> using _offset_ from args
    """
    maximum_total_timeframes_needed = max_timeframes_per_episode * num_episodes

    if offset is not None:
        if current_total_timeframes > maximum_total_timeframes_needed:
            n_shifted_episodes = min(num_episodes,
                                     int(round_up(selected_period.shape[0] / max_timeframes_per_episode, 0)))
        else:
            n_shifted_episodes = int(round_up(num_episodes / len(shifts), 0))
    else:
        n_shifted_episodes = 0

    if n_shifted_episodes != 0:
        selected_period_episode_len = int(selected_period.shape[0] / n_shifted_episodes)
        if selected_period_episode_len < max_timeframes_per_episode:
            start_offset = -selected_period_episode_len
        else:
            start_offset = 0
    else:
        selected_period_episode_len = None
        start_offset = -calculate_optimal_offset(timeframe)
        num_episodes = 0

    episodes_start_end_lst: List[Tuple] = []

    for current_shift in circular_shifts:
        selected_period = get_shifted_range(current_shift)

        one_shift_episodes = generate_episodes(selected_period,
                                               min_timeframes_per_episode,
                                               max_timeframes_per_episode,
                                               n_shifted_episodes,
                                               start_offset)

        msg = (f"shift = +{current_shift}: {selected_period_episode_len} < {max_timeframes_per_episode} "
               f"-> start_offset = +{start_offset} => {len(one_shift_episodes)}")
        logger.info(msg)
        episodes_start_end_lst += one_shift_episodes

        if len(episodes_start_end_lst) >= num_episodes != 0:
            episodes_start_end_lst = episodes_start_end_lst[:num_episodes]
            break
        elif current_shift == shifts[-1] and n_shifted_episodes == 0 and episodes_start_end_lst:
            logger.info(f"Prepared list with #{len(episodes_start_end_lst)} episodes (start-end)")
            break

    return episodes_start_end_lst

# def prepare_episodes_start_end_lst(num_episodes: int,
#                                    minute_timeframes: pd.Series,  # minute timeframe
#                                    min_timeframes_per_episode: int,  # number in chose timeframe
#                                    max_timeframes_per_episode: int,  # number in chose timeframe
#                                    timeframe: str,  # chose timeframe
#                                    ) -> List[Tuple]:
#     def get_shifted_range(shifted_minute_ix):
#         return pd.Series(index=pd.date_range(start=minute_timeframes[shifted_minute_ix], end=minute_timeframes[-1],
#                                              freq=convert_timeframe_to_freq(timeframe)), dtype=int)
#
#     """
#     if q-ty of current_total_timeframes (all minute shifts) greater
#     than maximum_total_timeframes_needed (all num_episodes)
#     """
#     shifts = generate_shifts(Constants.binsizes[timeframe])
#     circular_shifts = cycle(shifts)
#     current_shift = next(circular_shifts)
#     selected_period = get_shifted_range(current_shift)
#     remaining_episodes = num_episodes
#
#     current_total_timeframes = minute_timeframes.shape[0] * Constants.binsizes[timeframe]
#     maximum_total_timeframes_needed = max_timeframes_per_episode * num_episodes
#     if current_total_timeframes > maximum_total_timeframes_needed:
#         n_shifted_episodes = min(num_episodes,
#                                  int(round_up(minute_timeframes.shape[0] / max_timeframes_per_episode, 0)))
#     else:
#         n_shifted_episodes = int(round_up(num_episodes / len(shifts), 0))
#
#     episodes_start_end_lst: list = []
#     """ one_shift_start_end_lst to reverse each shift """
#     one_shift_start_end_lst: list = []
#
#     while remaining_episodes > 0:
#         selected_period_episodes_len = int(selected_period.shape[0] / n_shifted_episodes)
#
#         if selected_period_episodes_len < max_timeframes_per_episode:
#             start_offset = max_timeframes_per_episode - selected_period_episodes_len
#         else:
#             start_offset = 0
#
#         msg = (f"shift = +{current_shift}: {selected_period_episodes_len} < {max_timeframes_per_episode} "
#                f"-> start_offset = +{start_offset}")
#         logger.info(msg)
#
#         _end_datetime = selected_period.index[-1]
#
#         for episode_ix in range(n_shifted_episodes):
#             done = False
#             _start_datetime = None
#             while not done:
#                 timedelta_timeframes = random.randint(min_timeframes_per_episode, max_timeframes_per_episode)
#                 timedelta_kwargs = get_timedelta_kwargs(
#                     f'{timedelta_timeframes * Constants.binsizes[timeframe]}m',
#                     current_timeframe=timeframe)
#                 _start_datetime = _end_datetime - relativedelta(**timedelta_kwargs)
#                 if selected_period[:_start_datetime].shape[0] >= min_timeframes_per_episode:
#                     if _start_datetime >= minute_timeframes[0]:
#                         done = True
#                 else:
#                     done = True
#             one_shift_start_end_lst.append((_start_datetime, _end_datetime))
#             if selected_period[:_start_datetime].shape[0] < min_timeframes_per_episode:
#                 break
#
#             if start_offset:
#                 timedelta_kwargs = get_timedelta_kwargs(
#                     f'{start_offset * Constants.binsizes[timeframe]}m',
#                     current_timeframe=timeframe)
#                 _end_datetime = _start_datetime + relativedelta(**timedelta_kwargs)
#             else:
#                 _end_datetime = _start_datetime
#
#         episodes_start_end_lst += sorted(one_shift_start_end_lst)
#         one_shift_start_end_lst.clear()
#         unique_episodes_counts = len(list(set(episodes_start_end_lst)))
#         if unique_episodes_counts < num_episodes:
#             selected_period = get_shifted_range(current_shift)
#             ix += 1
#             if ix == len(shifts):
#                 ix = 0
#                 n_shifted_episodes = num_episodes - unique_episodes_counts
#             elif ix == len(shifts) - 1:
#                 n_shifted_episodes = num_episodes - unique_episodes_counts
#         else:
#             finished = True
#
#     return episodes_start_end_lst
