import sys
import copy
import math
import numpy as np
import pandas as pd

from dbbinance.fetcher.datautils import get_timeframe_bins
from dbbinance.fetcher.datautils import get_nearest_timeframe

from typing import Callable, Union, Dict, Tuple
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor

from rllab import LabSubprocVecEnv

__version__ = 0.0017


def get_base_env(wrapped_env: Union[Monitor, DummyVecEnv, SubprocVecEnv, LabSubprocVecEnv], env_class):
    """
    Find source environment class
    Args:
        wrapped_env (class instance):   wrapped environment where to find original
        env_class (class):              environment class to find
    """

    env_tmp = wrapped_env
    while isinstance(env_tmp, (DummyVecEnv, SubprocVecEnv, LabSubprocVecEnv, Monitor, VecEnv)):
        if isinstance(env_tmp, (DummyVecEnv, SubprocVecEnv, LabSubprocVecEnv, VecEnv)):
            env_tmp = env_tmp.envs[0]
        elif isinstance(env_tmp, Monitor):
            env_tmp = env_tmp.env
        if isinstance(env_tmp, env_class):
            return env_tmp
    return None


def get_lookback_timeframes(lookback_window: Union[str, int, None], timeframe) -> int:
    if lookback_window is None:
        lookback_timeframes: int = 0
    elif isinstance(lookback_window, int):
        lookback_timeframes = lookback_window
    elif isinstance(lookback_window, str):
        lookback_timeframes = int(
            get_timeframe_bins(lookback_window) // get_timeframe_bins(timeframe))
    else:
        msg = f'Error: unknown lookback_window type = "{type(lookback_window)}"'
        sys.exit(msg)
    return lookback_timeframes


def detect_timeframe(index: pd.DatetimeIndex) -> Tuple[str, int]:
    if len(index) < 2:
        raise ValueError("Error: Length of index < 2")

    delta_minutes = int(index.to_series().diff().dt.total_seconds().dropna()[0] / 60)
    timeframe = get_nearest_timeframe(delta_minutes)
    return timeframe, delta_minutes


def detect_timeframe_and_periods_per_year(index: pd.DatetimeIndex) -> Tuple[str, int]:
    """
    Detect timeframe and annualization factor from DatetimeIndex using its frequency.
    Returns (timeframe_label, periods_per_year)

    Args:
        index: Pandas DatetimeIndex with frequency information

    Returns:
        Tuple of (timeframe_label, periods_per_year)
        Example: ("15m", 365*24*4) for 15-minute data
    """
    if len(index) < 2:
        raise ValueError("Error: Length of index < 2")

    delta_minutes = index.to_series().diff().dt.total_seconds().dropna()[0] / 60
    timeframe = get_nearest_timeframe(delta_minutes)
    periods_per_year = int(365 * 24 * 60 / delta_minutes)
    return timeframe, periods_per_year


# def calculate_sharpe_ratio(returns: pd.Series, periods_per_year: int) -> float:
#     """
#     Calculate annualized Sharpe Ratio for crypto markets.
#     periods_per_year: Number of periods in a year from detect_timeframe()
#     """
#     if len(returns) < 2 or returns.std() == 0:
#         return 0.0
#
#     return (returns.mean() / returns.std()) * (periods_per_year ** 0.5)


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def deserialize_kwargs(_agent_kwargs: Union[dict, str], lab_serializer=None) -> Union[dict, Callable]:
    """

    Args:
        lab_serializer (dict):
    """
    if lab_serializer is None:
        lab_serializer = {}

    agent_kwargs = copy.deepcopy(_agent_kwargs)
    data_update: dict = {}
    if isinstance(agent_kwargs, dict):
        for _key, _value in agent_kwargs.items():
            if isinstance(_value, str):
                for serializer_key, serializer_value in lab_serializer.items():
                    if _value.lower() == serializer_key.lower():
                        deserialized_obj = lab_serializer.get(_value, None)
                        data_update.update({_key: deserialized_obj})
            elif isinstance(_value, dict):
                for serializer_key, serializer_value in lab_serializer.items():
                    if _key.lower() == serializer_key.lower():
                        for _k, _v in _value.items():
                            deserialized_obj = lab_serializer.get(_key, None).get(_k, None)
                            if deserialized_obj is not None:
                                data_update.update({_key: deserialized_obj(**_v)})
                            else:
                                deserialized_obj = lab_serializer.get(_key, None).get(f'{_k}_', None)
                                if deserialized_obj is not None:
                                    data_update.update({_key: deserialized_obj(**_v)()})
                if not data_update:
                    data_update.update({_key: deserialize_kwargs(_value, lab_serializer)})
        agent_kwargs.update(data_update)
    elif isinstance(agent_kwargs, str):
        deserialized_obj = lab_serializer.get(agent_kwargs, None)
        if deserialized_obj is not None:
            agent_kwargs = deserialized_obj
    return agent_kwargs
