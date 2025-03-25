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

__version__ = 0.0021


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


def check_val(val, old_item, new_item) -> any:
    item_type = type(old_item)
    if isinstance(val, dict):
        val = dive_kvreplace(val, old_item, new_item)
    elif isinstance(val, (list, tuple)):
        new_val = []
        for val_i in val:
            new_val.append(check_val(val_i, old_item, new_item))
        if isinstance(val, tuple):
            val = tuple(new_val)
        else:
            val = new_val
    elif isinstance(val, item_type):
        if item_type is str:
            if old_item in val:
                val = val.replace(old_item, new_item)
        else:
            if val == old_item:
                val = new_item
    return val


def dive_kvreplace(source_dict: dict, old_item: any, new_item: any) -> dict:
    new_dict = {}
    for k, v in source_dict.items():
        k = check_val(k, old_item, new_item)
        v = check_val(v, old_item, new_item)
        new_dict.update({k: v})
    return new_dict


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

#
# def find_matching_class(lab_serializer, value):
#     """
#     Recursively searches for a class in the lab_serializer dictionary that matches the type of the given value.
#
#     Args:
#         lab_serializer (dict): Dictionary containing classes for serialization.
#         value: Value whose type needs to be matched with a class from lab_serializer.
#
#     Returns:
#         Class that matches the type of the value, or None if no match is found.
#     """
#     for key, val in lab_serializer.items():
#         if isinstance(val, dict):
#             # Recursively search inside nested dictionaries
#             found_class = find_matching_class(val, value)
#             if found_class:
#                 return found_class
#         elif isinstance(val, type):
#             # Check if the value's type matches the current class
#             if isinstance(value, val):
#                 return val
#     return None
#
#
# def check_kwargs(val, lab_serializer):
#     # Handle dictionaries
#     if isinstance(val, dict):
#         for key, value in val.items():
#             # Try to find a matching class for the value
#             matching_class = find_matching_class(lab_serializer, value)
#
#             if matching_class:
#                 # Store the class name as the serialized value
#                 serialized_data[key] = matching_class.__name__
#             elif isinstance(value, dict):
#                 # Recursively serialize nested dictionaries
#                 serialized_data[key] = serialize_kwargs(value, lab_serializer)
#             elif isinstance(value, list):
#                 # Serialize elements within the list
#                 serialized_data[key] = [serialize_kwargs(item, lab_serializer) for item in value]
#             else:
#                 # Keep primitive data types unchanged
#                 serialized_data[key] = value
#     # If agent_kwargs is a string
#     elif isinstance(agent_kwargs, str):
#         # Assume that the string is already serialized
#         return agent_kwargs
#
#     return serialized_data
#
#
# def serialize_kwargs(agent_kwargs: Union[dict, list, str], lab_serializer=None) -> Union[dict, list, str]:
#     """
#     Serializes the values of kwargs into strings.
#
#     Args:
#         agent_kwargs (Union[dict, list, str]): Original dictionary, list, or string.
#         lab_serializer (dict): Dictionary used to serialize objects.
#
#     Returns:
#         Union[dict, list, str]: Serialized dictionary, list, or string.
#     """
#     if lab_serializer is None:
#         lab_serializer = {}
#
#     new_agent_kwargs = {}
#
#     # # Handle lists
#     # if isinstance(agent_kwargs, list):
#     #     return [serialize_kwargs(item, lab_serializer) for item in agent_kwargs]
#
#     # serialized_data = {}
#
#     for k, v in agent_kwargs.items():
#         k = check_kwargs(k, lab_serializer)
#         v = check_kwargs(v, lab_serializer)
#         new_agent_kwargs.update({k: v})
#     return new_agent_kwargs


if __name__ == '__main__':
    """ Testing serializer """
    from rllab.labserializer import lab_serializer

    _timeframe = '15m'
    _discretization = '15m'
    total_timesteps = 3_000_000_000
    _start_datetime = '2023-07-20 01:00:00'
    _end_datetime = '2024-07-30 01:00:00'
    agents_n_env = int(940)
    n_steps = 600
    warmup_timesteps = (agents_n_env * n_steps) * 300
    indicators_sign = True

    data_processor_kwargs = dict(start_datetime=_start_datetime,
                                 end_datetime=_end_datetime,
                                 timeframe=_timeframe,
                                 discretization=_discretization,
                                 symbol_pair='BTCUSDT',
                                 market='spot',
                                 minimum_train_size=640,
                                 maximum_train_size=645,
                                 minimum_test_size=640,
                                 maximum_test_size=645,
                                 test_size=0.1,
                                 verbose=1,
                                 indicators_sign=indicators_sign
                                 )
    test_kwargs = dict(
        filename=144_384_000,
        # filename='best_model',
        reset_num_timesteps=True,
        total_timesteps=total_timesteps,
        env_kwargs_update={
            'data_processor_kwargs': data_processor_kwargs,
            'stable_cache_data_n': agents_n_env,
            'reuse_data_prob': 1.0,
            'verbose': 0,
            'render_mode': 'human',
            'gamma': 0.8,
        },

        agent_kwargs_update={
            'n_steps': n_steps,
            'batch_size': int(agents_n_env * n_steps // 20),
            'n_epochs': 10,
            'stats_window_size': 50,
            'clip_range': 0.07,
            'clip_range_vf': 0.07,
            'ent_coef': 0.01,
            'vf_coef': 1.0,
            'gamma': 0.8,
            'learning_rate': {'CoScheduler': dict(warmup=warmup_timesteps,
                                                  stable_warmup=True,
                                                  floor_learning_rate=1e-6,
                                                  min_learning_rate=2.5e-6,
                                                  learning_rate=4.5e-6,
                                                  total_epochs=total_timesteps,
                                                  epsilon=1,
                                                  pre_warmup_coef=0.04)
                              },
            'seed': 543,
        },
        env_wrapper='labsubproc',
        env_wrapper_kwargs_update={'use_threads': False},
        n_envs=agents_n_env,
        n_eval_episodes=100,
        eval_freq=n_steps,
        verbose=1,
    )

    total_timesteps = 16_000_000
    buffer_size = 1_000_000
    learning_start = 750_000
    batch_size = 1024

    # action_noise_box = OrnsteinUhlenbeckActionNoise(mean=5e-1 * np.ones(3), sigma=4.99e-1 * np.ones(3), dt=1e-2)

    sac_policy_kwargs = dict(
        features_extractor_class='MlpExtractorNN',
        features_extractor_kwargs=dict(features_dim=256, activation_fn='ReLU'),
        share_features_extractor=True,
        activation_fn='ReLU',
        # net_arch=net_arch,
    )
    sac_kwargs = dict(policy="MlpPolicy",
                      buffer_size=buffer_size,
                      learning_starts=learning_start,
                      policy_kwargs=sac_policy_kwargs,
                      batch_size=batch_size,
                      replay_buffer_class='HerReplayBuffer',
                      stats_window_size=100,
                      ent_coef='auto_0.0001',
                      learning_rate={'CoScheduler': dict(warmup=learning_start,
                                                         learning_rate=2e-4,
                                                         min_learning_rate=1e-5,
                                                         total_epochs=total_timesteps,
                                                         epsilon=100)},
                      action_noise={'OrnsteinUhlenbeckActionNoise': dict(mean=5e-1 * np.ones(3),
                                                                         sigma=4.99e-1 * np.ones(3),
                                                                         dt=1e-2)},
                      train_freq=(2, 'step'),
                      target_update_interval=10,  # update target network every 10 _gradient_ steps
                      device="auto",
                      verbose=1)

    test_kwargs = sac_kwargs
    deserialized_kwargs = deserialize_kwargs(test_kwargs, lab_serializer=lab_serializer)
    # new_kwargs = serialize_kwargs(deserialized_kwargs, lab_serializer=lab_serializer)

    print(test_kwargs)
    print(deserialized_kwargs)
    # print(new_kwargs)
