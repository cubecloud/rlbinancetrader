import warnings
import threading
import numpy as np
import gymnasium as gym

from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info

__version__ = 0.005


class EnvResult:
    def __init__(self, num_envs: int, obs_space: gym.Space):
        self.num_envs = num_envs
        self.obs_space = obs_space
        # self.envs = envs
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict(
            [(k, np.zeros((self.num_envs, *tuple(shapes[k])), dtype=dtypes[k])) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        # store info returned by the reset method
        self.reset_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]

        self.actions: np.ndarray = np.zeros((self.num_envs,), dtype=np.int32)
        self.lock = threading.RLock()

    def _save_obs(self, env_idx: int, obs: Union[gym.Space, Dict[str, gym.Space]]) -> None:
        """
        Save the observation in the buffer.

        :param env_idx: the index of the environment
        :param obs: the observation
        """
        if isinstance(obs, dict):
            for k, v in obs.items():
                self.buf_obs[k][env_idx, :] = v
        else:
            self.buf_obs["observation"][env_idx, :] = obs


def step_env(env, env_idx, env_result_obj: EnvResult) -> None:
    # Avoid circular imports
    with env_result_obj.lock:
        obs, env_result_obj.buf_rews[env_idx], terminated, truncated, env_result_obj.buf_infos[env_idx] = env.step(
            env_result_obj.actions[env_idx])
        # convert to SB3 VecEnv api
        env_result_obj.buf_dones[env_idx] = terminated or truncated
        # See https://github.com/openai/gym/issues/3102
        # Gym 0.26 introduces a breaking change
        env_result_obj.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

        if env_result_obj.buf_dones[env_idx]:
            # save final observation where user can get it, then reset
            env_result_obj._save_obs(env_idx, obs)
            # Call the reset method
            obs = env.reset()
            # Store the observation
            env_result_obj._save_obs(env_idx, obs)
        else:
            # Store the observation
            env_result_obj._save_obs(env_idx, obs)


def reset_env(env, seed, env_idx: int, env_result_obj: EnvResult, maybe_options) -> None:
    with env_result_obj.lock:
        obs, env_result_obj.reset_infos[env_idx] = env.reset(seed=seed, **maybe_options)
        env_result_obj._save_obs(env_idx, obs)


class LabMltVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in parallel on separate threads.
    This is useful for environments that can only be run on a single thread.

    Args:
        env_fns: A list of functions that return environments to vectorize.

    Raises:
        ValueError: If the same environment instance is passed as the output of two or more different env_fn.
    """

    actions: np.ndarray

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.envs = [_patch_env(fn()) for fn in env_fns]
        if len(set([id(env.unwrapped) for env in self.envs])) != len(self.envs):
            raise ValueError(
                "You tried to create multiple environments, but the function to create them returned the same instance "
                "instead of creating different objects. "
                "You are probably using `make_vec_env(lambda: env)` or `DummyVecEnv([lambda: env] * n_envs)`. "
                "You should replace `lambda: env` by a `make_env` function that "
                "creates a new instance of the environment at every call "
                "(using `gym.make()` for instance). You can take a look at the documentation for an example. "
                "Please read https://github.com/DLR-RM/stable-baselines3/issues/1151 for more information."
            )
        env = self.envs[0]
        super().__init__(len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.env_result_obj = EnvResult(self.num_envs, obs_space)
        self.metadata = env.metadata

    def step_async(self, actions: np.ndarray) -> None:
        self.env_result_obj.actions = actions
        # self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        # Start threads to step each environment
        threads = []
        for env_idx in range(self.num_envs):
            thread = threading.Thread(target=step_env, args=(self.envs[env_idx], env_idx, self.env_result_obj))
            thread.start()
            threads.append(thread)

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        return (self.env_result_obj.buf_obs,
                np.copy(self.env_result_obj.buf_rews),
                np.copy(self.env_result_obj.buf_dones),
                deepcopy(self.env_result_obj.buf_infos))

    def reset(self) -> VecEnvObs:
        threads = []
        for env_idx in range(self.num_envs):
            maybe_options = {"options": self._options[env_idx]} if self._options[env_idx] else {}

            thread = threading.Thread(target=reset_env, args=(self.envs[env_idx],
                                                              self._seeds[env_idx],
                                                              env_idx,
                                                              self.env_result_obj,
                                                              maybe_options))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        self.reset_infos = self.env_result_obj.reset_infos
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()

        """ returning data directly from self.env_result_obj"""
        return self._obs_from_buf()

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.envs]
        return [env.render() for env in self.envs]  # type: ignore[misc]

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.

        :param mode: The rendering type.
        """
        return super().render(mode=mode)

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, copy_obs_dict(self.env_result_obj.buf_obs))

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]
