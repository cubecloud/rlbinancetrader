import logging
import time
import threading
import warnings
import multiprocessing as mp
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from dbbinance.fetcher.slocks import SThLock

from rllab.fakelock import FakeRLock
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env

__version__ = 0.019


class EnvWrapper(gym.Env):

    def __init__(self, envs: dict):
        self.envs = envs
        self.tsteps = 0
        pass

    def step(self, actions: List[np.ndarray]) -> Dict[int, Any]:
        envs_results: dict = {}
        for ix, env_idx in enumerate(self.envs.keys()):
            reset_info: Optional[Dict[str, Any]] = {}
            observation, reward, terminated, truncated, info = self.envs[env_idx].step(actions[ix])
            done = terminated or truncated
            info["TimeLimit.truncated"] = truncated and not terminated
            if done:
                # save final observation where user can get it, then reset
                info["terminal_observation"] = observation
                observation, reset_info = self.envs[env_idx].reset()
            envs_results.update({env_idx: (observation, reward, done, info, reset_info)})
        return envs_results

    def reset(self, seed: Union[List[int], List[None]] = [], options: Optional[List[Dict]] = None) -> Dict[int, Any]:
        envs_results: dict = {}
        for ix, env_idx in enumerate(self.envs.keys()):
            observation, reset_info = self.envs[env_idx].reset(seed=seed[ix], options=options[ix])
            envs_results.update({env_idx: (observation, reset_info)})
        return envs_results

    def render(self, mode: Optional[str] = None) -> Dict[int, Any]:
        envs_results: dict = {}
        for env_idx in self.envs.keys():
            frame = self.envs[env_idx].render(mode=mode)
            envs_results.update({env_idx: frame})
        return envs_results

    def close(self):
        for env in self.envs.values():
            env.close()

    def get_spaces(self, indices) -> Dict[int, Any]:
        envs_results: dict = {}
        for env_idx in indices:
            envs_results.update({env_idx: (self.envs[env_idx].observation_space, self.envs[env_idx].action_space)})
        return envs_results

    def get_attr(self, attr, indices: List[int]) -> Dict[int, Any]:
        envs_results: dict = {}
        if attr == 'action_masks':
            for env_idx in indices:
                envs_results.update({env_idx: True})
        else:
            for env_idx in indices:
                envs_results.update({env_idx: getattr(self.envs[env_idx].unwrapped, attr)})
        return envs_results

    def set_attr(self, attr_name: str, value: Any, indices: List[int]):
        for env_idx in indices:
            setattr(self.envs[env_idx], attr_name, value)

    def env_method(self, method_name, args, kwargs, indices: List[int]) -> Dict[int, Any]:
        envs_results: dict = {}
        for env_idx in indices:
            env_data = getattr(self.envs[env_idx].unwrapped, method_name)(*args, **kwargs)
            envs_results.update({env_idx: env_data})
        return envs_results

    def is_wrapped(self, wrapper_class, indices: List[int]) -> Dict[int, bool]:
        envs_results: dict = {}
        for env_idx in indices:
            envs_results.update({env_idx: isinstance(self.envs[env_idx], wrapper_class)})
        return envs_results

    
def _worker(remote: mp.connection.Connection, parent_remote: mp.connection.Connection,
            envs_fn_lst_wrapper: CloudpickleWrapper, process_env_indices: List[int]) -> None:


    """
    we don't need in our case threading.RLock for MpCacheManager cos all environments method's
    called sequentially. We just need FakeRlock with singleton wrapper
    """
    _lock = SThLock(FakeRLock(), unique_name=f'train_rlock')

    parent_remote.close()
    envs = [_patch_env(env()) for env in envs_fn_lst_wrapper.var]
    env = EnvWrapper(dict(zip(process_env_indices, envs)))

    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(env.step(data))
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                remote.send(env.reset(seed=data[0], **maybe_options))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                envs_data = env.get_spaces(data)
                remote.send(envs_data)
            elif cmd == "env_method":
                envs_data = env.env_method(data[0], data[1], data[2], data[3])
                remote.send(envs_data)
            elif cmd == "get_attr":
                envs_data = env.get_attr(data[0], data[1])
                remote.send(CloudpickleWrapper(envs_data))
            elif cmd == "set_attr":
                envs_data = env.set_attr(data[0], data[1], data[2])  # type: ignore[func-returns-value]
                remote.send(envs_data)
            elif cmd == "is_wrapped":
                envs_data = env.is_wrapped(data[0], data[1])
                remote.send(envs_data)
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class LabSubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None,
                 n_processes: Optional[int] = None, process_shared_objs: Optional[Dict] = None):

        def calculate_indices(n_envs, n_processes) -> List[List[int]]:
            # Calculate the number of environments per process
            envs_per_process = n_envs // n_processes
            # Calculate the remaining environments
            remaining_envs = n_envs % n_processes
            # Initialize the list of indices
            indices = []
            # Initialize the start index
            start_idx = 0
            # Loop over the number of processes
            for i in range(n_processes):
                # Calculate the number of environments for this process
                num_envs = envs_per_process + (1 if i < remaining_envs else 0)
                # Calculate the end index
                end_idx = start_idx + num_envs
                # Append the indices for this process to the list
                env_lst = list(range(start_idx, end_idx))
                if env_lst:
                    indices.append(list(range(start_idx, end_idx)))
                else:
                    break
                # Update the start index
                start_idx = end_idx
            return indices

        if process_shared_objs is None:
            self.process_shared_objs = {}
        else:
            self.process_shared_objs = process_shared_objs
        self.waiting = False
        self.closed = False
        self.n_envs = len(env_fns)
        if n_processes is None:
            self.n_processes = max(1, mp.cpu_count() - 2)
        else:
            self.n_processes = n_processes

        self.env_indices_per_process = calculate_indices(self.n_envs, self.n_processes)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"

        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(len(self.env_indices_per_process))])

        self.processes = []
        for pr_idx in range(len(self.env_indices_per_process)):
            process_env_indices = self.env_indices_per_process[pr_idx]
            envs_lst = [env_fns[i] for i in process_env_indices]
            args = (self.work_remotes[pr_idx], self.remotes[pr_idx], CloudpickleWrapper(envs_lst), process_env_indices)
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            self.work_remotes[pr_idx].close()

        """
        Getting the observation and action spaces from process 
        with idx [0] and from environment with idx [0] 
        """
        count = 0
        while True:
            data = {}
            indices = self._get_indices([0])
            self.remotes[0].send(("get_spaces", indices))
            try:
                result = self.remotes[0].recv()
            except ConnectionResetError as e:
                time.sleep(0.1)
                count += 1
                if count > 200:
                    raise Exception(e)
            else:
                data.update(result)
                break
        observation_space, action_space = data[0]
        super().__init__(len(env_fns), observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        # assert len(actions) == self.n_envs
        for pr_idx in range(len(self.env_indices_per_process)):
            process_env_indices = self.env_indices_per_process[pr_idx]
            # actions_indices = [actions[ix] for ix in process_env_indices]
            actions_indices = actions[np.asarray(process_env_indices)]
            self.remotes[pr_idx].send(("step", actions_indices))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [None] * self.n_envs
        data = {}
        for pr_idx in range(len(self.env_indices_per_process)):
            data.update(self.remotes[pr_idx].recv())
        for env_ix in range(self.n_envs):
            results[env_ix] = data[env_ix]

        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results)  # type: ignore[assignment]
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(
            dones), infos  # type: ignore[return-value]

    def reset(self) -> VecEnvObs:
        _seed = []
        _options = []
        results = [None] * self.n_envs

        for pr_idx in range(len(self.env_indices_per_process)):
            process_env_indices = self.env_indices_per_process[pr_idx]
            for env_idx in process_env_indices:
                _seed.append(self._seeds[env_idx])
                _options.append(self._options[env_idx])
            self.remotes[pr_idx].send(("reset", (_seed, _options)))

        data = {}
        for pr_idx in range(len(self.env_indices_per_process)):
            data.update(self.remotes[pr_idx].recv())
        for env_ix in range(self.n_envs):
            results[env_ix] = data[env_ix]
        obs, self.reset_infos = zip(*results)  # type: ignore[assignment]
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return _flatten_obs(obs, self.observation_space)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for pr_idx in range(len(self.env_indices_per_process)):
                """ Just rcv() and discard the data """
                self.remotes[pr_idx].recv()

        for pr_idx in range(len(self.env_indices_per_process)):
            self.remotes[pr_idx].send(("close", None))

        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        outputs = [None] * self.n_envs
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.remotes]

        for pr_idx in range(len(self.env_indices_per_process)):
            # gather render return from subprocesses
            self.remotes[pr_idx].send(("render", None))

        data = {}
        for pr_idx in range(len(self.env_indices_per_process)):
            data.update(self.remotes[pr_idx].recv())
        for env_ix in range(self.n_envs):
            outputs[env_ix] = data[env_ix]
        return outputs

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        indices = self._get_indices(indices)
        indices_dict = {}
        for pr_idx in range(len(self.env_indices_per_process)):
            _actual_pr_indices = list(set(self.env_indices_per_process[pr_idx]) & set(indices))
            if _actual_pr_indices:
                indices_dict.update({pr_idx: _actual_pr_indices})
                self.remotes[pr_idx].send(("get_attr", (attr_name, _actual_pr_indices)))

        data = {}
        for pr_idx in indices_dict.keys():
            envs_data = self.remotes[pr_idx].recv().var
            data.update(envs_data)
        return [data[env_idx] for env_idx in indices]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        indices = self._get_indices(indices)
        indices_dict = {}
        for pr_idx in range(len(self.env_indices_per_process)):
            _actual_pr_indices = list(set(self.env_indices_per_process[pr_idx]) & set(indices))
            if _actual_pr_indices:
                indices_dict.update({pr_idx: _actual_pr_indices})
                self.remotes[pr_idx].send(("set_attr", (attr_name, value, _actual_pr_indices)))
        for pr_idx in indices_dict.keys():
            self.remotes[pr_idx].recv()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        indices = self._get_indices(indices)
        indices_dict = {}
        for pr_idx in range(len(self.env_indices_per_process)):
            _actual_pr_indices = list(set(self.env_indices_per_process[pr_idx]) & set(indices))
            if _actual_pr_indices:
                indices_dict.update({pr_idx: _actual_pr_indices})
                self.remotes[pr_idx].send(("env_method", (method_name, method_args, method_kwargs, _actual_pr_indices)))

        data = {}
        for pr_idx in indices_dict.keys():
            envs_data = self.remotes[pr_idx].recv()
            data.update(envs_data)
        return [data[env_idx] for env_idx in indices]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        indices = self._get_indices(indices)
        indices_dict = {}
        for pr_idx in range(len(self.env_indices_per_process)):
            _actual_pr_indices = list(set(self.env_indices_per_process[pr_idx]) & set(indices))
            if _actual_pr_indices:
                indices_dict.update({pr_idx: _actual_pr_indices})
                self.remotes[pr_idx].send(("is_wrapped", (wrapper_class, _actual_pr_indices)))

        data = {}
        for pr_idx in indices_dict.keys():
            data.update(self.remotes[pr_idx].recv())
        return [data[env_idx] for env_idx in indices]


def _flatten_obs(obs: Union[List[VecEnvObs], Tuple[VecEnvObs]], space: spaces.Space) -> VecEnvObs:
    """
    Flatten observations, depending on the observation space.

    :param obs: observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(np.stack([o[i] for o in obs]) for i in range(obs_len))  # type: ignore[index]
    else:
        return np.stack(obs)  # type: ignore[arg-type]
