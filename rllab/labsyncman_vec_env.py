import os
import threading
from turtle import done
import numpy as np
from time import sleep
from collections import OrderedDict

import multiprocessing as mp
from multiprocessing.managers import SyncManager
from stable_baselines3.common.monitor import Monitor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env

__version__ = 0.011


class MpSync(SyncManager):
    pass


class MpSyncManBase:
    def __init__(self,
                 n_envs: int,
                 start_host: bool = True,
                 host: str = "127.0.0.1",
                 port: int = 5003,
                 authkey: bytes = b"password",
                 mp_rlock: Union[object, None] = None):

        self.start_host = start_host
        self.n_envs = n_envs
        self.manager = MpSync((host, port), authkey=authkey)
        if start_host:
            self.__command: dict = dict({i: ('wait', None) for i in range(n_envs)})
            self.__command_done: dict = dict({i: False for i in range(n_envs)})
            self.__actions: dict = dict({i: None for i in range(n_envs)})
            self.__observations: dict = dict({i: None for i in range(n_envs)})
            self.__rewards: dict = dict({i: None for i in range(n_envs)})
            self.__dones: dict = dict({i: False for i in range(n_envs)})
            self.__infos: dict = dict({i: {} for i in range(n_envs)})
            self.__reset_infos: dict = dict({i: {} for i in range(n_envs)})
            self.__other: dict = dict({i: None for i in range(n_envs)})
            self.manager.register('get_command', callable=lambda: self.__command)
            self.manager.register('get_command_done', callable=lambda: self.__command_done)
            self.manager.register('get_actions', callable=lambda: self.__actions)
            self.manager.register('get_observations', callable=lambda: self.__observations)
            self.manager.register('get_rewards', callable=lambda: self.__rewards)
            self.manager.register('get_dones', callable=lambda: self.__dones)
            self.manager.register('get_infos', callable=lambda: self.__infos)
            self.manager.register('get_reset_infos', callable=lambda: self.__reset_infos)
            self.manager.register('get_other', callable=lambda: self.__other)
            self.manager.start()
        else:
            self.manager.register('get_command')
            self.manager.register('get_command_done')
            self.manager.register('get_actions')
            self.manager.register('get_observations')
            self.manager.register('get_rewards')
            self.manager.register('get_dones')
            self.manager.register('get_infos')
            self.manager.register('get_reset_infos')
            self.manager.register('get_other')
            self.manager.connect()
        self.thread_rlock = self.manager.RLock()

    @classmethod
    def is_server_running(cls, host: str = "127.0.0.1", port: int = 5003, authkey: bytes = b"password"):
        try:
            _ = MpSyncManBase(n_envs=1, start_host=False, host=host, port=port, authkey=authkey)
            return True
        except ConnectionRefusedError:
            return False

    @property
    def command(self) -> dict:
        return self.manager.get_command()

    @property
    def command_done(self) -> dict:
        with self.thread_rlock:
            return self.manager.get_command_done()

    @property
    def is_command_done_all(self) -> dict:
        with self.thread_rlock:
            return all(self.manager.get_command_done().values())

    @property
    def actions(self):
        return self.manager.get_actions()

    @property
    def observations(self):
        return self.manager.get_observations()

    @property
    def rewards(self) -> dict:
        return self.manager.get_rewards()

    @property
    def dones(self) -> dict:
        return self.manager.get_dones()

    @property
    def infos(self) -> dict:
        return self.manager.get_infos()

    @property
    def reset_infos(self) ->dict:
        return self.manager.get_reset_infos()

    @property
    def other(self) -> dict:
        return self.manager.get_other()

    def set_step(self,
                 env_idx: int,
                 observation: Any,
                 reward: Any,
                 done: Any,
                 info: dict,
                 reset_info: dict) -> None:
        with self.thread_rlock:
            self.observations.update({env_idx: observation})
            self.rewards.update({env_idx: reward})
            self.dones.update({env_idx: done})
            self.infos.update({env_idx: info})
            self.reset_infos.update({env_idx: reset_info})


class EnvWrapper(gym.Env):
    def __init__(self, env: gym.Env, sync_manager_obj: MpSyncManBase, env_idx: int):
        self.env = env
        self.sync_manager = sync_manager_obj
        self.env_idx = env_idx
        self.tsteps = 0

    def _set_wait(self) -> None:
        self.sync_manager.command.update({self.env_idx: ("wait", None)})

    def _set_done(self) -> None:
        self.sync_manager.command_done.update({self.env_idx: True})

    def step(self):
        reset_info: Optional[Dict[str, Any]] = {}
        action = self.sync_manager.actions.get(self.env_idx)
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        info["TimeLimit.truncated"] = truncated and not terminated
        if done:
            # save final observation where user can get it, then reset
            info["terminal_observation"] = observation
            observation, reset_info = self.env.reset()
        with self.sync_manager.thread_rlock:
            self.sync_manager.set_step(self.env_idx, observation, reward, done, info, reset_info)
            self._set_done()
            self._set_wait()
        # print(f'#{self.env_idx}/{self.tsteps}', flush=True)
        # self.tsteps += 1

    def reset(self, seed=None, options: Optional[Dict] = None):
        print(f'#{self.env_idx} - reset, tsteps = {self.tsteps}', flush=True)
        observation, reset_info = self.env.reset(seed=seed, options=options)
        with self.sync_manager.thread_rlock:
            self.sync_manager.set_step(self.env_idx, observation, None, None, {}, reset_info)
            self._set_done()
            self._set_wait()
        # self.tsteps = 0
            # print(f'#{self.env_idx}/{self.tsteps}', flush=True)

    def close(self):
        self.env.close()
        with self.sync_manager.thread_rlock:
            self._set_done()
            self._set_wait()

    def get_spaces(self):
        with self.sync_manager.thread_rlock:
            self.sync_manager.other.update({self.env_idx:(self.env.observation_space, self.env.action_space)})
            self._set_done()
            self._set_wait()

    def get_attr(self, attr):
        with self.sync_manager.thread_rlock:
            self.sync_manager.other.update({self.env_idx: getattr(self.env, attr)})
            self._set_done()
            self._set_wait()

    def set_attr(self, attr_name: str, value: Any):
        setattr(self.env, attr_name, value)
        with self.sync_manager.thread_rlock:
            self._set_done()
            self._set_wait()

    def env_method(self, method_name, args, kwargs):
        with self.sync_manager.thread_rlock:
            self.sync_manager.other.update({self.env_idx: getattr(self.env, method_name)(*args, **kwargs)})
            self._set_done()
            self._set_wait()

    def is_wrapped(self, wrapper_class):
        # with self.sync_manager.thread_rlock:
        self.sync_manager.other.update({self.env_idx: isinstance(self.env, wrapper_class)})
        self._set_done()
        self._set_wait()

    def render(self, mode: Optional[str] = None):
        with self.sync_manager.thread_rlock:
            self.sync_manager.other.update({self.env_idx: self.env.render(mode=mode)})
            self._set_done()
            self._set_wait()


def thread_runner(env: EnvWrapper, env_idx: int):
    # Import here to avoid a circular import
    # from stable_baselines3.common.env_util import is_wrapped
    while True:
        try:
            cmd, data = env.sync_manager.command.get(env_idx)
            # print(f'#{env_idx}', cmd, data, flush=True)
            if cmd == "wait":
                pass
            elif cmd == "step":
                env.step()
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                env.reset(seed=data[0], **maybe_options)
            elif cmd == "render":
                env.render(data)
            elif cmd == "close":
                env.close()
                break
            elif cmd == "get_spaces":
                env.get_spaces()
            elif cmd == "env_method":
                env.env_method(data[0], data[1], data[2])
            elif cmd == "get_attr":
                env.get_attr(data)
            elif cmd == "set_attr":
                env.set_attr(data[0], data[1])  # type: ignore[func-returns-value]
            elif cmd == "is_wrapped":
                env.is_wrapped(data)
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


def process_runner(env_cls, env_kwargs: dict, env_indices: list, sync_manager_kwargs: dict, mp_rlock):
    """
    Create the sync manager for each process. 
    Threads using sync_manager_obj for thread-safe communication with thread_rlock 
    """

    def init_env(rank) -> gym.Env:
        # For type checker:
        # assert monitor_kwargs is not None
        # assert wrapper_kwargs is not None
        # assert env_kwargs is not None

        if isinstance(env_cls, str):
            # if the render mode was not specified, we set it to `rgb_array` as default.
            kwargs = {"render_mode": "rgb_array"}
            kwargs.update(env_kwargs)
            try:
                env_ = gym.make(env_cls, **kwargs)  # type: ignore[arg-type]
            except TypeError:
                env_ = gym.make(env_cls, **env_kwargs)
        else:
            env_ = env_cls(**env_kwargs)
            # Patch to support gym 0.21/0.26 and gymnasium
            env_ = _patch_env(env_)
        seed = env_kwargs.get('seed', 42)
        if seed is not None:
            # Note: here we only seed the action space
            # We will seed the env at the next reset
            env_.action_space.seed(seed + rank)
        # Wrap the env in a Monitor wrapper
        # to have additional training information
        # monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
        # # Create the monitor folder if needed
        # if monitor_path is not None and monitor_dir is not None:
        #     os.makedirs(monitor_dir, exist_ok=True)
        env_ = Monitor(env_, **{})
        # Optionally, wrap the environment with the provided wrapper
        return env_

    sync_manager_kwargs.update({"mp_rlock": mp_rlock, "start_host": False, "n_envs": len(env_indices)})

    sync_manager_obj = MpSyncManBase(**sync_manager_kwargs)

    envs = {}
    for env_idx in env_indices:
        env = init_env(env_idx)
        env = EnvWrapper(env, sync_manager_obj, env_idx)
        envs.update({env_idx: env})

    threads = []
    for env_idx in env_indices:
        thread = threading.Thread(target=thread_runner, args=(envs[env_idx], env_idx))
        thread.start()
        threads.append(thread)

    # Wait for all threads to finish
    for thread in threads:
        thread.join()


class SyncManVecEnv(VecEnv):
    def __init__(self,
                 env_cls,
                 env_kwargs: dict = {},
                 n_envs: int = 1,
                 start_host: bool = True,
                 host: str = "127.0.0.1",
                 port: int = 5003,
                 authkey: bytes = b"password",
                 n_process: Optional[int] = None,
                 start_method: Optional[str] = None
                 ):

        def calculate_indices(n_envs, n_processes):
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

        self.waiting = False
        self.closed = False

        # Create HOST sync manager 
        self.n_envs = n_envs

        self.mp_rlock = mp.RLock()
        self.host_manager = MpSyncManBase(self.n_envs, start_host, host, port, authkey, self.mp_rlock)

        if n_process is None:
            n_process = max(1, mp.cpu_count() - 2)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.processes = []

        processes_env_indices = calculate_indices(n_envs, n_process)

        sync_manager_kwargs = {"start_host": False, "host": host, "port": port, "authkey": authkey}

        for pr_idx in range(len(processes_env_indices)):
            args = (env_cls, env_kwargs, processes_env_indices[pr_idx], sync_manager_kwargs, self.mp_rlock)
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=process_runner, args=args, daemon=True)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)

        self.host_manager.command.update({0: ("get_spaces", None)})
        while not self.host_manager.command_done.get(0, False):
            pass
        observation_space, action_space = self.host_manager.other.get(0)
        super().__init__(n_envs, observation_space, action_space)

    def _set_command(self, cmd: tuple) -> None:
        self.host_manager.command.update({env_idx: cmd for env_idx in range(self.n_envs)})

    def _set_command_idx(self, cmd: tuple, env_idx: int) -> None:
        self.host_manager.command.update({env_idx: cmd})

    def _set_command_done(self, status: bool) -> None:
        self.host_manager.command_done.update({env_idx: status for env_idx in range(self.n_envs)})

    def _set_command_done_idx(self, status: bool, env_idx: int) -> None:
        self.host_manager.command_done.update({env_idx: status})

    def step_async(self, actions: np.ndarray) -> None:
        self.host_manager.actions.update({env_idx: actions[env_idx] for env_idx in range(self.n_envs)})
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        self._set_command(("step", None))
        # all_indices = list(range(self.n_envs))
        observations: list = [None] * self.n_envs
        rewards: list = [None] * self.n_envs
        dones: list = [False] * self.n_envs
        infos: list = [{} for _ in range(self.n_envs)]
        self.reset_infos: list = [{} for _ in range(self.n_envs)]
        # if not self.host_manager.is_command_done_all:
        #     while all_indices:
        #         current_done_indices = [env_idx for env_idx, done in self.host_manager.command_done.items() if done and (env_idx in all_indices)]
        #         for env_idx in current_done_indices:
        #             observations[env_idx] = self.host_manager.observations.get(env_idx)
        #             rewards[env_idx] = self.host_manager.rewards.get(env_idx)
        #             dones[env_idx] = self.host_manager.dones.get(env_idx)
        #             infos[env_idx] = self.host_manager.infos.get(env_idx)
        #             reset_infos[env_idx] = self.host_manager.reset_infos.get(env_idx)
        #             all_indices.remove(env_idx)
        # else:
        while not self.host_manager.is_command_done_all:
            pass
        observations_items = self.host_manager.observations.items()
        for env_idx, obs in sorted(observations_items):
            observations[env_idx] = obs

        rewards_items = self.host_manager.rewards.items()
        for env_idx, rew in sorted(rewards_items):
            rewards[env_idx] = rew

        dones_items = self.host_manager.dones.items()
        for env_idx, done in sorted(dones_items):
            dones[env_idx] = done

        infos_items = self.host_manager.infos.items()
        for env_idx, info in sorted(infos_items):
            infos[env_idx] = info

        reset_infos_items = self.host_manager.reset_infos.items()
        for env_idx, info in sorted(reset_infos_items):
            self.reset_infos[env_idx] = info
        
        result = _flatten_obs(observations, self.observation_space), np.stack(rewards), np.stack(dones), infos  # type: ignore[return-value]
        self.waiting = False
        return result

    def reset(self) -> VecEnvObs:
        for env_idx in range(self.n_envs):
            self._set_command_idx(cmd=("reset", (self._seeds[env_idx], self._options[env_idx])), env_idx=env_idx)
        observations: list = [None] * self.n_envs
        self.reset_infos: list = [{} for _ in range(self.n_envs)]

        while not self.host_manager.is_command_done_all:
            pass
        observations_items = self.host_manager.observations.items()
        for env_idx, obs in sorted(observations_items):
            observations[env_idx] = obs
        
        reset_infos_items = self.host_manager.reset_infos.items()
        for env_idx, info in sorted(reset_infos_items):
            self.reset_infos[env_idx] = info

        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        result = _flatten_obs(observations, self.observation_space)
        return result

    def close(self) -> None:
        if self.closed:
            return
        self._set_command(("close", None))
        while not self.host_manager.is_command_done_all:
            pass
        for process in self.processes:
            process.join()
        self.closed = True

    def get_attr(self, attr_name, indices=None):
        """Get attribute from each environment.

        Args:
            attr_name (str): Attribute name.
            indices (list): Indices of environments.

        Returns:
            list: List of attributes.
        """
        if indices is None:
            self._set_command(("get_attr", attr_name))
            while not self.host_manager.is_command_done_all:
                pass
            result = [self.host_manager.other.get(env_idx, None) for env_idx in range(self.n_envs)] 
        else:
            for env_idx in range(self.n_envs):
                if env_idx in indices:
                    self._set_command_idx(("get_attr", attr_name), env_idx)
                else:
                    self._set_command_idx(("wait", None), env_idx)
            while not all([self.host_manager.command_done.get(env_idx, False) for env_idx in indices]):
                pass
            result = [self.host_manager.other.get(env_idx, None) for env_idx in indices]
        return result

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute for each environment.

        Args:
            attr_name (str): Attribute name.
            value: Value to set.
            indices (list): Indices of environments.
        """
        if indices is None:
            self._set_command(("set_attr", (attr_name, value)))
            while not self.host_manager.is_command_done_all:
                pass
        else:
            for env_idx in range(self.n_envs):
                if env_idx in indices:
                    self._set_command_idx(("set_attr", (attr_name, value)), env_idx)
                else:
                    self._set_command_idx(("wait", None), env_idx)
            while not all([self.host_manager.command_done.get(env_idx, False) for env_idx in indices]):
                pass

    def env_method(self, method_name, *args, indices=None, **kwargs):
        """Call method on each environment.

        Args:
            method_name (str): Method name.
            *args: Arguments to pass to the method.
            indices (list): Indices of environments.
            **kwargs: Keyword arguments to pass to the method.

        Returns:
            list: List of results from each environment.
        """
        if indices is None:
            self._set_command(("env_method", (method_name, args, kwargs)))
            while not self.host_manager.is_command_done_all:
                sleep(0.001)
            result = [self.host_manager.other.get(env_idx, None) for env_idx in range(self.n_envs)]
        else:
            for env_idx in range(self.n_envs):
                if env_idx in indices:
                    self._set_command_idx(("env_method", (method_name, args, kwargs)), env_idx)
                else:
                    self._set_command_idx(("wait", None), env_idx)
            while not all([self.host_manager.command_done.get(env_idx, False) for env_idx in indices]):
                sleep(0.001)
            result = [self.host_manager.other.get(env_idx, None) for env_idx in indices]
        return result

    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if environments are wrapped with a specific wrapper.

        Args:
            wrapper_class (type): Wrapper class.
            indices (list): Indices of environments.

        Returns:
            list: List of boolean values indicating whether each environment is wrapped.
        """
        if indices is None:
            self._set_command(("is_wrapped", wrapper_class))
            while not self.host_manager.is_command_done_all:
                sleep(0.001)
            result = [self.host_manager.other.get(env_idx, None) for env_idx in range(self.n_envs)]
        else:
            for env_idx in range(self.n_envs):
                if env_idx in indices:
                    self._set_command_idx(("is_wrapped", wrapper_class), env_idx)
                else:
                    self._set_command_idx(("wait", None), env_idx)
            while not all([self.host_manager.command_done.get(env_idx, False) for env_idx in indices]):
                sleep(0.001)
            result = [self.host_manager.other.get(env_idx, None) for env_idx in indices]
        return result

    def get_images(self):
        """Get images from each environment.

        Returns:
            list: List of images.
        """
        self._set_command(("get_attr", 'rgb_array'))
        while not self.host_manager.is_command_done_all:
            sleep(0.001)
        result = [self.host_manager.other.get(env_idx, None) for env_idx in range(self.n_envs)]
        return result


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


if __name__ == "__main__":
    pass
