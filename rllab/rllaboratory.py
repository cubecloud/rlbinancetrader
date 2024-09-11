import copy
import os
import logging
import datetime
from pytz import timezone

import pandas as pd

from typing import List, Union, Dict, Type, Optional

from binanceenv.cache import CacheManager
from binanceenv.cache import cache_manager_obj
from binanceenv.cache import eval_cache_manager_obj

from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback
# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C, PPO, DQN, TD3, DDPG, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.env_util import unwrap_wrapper
from dataclasses import asdict, dataclass, field, make_dataclass
# from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

import gymnasium
from gymnasium.utils import seeding

from rllab import ConfigMethods
from rllab import lab_evaluate_policy
from rllab import LabEvalCallback
from rllab.labtools import deserialize_kwargs
from rllab.labserializer import lab_serializer

__version__ = 0.039

TZ = timezone('Europe/Moscow')

logger = logging.getLogger()


# ACTION_NOISE: dict = {'ornstein_uhlenbeck': OrnsteinUhlenbeckActionNoise, 'normal': NormalActionNoise}


def get_exp_id() -> str:
    return f'exp-{datetime.datetime.now(TZ).strftime("%d%m-%H%M%S")}'


@dataclass(init=True)
class LABConfig:
    EXP_ID: str = field(default_factory=get_exp_id, init=True)
    EXPERIMENT_PATH: str = field(default_factory=str, init=True)
    ENV_NAME: str = field(default_factory=str, init=True)
    ALGO: str = field(default_factory=str, init=True)
    OBS_TYPE: str = field(default_factory=str, init=True)
    DIRS: dict = field(default_factory=dict, init=True)
    FILENAME: str = field(default_factory=str, init=True)
    DETERMINISTIC: bool = field(default_factory=bool, init=True)
    AGENTS_N_ENV: int = field(default_factory=int, init=True)
    ENV_WRAPPER: str = field(default_factory=str, init=True)
    TOTAL_TIMESTEPS: int = field(default_factory=int, init=True)
    EVAL_FREQ: int = field(default_factory=int, init=True)
    N_EVAL_EPISODES: int = field(default_factory=int, init=True)
    LOG_INTERVAL: int = field(default_factory=int, init=True)


env_wrapper_dict: dict = {'dummy': DummyVecEnv, 'subproc': SubprocVecEnv}


class LabBase:
    def __init__(self,
                 env_cls: Union[object, List[object,]],
                 agents_cls: Union[object, List[object]],
                 env_kwargs: Union[List[dict,], dict],
                 agents_kwargs: Union[List[dict,], dict],
                 agents_n_env: Union[List[int], int, None] = None,
                 env_wrapper: str = 'dummy',
                 total_timesteps: int = 500_000,
                 experiment_path: str = './',
                 eval_freq: int = 50_000,
                 checkpoint_num: int = 20,
                 n_eval_episodes: int = 10,
                 log_interval: int = 200,
                 deterministic=True,
                 verbose: int = 1,
                 exp_cfg: Union[LABConfig, None] = None,
                 seed: int = 42
                 ):
        """
        Get all arguments to instantiate all classes and object inside the Lab

        Args:
            env_cls (class or list):        environment classes
            agents_cls (class or list):     agents classes
            env_kwargs (dict or list):      kwargs for environment
            agents_kwargs (dict or list):   kwargs for each agent
            agents_n_env (int, list, None): n of environment or list
            env_wrapper: (str)              'dummy' or 'subproc'
            total_timesteps (int):          total timesteps
            experiment_path (str):          path
            checkpoint_num (int):           n of checkpoints to save
            eval_freq (int):                frequency of evaluation
            n_eval_episodes (int):          n of episodes to evaluate
            deterministic (bool):           deterministic or stochastic
        """

        self.policy_n_env_default: dict = {PPO: 3, A2C: 3, DQN: 3, TD3: 3, DDPG: 3, SAC: 2}
        self.eval_freq = eval_freq
        self.checkpoint_num = checkpoint_num
        self.n_eval_episodes = n_eval_episodes
        self.log_interval = log_interval
        self.experiment_path = experiment_path
        self.total_timesteps = total_timesteps
        self.deterministic = deterministic
        self.env_cls = env_cls
        self.env_kwargs = env_kwargs
        self.np_random = None
        self.seed = self.get_seed(seed)
        self.env_wrapper = env_wrapper
        self.env_wrapper_cls = env_wrapper_dict.get(env_wrapper, DummyVecEnv)
        self.verbose = verbose

        if exp_cfg is None:
            self.base_cfg = LABConfig()
            self.base_cfg.EXPERIMENT_PATH = experiment_path
            self.base_cfg.TOTAL_TIMESTEPS = self.total_timesteps
            self.base_cfg.EVAL_FREQ = self.eval_freq
            self.base_cfg.ENV_WRAPPER = self.env_wrapper
            self.base_cfg.DETERMINISTIC = self.deterministic
            self.base_cfg.N_EVAL_EPISODES = self.n_eval_episodes
            self.base_cfg.LOG_INTERVAL = self.log_interval
        else:
            self.base_cfg = exp_cfg

        if not isinstance(env_cls, list):
            self.env_classes_lst = list([env_cls, ])
        else:
            self.env_classes_lst = env_cls
        assert len(self.env_classes_lst) == len(env_kwargs), \
            "Error: list of env kwargs is not equal env_cls list"

        if not isinstance(agents_cls, list):
            self.agents_classes_lst = list([agents_cls, ])
        else:
            self.agents_classes_lst = agents_cls
        assert len(self.agents_classes_lst) == len(agents_kwargs), \
            "Error: list of agents kwargs is not equal agents list"

        if not isinstance(env_kwargs, list):
            env_kwargs.update({'deterministic': self.deterministic})
            self.env_kwargs_lst = list([env_kwargs, ])
        else:
            self.env_kwargs_lst = env_kwargs
            for kwargs in self.env_kwargs_lst:
                kwargs.update({'deterministic': self.deterministic})

        if not isinstance(agents_kwargs, list):
            self.update_agent_kwargs(agents_kwargs)
            self.agents_kwargs_lst = list([agents_kwargs, ])
        else:
            self.agents_kwargs_lst = agents_kwargs
            for kwargs in self.agents_kwargs_lst:
                self.update_agent_kwargs(kwargs)

        if agents_n_env is None:
            self.agents_n_env = []
            for ix in range(len(self.agents_classes_lst)):
                self.agents_n_env.append(self.policy_n_env_default.get(self.agents_classes_lst[ix], 1))
        elif not isinstance(agents_n_env, list):
            self.agents_n_env = [agents_n_env for _ in range(len(agents_kwargs))]
        else:
            self.agents_n_env = agents_n_env
            assert len(self.agents_n_env) == len(agents_kwargs), \
                "Error: list of agents n_env is not equal agents list"

        logger.info(f'{self.__class__.__name__}: Initialize environment...')

        self.agents_kwargs = agents_kwargs
        self.agents_obj: list = []
        self.agents_cfg: list = []
        self.train_vecenv_lst: list = []
        self.eval_vecenv_lst: list = []
        self.cache_manager: CacheManager = cache_manager_obj
        self.eval_cache_manager: CacheManager = eval_cache_manager_obj
        # self.init_agents()

    def update_agent_kwargs(self, agent_kwargs):
        agent_kwargs.update({'tensorboard_log': os.path.join(self.base_cfg.EXPERIMENT_PATH, 'TB')})
        return agent_kwargs

    def create_exp_dirs(self, cfg: LABConfig):
        dirs = dict()
        dirs['tb'] = os.path.join(cfg.EXPERIMENT_PATH, 'TB')
        dirs['exp'] = os.path.join(cfg.EXPERIMENT_PATH,
                                   cfg.ENV_NAME,
                                   cfg.ALGO,
                                   cfg.EXP_ID)

        dirs['training'] = os.path.join(dirs['exp'], 'training')
        dirs['evaluation'] = os.path.join(dirs['exp'], 'evaluation')
        dirs['best'] = os.path.join(dirs['exp'], 'best')
        os.makedirs(dirs['tb'], exist_ok=True)
        os.makedirs(dirs['training'], exist_ok=True)
        os.makedirs(dirs['evaluation'], exist_ok=True)

        return dirs

    def get_agent_requisite(self, agent_num):
        agent_obj = self.agents_obj[agent_num]
        agent_cfg = self.agents_cfg[agent_num]
        agent_kwargs = self.agents_kwargs[agent_num]
        return agent_obj, agent_cfg, agent_kwargs

    def learn_agent(self, ix):
        agent_obj, agent_cfg, agent_kwargs = self.get_agent_requisite(ix)
        # ConfigMethods.save_config(agent_cfg, os.path.join(agent_cfg.DIRS['exp'], f'{agent_cfg.FILENAME}_cfg.json'))
        # ConfigMethods.save_config(agent_kwargs,
        #                           os.path.join(agent_cfg.DIRS['exp'], f'{agent_cfg.FILENAME}_kwargs.json'))

        checkpoint_callback = CheckpointCallback(save_freq=self.eval_freq,
                                                 save_path=agent_cfg.DIRS["training"],
                                                 name_prefix=f'{agent_cfg.FILENAME}_chkp',
                                                 )
        eval_callback = LabEvalCallback(self.eval_vecenv_lst[ix],
                                        best_model_save_path=agent_cfg.DIRS["best"],
                                        n_eval_episodes=self.n_eval_episodes,
                                        log_path=agent_cfg.DIRS["evaluation"],
                                        eval_freq=self.eval_freq,
                                        deterministic=self.deterministic
                                        )
        # Create the callback list
        callbacks = CallbackList([checkpoint_callback, eval_callback])

        agent_obj.learn(total_timesteps=self.total_timesteps,
                        callback=callbacks,
                        log_interval=self.log_interval,
                        progress_bar=False,
                        tb_log_name=f'{agent_cfg.ENV_NAME}/{agent_cfg.ALGO}/{agent_cfg.OBS_TYPE}/{agent_cfg.EXP_ID}')

        agent_obj.save(path=os.path.join(f'{agent_cfg.DIRS["training"]}', agent_cfg.FILENAME))

    # def __get_env(self, env_cls, env_kwargs):
    #     env = env_cls(**env_kwargs)
    #     return env

    def init_agent(self, ix, save_config: bool = True):
        env = self.env_classes_lst[ix](**self.env_kwargs_lst[ix])
        logger.info(f'{self.__class__.__name__}: Environment check {self.env_classes_lst[ix].__class__.__name__}')
        check_env(env)
        self.base_cfg.ENV_NAME = f'{env.__class__.__name__}'

        train_env_kwargs = copy.deepcopy(self.env_kwargs_lst[ix])

        train_env_kwargs.update({'use_period': 'train', 'verbose': self.verbose, })

        """ Rlock object can't be copied """
        eval_env_kwargs = copy.deepcopy(train_env_kwargs)

        if self.env_wrapper != 'subproc':
            train_env_kwargs.update({'cache_obj': cache_manager_obj})
            eval_env_kwargs.update({'cache_obj': eval_cache_manager_obj})

        logger.info(f'{self.__class__.__name__}: Using vectorized env "{self.env_wrapper}"')

        eval_env_kwargs.update({'use_period': 'test',
                                'verbose': self.verbose,
                                'stable_cache_data_n': self.n_eval_episodes
                                })

        train_vec_env_kwargs = dict(env_id=self.env_classes_lst[ix],
                                    n_envs=self.agents_n_env[ix],
                                    seed=env.get_seed(),
                                    env_kwargs=train_env_kwargs,
                                    vec_env_cls=self.env_wrapper_cls if self.agents_n_env[ix] > 1 else DummyVecEnv)

        eval_vec_env_kwargs = dict(env_id=self.env_classes_lst[ix],
                                   n_envs=1,
                                   seed=env.get_seed(),
                                   env_kwargs=eval_env_kwargs,
                                   vec_env_cls=self.env_wrapper_cls if self.agents_n_env[ix] > 1 else DummyVecEnv)

        self.train_vecenv_lst.append(make_vec_env(**train_vec_env_kwargs))
        self.eval_vecenv_lst.append(make_vec_env(**eval_vec_env_kwargs))

        agent_kwargs: dict = copy.deepcopy(self.agents_kwargs[ix])

        agent_obj = self.agents_classes_lst[ix](env=self.train_vecenv_lst[ix],
                                                **deserialize_kwargs(agent_kwargs, lab_serializer))
        logger.info(
            f'{self.__class__.__name__}: Creating agent: #{ix:02d} {agent_obj.__class__.__name__}')

        agent_cfg = LABConfig(**asdict(self.base_cfg))
        agent_cfg.ALGO = agent_obj.__class__.__name__
        agent_cfg.OBS_TYPE = train_env_kwargs.get('observation_type', None)
        agent_cfg.AGENTS_N_ENV = self.agents_n_env[ix]

        assert agent_cfg.OBS_TYPE is not None, 'Error: something wrong with "observation type"'

        agent_cfg.FILENAME = f'{agent_obj.__class__.__name__}_{self.env_classes_lst[ix].__name__}_{self.total_timesteps}'

        self.agents_obj.append(agent_obj)
        dirs = self.create_exp_dirs(agent_cfg)
        agent_cfg.DIRS = dirs
        self.agents_cfg.append(agent_cfg)
        logger.debug(f'{self.__class__.__name__}: Initialized agent: #{ix:02d} {agent_obj.__class__.__name__}')
        if save_config:
            ConfigMethods.save_config(agent_cfg, os.path.join(agent_cfg.DIRS['exp'], f'{agent_cfg.FILENAME}_cfg.json'))
            ConfigMethods.save_config(self.agents_kwargs[ix],
                                      os.path.join(agent_cfg.DIRS['exp'], f'{agent_cfg.FILENAME}_kwargs.json'))
            ConfigMethods.save_config(self.env_kwargs_lst[ix],
                                      os.path.join(agent_cfg.DIRS['exp'], f'{agent_cfg.FILENAME}_env_kwargs.json'))
        del env

    def init_agents(self):
        for ix in range(len(self.agents_classes_lst)):
            self.init_agent(ix)

    def learn(self):
        self.init_agents()
        for ix in range(len(self.agents_obj)):
            logger.info(
                f'{self.__class__.__name__}: Start learning agent #{ix:02d}: {self.agents_obj[ix].__class__.__name__}\n')
            self.learn_agent(ix)
            self.evaluate_agent(ix)

    def evaluate_agent(self, ix, verbose=1):
        agent_obj, agent_cfg, agent_kwargs = self.get_agent_requisite(ix)

        """ Create independent evaluation env """
        eval_env_kwargs = copy.deepcopy(self.env_kwargs_lst[ix])
        eval_env_kwargs.update({'use_period': 'test', 'verbose': verbose, 'stable_cache_data_n': self.n_eval_episodes})
        eval_vec_env_kwargs = dict(env_id=self.env_classes_lst[ix],
                                   n_envs=1,
                                   seed=42,
                                   env_kwargs=eval_env_kwargs,
                                   vec_env_cls=DummyVecEnv)

        eval_vec_env = make_vec_env(**eval_vec_env_kwargs)

        # filename = agent_cfg.FILENAME
        agent_obj.load(path=os.path.join(f'{agent_cfg.DIRS["best"]}', 'best_model'), env=eval_vec_env)
        result = lab_evaluate_policy(agent_obj,
                                     agent_obj.get_env(),
                                     n_eval_episodes=self.n_eval_episodes,
                                     deterministic=self.deterministic,
                                     return_episode_rewards=True)

        self.show_result(result, agent_cfg, save_csv=True)

    def test_agent(self, ix=0, filename: str = 'best_model', verbose=1):
        """ Create independent evaluation env """
        eval_env_kwargs = copy.deepcopy(self.env_kwargs_lst[ix])
        eval_env_kwargs.update({'use_period': 'test',
                                'verbose': verbose,
                                'stable_cache_data_n': self.n_eval_episodes})
        if self.env_wrapper != 'subproc':
            eval_env_kwargs.update({'cache_obj': eval_cache_manager_obj})

        eval_vec_env_kwargs = dict(env_id=self.env_classes_lst[ix],
                                   n_envs=1,
                                   seed=42,
                                   env_kwargs=eval_env_kwargs,
                                   vec_env_cls=DummyVecEnv)

        eval_vec_env = make_vec_env(**eval_vec_env_kwargs)

        logger.info(
            f'{self.__class__.__name__}: Creating agent: #{ix:02d} {self.agents_classes_lst[ix].__class__.__name__}')
        # agent_kwargs: dict = copy.deepcopy(self.agents_kwargs[ix])
        # _ = agent_kwargs.pop('action_noise')

        agent_cfg = LABConfig(**asdict(self.base_cfg))

        if filename != 'best_model':
            agent_obj = self.agents_classes_lst[ix].load(path=os.path.join(f'{agent_cfg.DIRS["training"]}', filename),
                                                         env=eval_vec_env)
        else:
            agent_obj = self.agents_classes_lst[ix].load(path=os.path.join(f'{agent_cfg.DIRS["best"]}', filename),
                                                         env=eval_vec_env)

        result = lab_evaluate_policy(agent_obj,
                                     agent_obj.get_env(),
                                     n_eval_episodes=self.n_eval_episodes,
                                     deterministic=self.deterministic,
                                     return_episode_rewards=True)
        self.show_result(result, agent_cfg, save_csv=False)

    def show_result(self, result, agent_cfg, save_csv=True):
        result_df = pd.DataFrame(data={'reward': result[0], 'ep_length': result[1], 'ep_pnl': result[2]})
        result_df = result_df.astype({"reward": float, "ep_length": int, 'ep_pnl': float})
        total_df = result_df.groupby(result_df.index // self.n_eval_episodes).mean()
        msg = f'{self.__class__.__name__}: Evaluation result on BEST model:\n{result_df.to_string()}\n{total_df.to_string()}'
        logger.info(msg)
        if save_csv:
            result_df.to_csv(os.path.join(f'{agent_cfg.DIRS["evaluation"]}', f'{agent_cfg.FILENAME}.csv'))

    # TODO Load and Learn again
    def loaded_learn(self, ix=0,
                     filename: Union[str, int] = 'best_model',
                     render_mode: Union[str, None] = None,
                     reset_num_timesteps: bool = False,
                     total_timesteps: Union[int, None] = None,
                     verbose=1):

        # agent_kwargs: dict = copy.deepcopy(self.agents_kwargs[ix])

        # agent_kwargs: dict = copy.deepcopy(self.agents_kwargs[ix])
        # _ = agent_kwargs.pop('action_noise')
        agent_cfg = LABConfig(**asdict(self.base_cfg))

        train_env_kwargs = copy.deepcopy(self.env_kwargs_lst[ix])
        train_env_kwargs.update({'use_period': 'train', 'verbose': self.verbose, })

        """ Create independent evaluation env """
        eval_env_kwargs = copy.deepcopy(self.env_kwargs_lst[ix])
        eval_env_kwargs.update({'use_period': 'test',
                                'verbose': verbose,
                                'stable_cache_data_n': self.n_eval_episodes,
                                'render_mode': render_mode})

        if self.env_wrapper != 'subproc':
            eval_env_kwargs.update({'cache_obj': eval_cache_manager_obj})

        train_vec_env_kwargs = dict(env_id=self.env_classes_lst[ix],
                                    n_envs=self.agents_n_env[ix],
                                    seed=self.seed,
                                    env_kwargs=train_env_kwargs,
                                    vec_env_cls=self.env_wrapper_cls if self.agents_n_env[ix] > 1 else DummyVecEnv)
        # vec_env_cls=DummyVecEnv)

        eval_vec_env_kwargs = dict(env_id=self.env_classes_lst[ix],
                                   n_envs=1,
                                   seed=self.seed,
                                   env_kwargs=eval_env_kwargs,
                                   vec_env_cls=self.env_wrapper_cls)

        eval_vec_env = make_vec_env(**eval_vec_env_kwargs)

        train_vec_env = make_vec_env(**train_vec_env_kwargs)
        # self.eval_vecenv_lst.append(make_vec_env(**eval_vec_env_kwargs))

        # agent_kwargs: dict = copy.deepcopy(self.agents_kwargs[ix])

        # agent_kwargs: dict = copy.deepcopy(self.agents_kwargs[ix])
        # _ = agent_kwargs.pop('action_noise')

        # agent_cfg = LABConfig(**asdict(self.base_cfg))

        if isinstance(filename, int):
            filename = f'{agent_cfg.FILENAME}_chkp_{filename}_steps'

        if filename != 'best_model':
            agent_obj = self.agents_classes_lst[ix].load(path=os.path.join(f'{agent_cfg.DIRS["training"]}', filename),
                                                         env=train_vec_env)
            path_filename = os.path.join(f'{agent_cfg.DIRS["training"]}', filename)
        else:
            agent_obj = self.agents_classes_lst[ix].load(path=os.path.join(f'{agent_cfg.DIRS["best"]}', filename),
                                                         env=train_vec_env)
            path_filename = os.path.join(f'{agent_cfg.DIRS["best"]}', 'best_model')

        logger.info(
            f'{self.__class__.__name__}: Creating agent: #{ix:02d} {agent_obj.__class__.__name__}')

        # agent_obj = self.agents_classes_lst[ix](env=self.train_vecenv_lst[ix],
        #                                         **deserialize_kwargs(agent_kwargs, lab_serializer))

        if reset_num_timesteps:
            """ Setting new exp """
            new_agent_cfg = copy.deepcopy(agent_cfg)
            new_agent_cfg.EXP_ID = get_exp_id()
            # new_agent_cfg = LABConfig()
            # new_agent_cfg.EXPERIMENT_PATH = agent_cfg.EXPERIMENT_PATH
            if total_timesteps is None:
                new_agent_cfg.TOTAL_TIMESTEPS = agent_cfg.TOTAL_TIMESTEPS
            else:
                new_agent_cfg.TOTAL_TIMESTEPS = total_timesteps
                self.total_timesteps = total_timesteps

            # new_agent_cfg.EVAL_FREQ = agent_cfg.EVAL_FREQ
            # new_agent_cfg.ENV_WRAPPER = agent_cfg.ENV_WRAPPER
            # new_agent_cfg.DETERMINISTIC = agent_cfg.DETERMINISTIC
            # new_agent_cfg.N_EVAL_EPISODES = agent_cfg.N_EVAL_EPISODES
            # new_agent_cfg.LOG_INTERVAL = agent_cfg.LOG_INTERVAL
            #
            # new_agent_cfg.ALGO = agent_obj.__class__.__name__
            # new_agent_cfg.ENV_NAME = agent_cfg.ENV_NAME
            # new_agent_cfg.OBS_TYPE = train_env_kwargs.get('observation_type', None)
            # new_agent_cfg.AGENTS_N_ENV = agent_cfg.AGENTS_N_ENV

            # assert new_agent_cfg.OBS_TYPE is not None, 'Error: something wrong with "observation type"'
            new_agent_cfg.FILENAME = f'{agent_obj.__class__.__name__}_{self.env_classes_lst[ix].__name__}_{self.total_timesteps}'
            dirs = self.create_exp_dirs(new_agent_cfg)
            new_agent_cfg.DIRS = dirs

            ConfigMethods.save_config(new_agent_cfg,
                                      os.path.join(new_agent_cfg.DIRS['exp'], f'{new_agent_cfg.FILENAME}_cfg.json'))
            import shutil
            shutil.copy(os.path.join(agent_cfg.DIRS['exp'], f'{agent_cfg.FILENAME}_kwargs.json'),
                        os.path.join(new_agent_cfg.DIRS['exp'], f'{new_agent_cfg.FILENAME}_kwargs.json'))
            shutil.copy(os.path.join(agent_cfg.DIRS['exp'], f'{agent_cfg.FILENAME}_env_kwargs.json'),
                        os.path.join(new_agent_cfg.DIRS['exp'], f'{new_agent_cfg.FILENAME}_env_kwargs.json'))
            # ConfigMethods.save_config(self.agents_kwargs[ix],
            #                           os.path.join(agent_cfg.DIRS['exp'], f'{agent_cfg.FILENAME}_kwargs.json'))
            # ConfigMethods.save_config(self.env_kwargs_lst[ix],
            #                           os.path.join(agent_cfg.DIRS['exp'], f'{agent_cfg.FILENAME}_env_kwargs.json'))
            agent_cfg = new_agent_cfg

        checkpoint_callback = CheckpointCallback(save_freq=self.eval_freq * self.agents_n_env[ix],
                                                 save_path=agent_cfg.DIRS["training"],
                                                 name_prefix=f'{agent_cfg.FILENAME}_chkp',
                                                 )
        eval_callback = LabEvalCallback(eval_vec_env,
                                        best_model_save_path=agent_cfg.DIRS["best"],
                                        n_eval_episodes=self.n_eval_episodes,
                                        log_path=agent_cfg.DIRS["evaluation"],
                                        eval_freq=self.eval_freq * self.agents_n_env[ix],
                                        deterministic=self.deterministic
                                        )
        # Create the callback list
        callbacks = CallbackList([checkpoint_callback, eval_callback])

        agent_obj.set_env(train_vec_env)
        # env = agent_obj.get_env()

        agent_obj.learn(total_timesteps=self.total_timesteps,
                        callback=callbacks,
                        log_interval=self.log_interval,
                        progress_bar=False,
                        reset_num_timesteps=reset_num_timesteps,
                        tb_log_name=f'{agent_cfg.ENV_NAME}/{agent_cfg.ALGO}/{agent_cfg.OBS_TYPE}/{agent_cfg.EXP_ID}')

        agent_obj.save(path=os.path.join(f'{agent_cfg.DIRS["training"]}', agent_cfg.FILENAME))

    def backtesting_agent(self, ix=0, filename: Union[str, int] = 'best_model', render_mode=None, n_tests=10,
                          verbose=1):
        """ Create independent evaluation env """
        eval_env_kwargs = copy.deepcopy(self.env_kwargs_lst[ix])
        eval_env_kwargs.update({'use_period': 'test',
                                'verbose': verbose,
                                'stable_cache_data_n': self.n_eval_episodes,
                                'render_mode': render_mode})
        if self.env_wrapper != 'subproc':
            eval_env_kwargs.update({'cache_obj': eval_cache_manager_obj})

        eval_vec_env_kwargs = dict(env_id=self.env_classes_lst[ix],
                                   n_envs=1,
                                   seed=42,
                                   env_kwargs=eval_env_kwargs,
                                   vec_env_cls=DummyVecEnv)

        eval_vec_env = make_vec_env(**eval_vec_env_kwargs)

        # agent_kwargs: dict = copy.deepcopy(self.agents_kwargs[ix])
        # _ = agent_kwargs.pop('action_noise')

        agent_cfg = LABConfig(**asdict(self.base_cfg))

        if isinstance(filename, int):
            filename = f'{agent_cfg.FILENAME}_chkp_{filename}_steps'
        if filename != 'best_model':
            agent_obj = self.agents_classes_lst[ix].load(path=os.path.join(f'{agent_cfg.DIRS["training"]}', filename),
                                                         env=eval_vec_env)
            path_filename = os.path.join(f'{agent_cfg.DIRS["training"]}', filename)
        else:
            agent_obj = self.agents_classes_lst[ix].load(path=os.path.join(f'{agent_cfg.DIRS["best"]}', filename),
                                                         env=eval_vec_env)
            path_filename = os.path.join(f'{agent_cfg.DIRS["best"]}', 'best_model')

        logger.info(
            f'{self.__class__.__name__}: Creating agent: #{ix:02d} {agent_obj.__class__.__name__}')

        env = agent_obj.get_env()
        unwrapped_env = self.get_base_env(env, self.env_classes_lst[ix])

        for ix in range(n_tests):
            unwrapped_env.set_render_output(f'{path_filename}_{ix}')
            episode_rewards = .0
            obs = env.reset()
            while True:
                action, _states = agent_obj.predict(obs)
                obs, rewards, dones, info = env.step(action)
                episode_rewards += rewards
                if dones.any():
                    unwrapped_env.render_all(unwrapped_env.get_last_render_df())
                    break

    def get_seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    @staticmethod
    def get_base_env(wrapped_env: Union[DummyVecEnv, SubprocVecEnv], env_class):
        env_tmp = wrapped_env
        while isinstance(env_tmp, (DummyVecEnv, SubprocVecEnv, Monitor)):
            if isinstance(env_tmp, (DummyVecEnv, SubprocVecEnv)):
                env_tmp = env_tmp.envs[0]
            elif isinstance(env_tmp, Monitor):
                env_tmp = env_tmp.env
            if isinstance(env_tmp, env_class):
                return env_tmp
        return None

    @classmethod
    def load_agent(cls, json_path_filename, verbose=1):
        config_kwargs = ConfigMethods.load_config(json_path_filename)
        agent_cfg = LABConfig(**config_kwargs)
        env_kwargs = ConfigMethods.load_config(
            os.path.join(agent_cfg.DIRS['exp'], f'{agent_cfg.FILENAME}_env_kwargs.json'))
        agent_kwargs = ConfigMethods.load_config(
            os.path.join(agent_cfg.DIRS['exp'], f'{agent_cfg.FILENAME}_kwargs.json'))

        if isinstance(agent_kwargs.get('train_freq', None), list):
            agent_kwargs.update({'train_freq': tuple(agent_kwargs['train_freq'])})
        lab_kwargs: dict = {'env_cls': [deserialize_kwargs(agent_cfg.ENV_NAME, lab_serializer)],
                            'agents_cls': [deserialize_kwargs(agent_cfg.ALGO, lab_serializer)],
                            'env_kwargs': [deserialize_kwargs(env_kwargs, lab_serializer)],
                            'agents_kwargs': [deserialize_kwargs(agent_kwargs, lab_serializer)],
                            'agents_n_env': agent_cfg.AGENTS_N_ENV,
                            'env_wrapper': agent_cfg.ENV_WRAPPER,
                            'total_timesteps': agent_cfg.TOTAL_TIMESTEPS,
                            'experiment_path': agent_cfg.DIRS['exp'],
                            'eval_freq': int(agent_cfg.EVAL_FREQ // agent_cfg.AGENTS_N_ENV),
                            'checkpoint_num': int(agent_cfg.TOTAL_TIMESTEPS // agent_cfg.EVAL_FREQ),
                            'n_eval_episodes': agent_cfg.N_EVAL_EPISODES,
                            'log_interval': agent_cfg.LOG_INTERVAL,
                            'deterministic': agent_cfg.DETERMINISTIC,
                            'verbose': verbose,
                            }

        new_instance = cls.__new__(cls)
        deserialized_kwargs = deserialize_kwargs(lab_kwargs, lab_serializer)
        cls.__init__(new_instance, exp_cfg=agent_cfg, **deserialized_kwargs)
        return new_instance

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
