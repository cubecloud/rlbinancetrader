import copy
import os
import logging
import datetime
from pytz import timezone

import pandas as pd

from typing import List, Union, Dict

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from dataclasses import asdict, dataclass, field, make_dataclass
from rllab.configtools import ConfigMethods

__version__ = 0.016

TZ = timezone('Europe/Moscow')

logger = logging.getLogger()


def get_exp_id() -> str:
    return f'exp-{datetime.datetime.now(TZ).strftime("%d%m-%H%M%S")}'


@dataclass(init=True)
class LABConfig:
    EXP_ID: str = field(default_factory=get_exp_id, init=True)
    EXPERIMENT_PATH: str = str()
    ENV_NAME: str = str()


class LabBase:
    def __init__(self, env_cls: Union[object, List[object,]], agents_cls: Union[object, List[object]],
                 env_kwargs: Union[List[dict,], dict], agents_kwargs: Union[List[dict,], dict],
                 total_timesteps: int = 500_000, experiment_path: str = './',
                 checkpoint_num: int = 20,
                 eval_freq: int = 50_000, n_eval_episodes: int = 10,

                 ):
        """
        Get all arguments to instantiate all classes and object inside the Lab

        Args:
            env_cls (class or list):        environment classes
            agents_cls (class or list):     agents classes
            env_kwargs (dict or list):      kwargs for environment
            agents_kwargs (dict or list):   kwargs for each agent
            experiment_path (str):  path
            total_timesteps (int):  total timesteps
        """
        self.eval_freq = eval_freq
        self.checkpoint_num = checkpoint_num
        self.n_eval_episodes = n_eval_episodes
        self.experiment_path = experiment_path
        self.total_timesteps = total_timesteps
        self.env_cls = env_cls
        self.env_kwargs = env_kwargs
        self.base_cfg = LABConfig()
        self.base_cfg.EXPERIMENT_PATH = experiment_path

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
            self.env_kwargs_lst = list([env_kwargs, ])
        else:
            self.env_kwargs_lst = env_kwargs

        if not isinstance(agents_kwargs, list):
            self.agents_kwargs_lst = list([agents_kwargs, ])
        else:
            self.agents_kwargs_lst = agents_kwargs

        logger.info(f'{self.__class__.__name__}: Initialize environment...')

        self.agents_kwargs = agents_kwargs
        self.agents_obj: list = []
        self.agents_cfg: list = []
        self.train_vecenv_lst: list = []
        self.eval_vecenv_lst: list = []

        self.init_agents()

    def create_exp_dirs(self, cfg: LABConfig):
        dirs = dict()
        dirs['exp'] = os.path.join(cfg.EXPERIMENT_PATH,
                                   cfg.ENV_NAME,
                                   cfg.ALGO,
                                   cfg.EXP_ID)

        dirs['training'] = os.path.join(dirs['exp'], 'training')
        dirs['evaluation'] = os.path.join(dirs['exp'], 'evaluation')
        dirs['best'] = os.path.join(dirs['exp'], 'best')
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
        ConfigMethods.save_config(agent_cfg, os.path.join(agent_cfg.DIRS['exp'], f'{agent_cfg.FILENAME}_cfg.json'))
        ConfigMethods.save_config(agent_kwargs,
                                  os.path.join(agent_cfg.DIRS['exp'], f'{agent_cfg.FILENAME}_kwargs.json'))

        checkpoint_callback = CheckpointCallback(save_freq=int(self.total_timesteps // self.checkpoint_num),
                                                 save_path=agent_cfg.DIRS["training"],
                                                 name_prefix=f'{agent_cfg.FILENAME}_chkp',
                                                 )
        eval_callback = EvalCallback(self.eval_vecenv_lst[ix],
                                     best_model_save_path=agent_cfg.DIRS["best"],
                                     n_eval_episodes=self.n_eval_episodes,
                                     log_path=agent_cfg.DIRS["evaluation"],
                                     eval_freq=int(self.total_timesteps // self.checkpoint_num),
                                     deterministic=True
                                     )
        # Create the callback list
        callbacks = CallbackList([
            checkpoint_callback,
            eval_callback
        ])

        agent_obj.learn(total_timesteps=self.total_timesteps, callback=callbacks, progress_bar=True)
        agent_obj.save(path=os.path.join(f'{agent_cfg.DIRS["training"]}', agent_cfg.FILENAME))

    def evaluate_agent(self, ix):
        agent_obj, agent_cfg, agent_kwargs = self.get_agent_requisite(ix)
        # filename = agent_cfg.FILENAME
        agent_obj.load(path=os.path.join(f'{agent_cfg.DIRS["training"]}', agent_cfg.FILENAME))
        result = evaluate_policy(agent_obj, self.eval_vecenv_lst[ix], n_eval_episodes=self.n_eval_episodes,
                                 return_episode_rewards=True)
        result = pd.DataFrame(data=result, index=['reward', 'ep_length']).astype(int)
        msg = (f'{self.__class__.__name__}: Agent #{ix:02d}: {agent_obj.__class__.__name__} '
               f'Evaluation result:\n {result.to_string()}')
        logger.debug(msg)
        result.to_csv(os.path.join(f'{agent_cfg.DIRS["evaluation"]}', f'{agent_cfg.FILENAME}.csv'))

    def __get_env(self, env_cls, env_kwargs):
        env = env_cls(**env_kwargs)
        return env

    def init_agent(self, ix):
        n_envs = 1
        env = self.env_classes_lst[ix](**self.env_kwargs_lst[ix])
        logger.info(f'{self.__class__.__name__}: Environment check {self.env_classes_lst[ix].__class__.__name__}')
        check_env(env)
        self.base_cfg.ENV_NAME = env.__class__.__name__
        # del env
        # d_env = DummyVecEnv([lambda: env] * n_envs)
        self.train_vecenv_lst.append(env)
        # self.train_vecenv_lst.append(DummyVecEnv([self.__get_env(self.env_classes_lst[ix], self.env_kwargs_lst[ix])]))

        eval_env_kwargs = copy.deepcopy(self.env_kwargs_lst[ix])
        eval_env_kwargs.update({'use_period': 'test', 'reuse_data_prob': 0.00, 'verbose': 1, 'log_interval': 1})

        self.eval_vecenv_lst.append(
            make_vec_env(self.env_classes_lst[ix], n_envs=1, seed=42, env_kwargs=eval_env_kwargs))

        agent_obj = self.agents_classes_lst[ix](env=self.train_vecenv_lst[ix], **self.agents_kwargs[ix])

        agent_cfg = LABConfig(**asdict(self.base_cfg))

        agent_cfg.__class__ = make_dataclass(f'{LABConfig.__name__}_{agent_obj.__class__.__name__}',
                                             fields=[('ALGO', str, agent_obj.__class__.__name__),
                                                     ('DIRS', dict, field(default_factory=dict, init=False)),
                                                     ('FILENAME', str,
                                                      f'{agent_obj.__class__.__name__}_{self.env_classes_lst[ix].__name__}_{self.total_timesteps}')
                                                     ],
                                             bases=(LABConfig,))
        self.agents_obj.append(agent_obj)
        dirs = self.create_exp_dirs(agent_cfg)
        agent_cfg.DIRS = dirs
        self.agents_cfg.append(agent_cfg)
        logger.debug(f'{self.__class__.__name__}: Initialized agent: #{ix:02d} {agent_obj.__class__.__name__}')

    def init_agents(self):
        for ix in range(len(self.agents_classes_lst)):
            self.init_agent(ix)

    def learn(self):
        for ix in range(len(self.agents_obj)):
            logger.info(
                f'{self.__class__.__name__}: Start learning agent #{ix:02d}: {self.agents_obj[ix].__class__.__name__}')
            self.learn_agent(ix)
            self.evaluate_agent(ix)
