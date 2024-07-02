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
from dataclasses import asdict, dataclass, field, make_dataclass
from rllab.configtools import ConfigMethods

__version__ = 0.012

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
    def __init__(self, env_cls, agents_cls: Union[object, List[object]], total_timesteps: int = 500_000,
                 experiment_path: str = './', checkpoint_num: int = 20, eval_freq: int = 50_000,
                 n_eval_episodes: int = 10, env_kwargs: Union[dict, None] = None,
                 agents_kwargs: Union[list, None] = None,
                 ):
        """
        Get all arguments to instantiate all classes and object inside the Lab

        Args:
            env_cls (class):      environment classes
            agents_cls (class):     agents classes
            experiment_path (str):  path
            env_kwargs (dict):      kwargs for environment
            agents_kwargs (list):   kwargs for each agent
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

        msg = (f'{self.__class__.__name__}: Initialize environment, loading OHLCV and AI indicators data '
               f'from database, wait a few minutes')
        logger.info(msg)

        self.env = env_cls(**env_kwargs)
        logger.info(f'{self.__class__.__name__}: Environment check {self.env_cls.__class__.__name__}')
        check_env(self.env)

        # self.env = make_vec_env(self.env_cls, n_envs=1, seed=0, env_kwargs=self.env_kwargs)
        """ Creating the copy of the base environment for evaluation """
        logger.info(f'{self.__class__.__name__}: Creating copy of env for evaluation')

        eval_env_kwargs = copy.deepcopy(self.env_kwargs)
        eval_env_kwargs.update({'use_period': 'test', 'verbose': 1, 'log_interval': 1})
        self.eval_env = make_vec_env(self.env_cls, n_envs=1, seed=42, env_kwargs=eval_env_kwargs)

        self.base_cfg.ENV_NAME = self.env.__class__.__name__

        if not isinstance(agents_cls, list):
            self.agents_classes_list = list([agents_cls, ])
        else:
            self.agents_classes_list = agents_cls
        assert len(self.agents_classes_list) == len(agents_kwargs), \
            "Error: list of agents kwargs is not equal agents list"
        self.agents_kwargs = agents_kwargs
        self.agents_obj: list = []
        self.agents_cfg: list = []
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

    def learn_agent(self, agent_num):
        agent_obj, agent_cfg, agent_kwargs = self.get_agent_requisite(agent_num)
        # filename = f'{agent_obj.__class__.__name__}_{self.env_cls.__name__}_{self.total_timesteps}'
        ConfigMethods.save_config(agent_cfg, os.path.join(agent_cfg.DIRS['exp'], f'{agent_cfg.FILENAME}_cfg.json'))
        ConfigMethods.save_config(agent_kwargs,
                                  os.path.join(agent_cfg.DIRS['exp'], f'{agent_cfg.FILENAME}_kwargs.json'))

        checkpoint_callback = CheckpointCallback(save_freq=int(self.total_timesteps // self.checkpoint_num),
                                                 save_path=agent_cfg.DIRS["training"],
                                                 name_prefix=f'{agent_cfg.FILENAME}_chkp',
                                                 )
        # eval_env = Monitor(self.eval_env, agent_cfg.DIRS["evaluation"])
        eval_callback = EvalCallback(self.eval_env,
                                     best_model_save_path=agent_cfg.DIRS["best"],
                                     n_eval_episodes=10,
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

    def evaluate_agent(self, agent_num):
        agent_obj, agent_cfg, agent_kwargs = self.get_agent_requisite(agent_num)
        # filename = agent_cfg.FILENAME
        agent_obj.load(path=os.path.join(f'{agent_cfg.DIRS["training"]}', agent_cfg.FILENAME))
        result = evaluate_policy(agent_obj, self.eval_env, n_eval_episodes=self.n_eval_episodes,
                                 return_episode_rewards=True)
        result = pd.DataFrame(data=result, index=['reward', 'ep_length']).astype(int)
        msg = (f'{self.__class__.__name__}: Agent #{agent_num:02d}: {agent_obj.__class__.__name__} '
               f'Evaluation result:\n {result.to_string()}')
        logger.debug(msg)

    def init_agent(self, ix):

        agent_obj = self.agents_classes_list[ix](env=self.env, **self.agents_kwargs[ix])
        agent_cfg = LABConfig(**asdict(self.base_cfg))

        agent_cfg.__class__ = make_dataclass(f'{LABConfig.__name__}_{agent_obj.__class__.__name__}',
                                             fields=[('ALGO', str, agent_obj.__class__.__name__),
                                                     ('DIRS', dict, field(default_factory=dict, init=False)),
                                                     ('FILENAME', str,
                                                      f'{agent_obj.__class__.__name__}_{self.env_cls.__name__}_{self.total_timesteps}')
                                                     ],
                                             bases=(LABConfig,))

        self.agents_obj.append(agent_obj)
        dirs = self.create_exp_dirs(agent_cfg)
        agent_cfg.DIRS = dirs
        self.agents_cfg.append(agent_cfg)
        logger.debug(f'{self.__class__.__name__}: Initialized agent: #{ix:02d} {agent_obj.__class__.__name__}')

    def init_agents(self):
        for ix, (agent_cls, agent_kwargs) in enumerate(zip(self.agents_classes_list, self.agents_kwargs)):
            agent_obj = agent_cls(env=self.env, **agent_kwargs)
            agent_cfg = LABConfig(**asdict(self.base_cfg))
            agent_cfg.__class__ = make_dataclass(f'{LABConfig.__name__}_{agent_obj.__class__.__name__}',
                                                 fields=[('ALGO', str, agent_obj.__class__.__name__),
                                                         ('DIRS', dict, field(default_factory=dict, init=False)),
                                                         ('FILENAME', str,
                                                          f'{agent_obj.__class__.__name__}_{self.env_cls.__name__}_{self.total_timesteps}')
                                                         ],
                                                 bases=(LABConfig,))
            self.agents_obj.append(agent_obj)
            dirs = self.create_exp_dirs(agent_cfg)
            agent_cfg.DIRS = dirs
            self.agents_cfg.append(agent_cfg)
            logger.debug(f'{self.__class__.__name__}: Initialized agent: #{ix:02d} {agent_obj.__class__.__name__}')

    def learn(self):
        for ix in range(len(self.agents_obj)):
            logger.info(
                f'{self.__class__.__name__}: Start learning agent #{ix:02d}: {self.agents_obj[ix].__class__.__name__}')
            self.learn_agent(ix)
            self.evaluate_agent(ix)
