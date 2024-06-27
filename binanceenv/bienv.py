import math
import random
import logging
import gymnasium
import numpy as np
import pandas as pd
from gymnasium.spaces import Discrete
from gymnasium.spaces import Box

from indicators.flag import detect_flag, pivotid, pointpos

__version__ = 0.004

logger = logging.getLogger()


class IndicatorsObsSpaceContinuous:
    def __init__(self, ind_numbers):
        self.__observation_space = Box(low=0.0, high=1.0, shape=(1, ind_numbers), dtype=np.float32, seed=42)

    @property
    def observation_space(self):
        return self.__observation_space

    @observation_space.setter
    def observation_space(self, value):
        self.__observation_space = value


class BinanceEnvBase(gymnasium.Env):
    balance = 1000000
    reward = 0
    timecount = 0

    def __init__(self, _df: pd.DataFrame, render_mode=None):
        self.df = _df
        self.observation_space = IndicatorsObsSpaceContinuous(_df.shape[1]).observation_space
        self.action_space = Discrete(3, seed=42)  # {0, 1, 2}
        self._action_buy_hold_sell = ["продать", "владеть", "купить"]

        self.df['pivot'] = self.df.apply(lambda x: pivotid(self.df, x.name, 3, 3), axis=1)
        self.df['pointpos'] = self.df.apply(lambda row: pointpos(row), axis=1)

    def _get_obs(self, tick):
        return detect_flag(self.df, tick[0], 100, 3)  # self.observation_space.sample()

    def _get_info(self):
        return {"balance": self.balance}

    def step(self, action):
        terminated = False
        observation = self._get_obs(np.ndarray((1,), buffer=np.array([self.timecount]), dtype=int))
        old_balance = self.balance
        if action == 0:
            self.balance += self.df.iloc[self.timecount]["close"]
        if action == 2:
            self.balance -= self.df.iloc[self.timecount]["close"]
        self.reward += self.balance - old_balance
        what_action = self._action_buy_hold_sell[action]
        if self.timecount % 100 == 0:
            print("\tobservation\t", observation, "\taction", action, f"\tбыло {old_balance:.4f} \tдействие|",
                  what_action, "\t", self.df.iloc[self.timecount]["close"], f"\tстало {self.balance:.4f}",
                  f"\treward {self.reward:.4f}")

        terminated = bool(self.balance <= 0)
        info = self._get_info()
        self.timecount += 1
        return observation, self.reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print(f"\tвознаграждение {self.reward:.2f}", f"\tбаланс {self.balance:.2f}", "\treset\n", "*" * 60)
        self.balance = 1000000
        self.reward = 0
        observation = self._get_obs(np.ndarray((1,), buffer=np.array([0]), dtype=int))
        info = self._get_info()
        return observation, info

    def close(self):
        print(f"\tвознаграждение {self.reward:.2f}", f"\tбаланс {self.balance:.2f}", "\tclose\n", "*" * 60)


class BinanceEnv(gymnasium.Env):
    balance = 1000000
    reward = 0
    timecount = 0

    def __init__(self, render_mode=None, ohlcv_df=None):
        self.observation_space = Discrete(2, seed=42)  # {0, 1}
        self.action_space = Discrete(3, seed=42)  # {0, 1, 2}
        self._action_buy_hold_sell = ["продать", "владеть", "купить"]
        self.df = ohlcv_df
        self.df['pivot'] = self.df.apply(lambda x: pivotid(self.df, x.name, 3, 3), axis=1)
        self.df['pointpos'] = self.df.apply(lambda row: pointpos(row), axis=1)

    def _get_obs(self, tick):
        return detect_flag(self.df, tick[0], 100, 3)  # self.observation_space.sample()

    def _get_info(self):
        return {"balance": self.balance}

    def step(self, action):
        terminated = False
        observation = self._get_obs(np.ndarray((1,), buffer=np.array([self.timecount]), dtype=int))
        old_balance = self.balance
        if action == 0:
            self.balance += self.df.iloc[self.timecount]["close"]
        if action == 2:
            self.balance -= self.df.iloc[self.timecount]["close"]
        self.reward += self.balance - old_balance
        what_action = self._action_buy_hold_sell[action]
        if self.timecount % 100 == 0:
            print("\tobservation\t", observation, "\taction", action, f"\tбыло {old_balance:.4f} \tдействие|",
                  what_action, "\t", self.df.iloc[self.timecount]["close"], f"\tстало {self.balance:.4f}",
                  f"\treward {self.reward:.4f}")

        terminated = bool(self.balance <= 0)
        info = self._get_info()
        self.timecount += 1
        return observation, self.reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print(f"\tвознаграждение {self.reward:.2f}", f"\tбаланс {self.balance:.2f}", "\treset\n", "*" * 60)
        self.balance = 1000000
        self.reward = 0
        observation = self._get_obs(np.ndarray((1,), buffer=np.array([0]), dtype=int))
        info = self._get_info()
        return observation, info

    def close(self):
        print(f"\tвознаграждение {self.reward:.2f}", f"\tбаланс {self.balance:.2f}", "\tclose\n", "*" * 60)
