import sys
import math
import random
import logging
import gymnasium
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.spaces import Box
from gymnasium.utils import seeding

from datawizard.dataprocessor import IndicatorProcessor
from datawizard.dataprocessor import Constants

__version__ = 0.013

logger = logging.getLogger()


class IndicatorsSpace:
    def __init__(self, ind_num):
        self.__observation_space = Box(low=0.0, high=1.0, shape=(ind_num,), dtype=np.float32, seed=42)
        self.name = 'indicators'

    @property
    def observation_space(self):
        return self.__observation_space

    @observation_space.setter
    def observation_space(self, value):
        self.__observation_space = value


class IndicatorsAndAssetsSpace:
    def __init__(self, ind_num, assets_num):
        # low = np.zeros((assets_num + ind_num,))
        # high = np.ones((assets_num + ind_num,))
        # low[:assets_num] = 0.0
        # high[:assets_num] = 1.0
        self.__observation_space = Box(low=0.0, high=1.0, shape=(ind_num + assets_num,), dtype=np.float32, seed=42)
        # self.__observation_space = Box(low=low, high=high, dtype=np.float32, seed=42)
        self.name = 'indicators_assets'

    @property
    def observation_space(self):
        return self.__observation_space

    @observation_space.setter
    def observation_space(self, value):
        self.__observation_space = value


class IndicatorsAndPNLSpace:
    def __init__(self, ind_num, pnl_num):
        low = np.zeros((pnl_num + ind_num,))
        high = np.ones((pnl_num + ind_num,))
        low[:pnl_num] = -1.0
        high[:pnl_num] = 3.0
        self.__observation_space = Box(low=low, high=high, dtype=np.float32, seed=42)
        self.name = 'indicators_pnl'

    @property
    def observation_space(self):
        return self.__observation_space

    @observation_space.setter
    def observation_space(self, value):
        self.__observation_space = value


class DiscreteActionSpace:
    def __init__(self, n_action):
        self.__action_space = Discrete(n_action, seed=42)  # {0, 1, 2}
        self.name = 'discrete'

    @property
    def action_space(self):
        return self.__action_space

    @action_space.setter
    def action_space(self, value):
        self.__action_space = value


class BoxActionSpace:
    def __init__(self, n_action):
        self.__action_space = Box(low=np.array([0, 0]), high=np.array([n_action, 1]), dtype=np.float32)
        self.name = 'box'

    @property
    def action_space(self):
        return self.__action_space

    @action_space.setter
    def action_space(self, value):
        self.__action_space = value


class BinanceEnvBase(gymnasium.Env):
    count = 0
    target_balance: float = 100_000.
    target_symbol: str = 'USDT'

    coin_balance: float = 0.
    coin_symbol: str = 'BTC'
    minimum_coin_trade_amount = 0.00001

    reward: float = 0.
    timecount: int = 0
    commission: float = .002
    name = 'BinanceEnvBase'

    def __init__(self, data_processor_kwargs: dict, target_balance: float = 100_000., coin_balance: float = 0.,
                 pnl_stop: float = -0.5, verbose: int = 0, log_interval=500, max_lot_size=0.25,
                 observation_type='indicators', action_type='discrete', use_period='train', render_mode=None):

        BinanceEnvBase.count += 1
        self.idnum = int(BinanceEnvBase.count)
        self.data_processor_obj = IndicatorProcessor(**data_processor_kwargs)
        self.max_timesteps = self.data_processor_obj.max_timesteps
        self.max_lot_size = max_lot_size

        logger.info(f"{self.__class__.__name__} #{self.idnum}: MAX_TIMESTEPS = {self.max_timesteps}")
        self.use_period = use_period
        self.pnl_stop = pnl_stop
        self.log_interval = log_interval
        self.action_type = action_type

        self.initial_target_balance = target_balance
        self.initial_coin_balance = coin_balance
        self.target_balance = float(self.initial_target_balance)
        self.coin_balance = float(self.initial_coin_balance)
        self.account_value: list = []
        self.pnl: float = 0.

        self.verbose = verbose

        self.ohlcv_df, self.indicators_df = self.data_processor_obj.get_ohlcv_and_indicators_sample(
            index_type='target_time')

        self.__get_obs_func = None
        self._take_action_func = None
        if self.use_period == 'train':
            self._take_action_func = self._take_action_train
        else:
            self._take_action_func = self._take_action_test

        self.observation_space = self.get_observation_space(observation_type=observation_type)
        self.action_space = self.get_action_space(action_type=action_type)  # {0, 1, 2}

        self._action_buy_hold_sell = ["Sell", "Hold", "Buy"]
        self.old_total_assets: float = 0.
        self.initial_total_assets = self.initial_target_balance + (self.initial_coin_balance * self.price)

        self.actions_lst: list = []
        self.reset()
        self._seed()

    def __del__(self):
        BinanceEnvBase.count -= 1

    def get_observation_space(self, observation_type='indicators'):
        space_obj = IndicatorsSpace(self.indicators_df.shape[1])
        self.__get_obs_func = self._get_indicators_obs
        if observation_type == 'indicators_assets':
            space_obj = IndicatorsAndAssetsSpace(self.indicators_df.shape[1], 2)
            self.__get_obs_func = self._get_assets_indicators_obs
            if self.coin_balance == .0:
                self.coin_balance = 1e-7
        elif observation_type == 'indicators_pnl':
            space_obj = IndicatorsAndPNLSpace(self.indicators_df.shape[1], 1)
            self.__get_obs_func = self._get_pnl_indicators_obs
        observation_space = space_obj.observation_space
        self.name = f'{self.name}_{space_obj.name}'
        return observation_space

    def get_action_space(self, action_type='discrete'):
        space_obj = DiscreteActionSpace(3)
        if action_type == 'box':
            space_obj = BoxActionSpace(3)
        action_space = space_obj.action_space
        self.name = f'{self.name}_{space_obj.name}'
        return action_space

    @property
    def price(self):
        return self.ohlcv_df.iloc[self.timecount]['close']

    @property
    def total_assets(self):
        return self.target_balance + (self.coin_balance * self.price)

    @property
    def current_balance(self) -> str:
        current_balance = (
            f"CURRENT Assets => {self.target_symbol}: {self.target_balance:.4f}, {self.coin_symbol}: "
            f"{self.coin_balance:.4f}\tTotal assets: {self.total_assets} {self.target_symbol}")
        return current_balance

    def _get_obs(self):
        data = self.__get_obs_func()
        logger.debug(data)
        return data

    @staticmethod
    def logarithmic10_scaler(value):
        result = (1 / (1 + (math.e ** -(np.log10(value + 1))))) - 0.5
        return result

    def _get_assets_indicators_obs(self):
        target = self.logarithmic10_scaler(self.target_balance)
        coin = self.logarithmic10_scaler(self.coin_balance)
        return np.asarray(
            np.concatenate([self.indicators_df.iloc[self.timecount].values, [target, coin], ]),
            dtype=np.float32)

    def _get_pnl_indicators_obs(self):
        return np.asarray(np.concatenate([self.indicators_df.iloc[self.timecount].values, [self.pnl]]),
                          dtype=np.float32)

    def _get_indicators_obs(self):
        return np.asarray(self.indicators_df.iloc[self.timecount].values, dtype=np.float32)

    def _get_info(self):
        return {"name": self.name,
                # "balance": self.current_balance,
                # "start_datetime": self.ohlcv_df.index[0],
                # "end_datetime": self.ohlcv_df.index[-1],
                # "ohlcv_df": self.ohlcv_df,
                # "indicators_df": self.indicators_df
                }

    def _box_action_to_int(self, action):
        amount = action[1]
        # 0 - SELL, 1 - HOLD, 2 - BUY
        action = math.floor(action[0])
        # if action[0] < 1:
        #     action = 0  # SELL
        # elif action[0] < 2:
        #     action = 1  # HOLD
        # elif action[0] <= 3:
        #     action = 2  # BUY
        return action, amount

    def _take_action_test(self, action, amount):
        old_target_balance = float(self.target_balance)
        old_coin_balance = float(self.coin_balance)
        # action_commission = .0
        # action_symbol = self.target_symbol
        # size = .0
        # msg = str()
        if action == 0:  # sell
            # action_symbol = f'{self.coin_symbol}->{self.target_symbol}'
            # amount - value from Box action space. For Discrete action space it's equal 1.
            size = old_coin_balance * (1 - self.commission)
            if size > self.minimum_coin_trade_amount:
                size *= amount if amount > self.minimum_coin_trade_amount else self.minimum_coin_trade_amount
                action_commission = old_coin_balance * self.price * self.commission
                self.target_balance += (size * self.price) - action_commission
                self.coin_balance -= size
            # else:
            #     # penalty 10$ for trading without coin balance
            #     self.target_balance -= 10 if self.target_balance >= 10 else self.target_balance
            #     # msg = f'Penalty 10$: old coin balance: {old_coin_balance}, size: {size}, target balance: {self.target_balance}\n'

        if action == 2:  # buy
            # action_symbol = f'{self.target_symbol}->{self.coin_symbol}'
            # amount - value from Box action space. For Discrete action space it's equal 1.
            size = (old_target_balance * self.max_lot_size / self.price) * (1 - self.commission)
            if size > self.minimum_coin_trade_amount:
                size *= amount if amount > self.minimum_coin_trade_amount else self.minimum_coin_trade_amount
                action_commission = size / (1 - self.commission) * self.price
                self.target_balance -= ((size * self.price) + action_commission)
                self.coin_balance += size
            # else:
            #     # penalty 10$ for trading without coin balance
            #     self.coin_balance -= (10 / self.price) if self.coin_balance >= (
            #             10 / self.price) else self.coin_balance
            #     # msg = f'Penalty 10$: target balance: {self.target_balance}, size: {size}, coin balance: {self.coin_balance}\n'
        # what_action = self._action_buy_hold_sell[action]
        # self.account_value.append(self.total_assets)
        self.reward = self.total_assets - self.initial_total_assets
        self.pnl = ((self.target_balance + (self.coin_balance * self.price)) / self.initial_total_assets) - 1.

        # if self.verbose:
        #     if self.timecount % self.log_interval == 0 and self.verbose == 2:
        #         ohlcv = (
        #             f"{self.timecount}\t {self.ohlcv_df.index[self.timecount]} \t"
        #             f"open: \t{self.ohlcv_df.iloc[self.timecount]['open']:.2f} \t"
        #             f"high: \t{self.ohlcv_df.iloc[self.timecount]['high']:.2f} \t"
        #             f"low: \t{self.ohlcv_df.iloc[self.timecount]['low']:.2f} \t"
        #             f"close: \t{self.ohlcv_df.iloc[self.timecount]['close']:.2f} \t"
        #             f"volume: \t{self.ohlcv_df.iloc[self.timecount]['volume']:.2f}")
        #
        #         old_balance = (
        #             f"OLD Assets =>\t{self.target_symbol}: {old_target_balance:.4f}, {self.coin_symbol}: {old_coin_balance:.4f}"
        #             f"\tTotal assets(old): {(old_target_balance + (old_coin_balance * self.price)):.1f} {self.target_symbol}")
        #         msg = (f"{ohlcv}\n{msg}"
        #                f"\tAction num: {action}\t{old_balance} "
        #                f"\tACTION => {what_action}: size:{size:.4f}({amount:.3f}) {action_symbol}, commission: {action_commission:.6f}"
        #                f"\t{self.current_balance}\treward {self.reward:.2f}\tPNL {self.pnl:.5f}")
        #         logger.info(msg)

    def _take_action_train(self, action, amount):
        self._take_action_test(action, amount)

    def step(self, action):
        amount = 1.
        if self.action_type == 'box':
            action, amount = self._box_action_to_int(action)
        self._take_action_func(action, amount)
        self.actions_lst.append(action)
        observation = self._get_obs()

        terminated = bool(self.pnl < self.pnl_stop)

        info = self._get_info()

        self.old_total_assets = float(self.total_assets)
        # delay_modifier = (self.timecount / self.max_timesteps)
        # discounted_reward = self.reward * delay_modifier
        self.timecount += 1

        if self.timecount == self.ohlcv_df.shape[0]:
            terminated = True
            self.timecount -= 1

        return observation, self.reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=self._seed()[0])
        if self.verbose:
            values, counts = np.unique(self.actions_lst, return_counts=True)
            actions_counted: dict = dict(zip(values, counts))
            msg = (f"{self.__class__.__name__} #{self.idnum}: Episode length: {self.timecount}"
                   f"\treward {self.reward:.2f}\tAssets: {self.current_balance}\tPNL "
                   f"{self.pnl:.6f}\t{actions_counted}\treset")
            logger.info(msg)

        self.target_balance = float(self.initial_target_balance)
        self.coin_balance = float(self.initial_coin_balance)
        self.pnl: float = 0.
        self.timecount = 0
        self.reward = 0.
        self.actions_lst: list = []
        self.ohlcv_df, self.indicators_df = self.data_processor_obj.get_random_ohlcv_and_indicators(
            period_type=self.use_period)
        if self.ohlcv_df.shape[0] != self.indicators_df.shape[0]:
            msg = (f"{self.__class__.__name__} #{self.idnum}: ohlcv_df.shape = {self.ohlcv_df.shape}, "
                   f"indicators_df.shape = {self.indicators_df.shape}")
            logger.debug(msg)
            sys.exit('Error: Check data_processor, length of data is not equal!')
        self.initial_total_assets = self.initial_target_balance + (self.initial_coin_balance * self.price)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def close(self):
        if self.verbose:
            if self.verbose:
                msg = (f"Episode length: {self.timecount}\treward {self.reward:.2f}\tAssets: "
                       f"{self.current_balance}\tPNL {self.pnl:.6f}\tclose")
                logger.info(msg)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
