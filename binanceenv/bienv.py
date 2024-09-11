import sys
import math
import random
import logging
import gymnasium
import numpy as np
import numba
import pandas as pd
from numba import jit
from typing import Union, Optional

# from gymnasium.spaces import Discrete
# from gymnasium.spaces import Box
from gymnasium.utils import seeding
# from collections import OrderedDict

from binanceenv.cache import CacheManager
from binanceenv.cache import cache_manager_obj
from binanceenv.cache import eval_cache_manager_obj

from dbbinance.fetcher.datautils import get_timeframe_bins
from dbbinance.fetcher.datautils import get_nearest_timeframe

import multiprocessing as mp
from ctypes import c_bool
from dbbinance.fetcher.cachemanager import mlp_mutex

from collections import deque
from datawizard.dataprocessor import IndicatorProcessor
# from datawizard.dataprocessor import Constants
from binanceenv.spaces import *
from binanceenv.orderbook import TargetCash
from binanceenv.orderbook import Asset
import matplotlib.pyplot as plt

__version__ = 0.065

logger = logging.getLogger()


@jit(nopython=True)
def _new_logarithmic_scaler(value, value_sign):
    return ((1 / (1 + (math.e ** -(np.log10(value + 1)))) - 0.5) / 0.5) * value_sign


def new_logarithmic_scaler(value):
    return _new_logarithmic_scaler(abs(value), np.sign(value))


@jit(nopython=True)
def abs_logarithmic_scaler(value):
    _temp = (1 / (1 + (math.e ** -(np.log10(abs(value) + 1)))) - 0.5) / 0.5
    return _temp


_target_obj = TargetCash(symbol='USDT', initial_cash=100_000)

mp_timesteps_counter = mp.Value('i', 1)
mp_episodes_counter = mp.Value('i', -1)
mp_count = mp.Value('i', 0)


class BinanceEnvBase(gymnasium.Env):
    name = 'BinanceEnvBase'
    count = mp_count

    def __init__(self,
                 data_processor_kwargs: Union[dict, None],
                 target_balance: float = 100_000.,
                 target_minimum_trade: float = 100.,
                 target_maximum_trade: float = 1000.,
                 target_scale_decay: int = 1_000_000,
                 coin_balance: float = 0.,
                 pnl_stop: float = -0.5,
                 verbose: int = 0,
                 log_interval: int = 500,
                 observation_type: str = 'indicators',
                 action_type: str = 'discrete',
                 use_period: str = 'train',
                 stable_cache_data_n: int = 30,
                 reuse_data_prob: float = 0.99,
                 eval_reuse_prob: float = 0.99,
                 seed: int = 41,
                 lookback_window: Union[str, int, None] = None,
                 # max_lot_size: float = 0.25,
                 max_hold_timeframes='72h',
                 penalty_value: float = 1e-5,
                 invalid_actions: int = 60,
                 total_timesteps: int = 3_000_000,
                 eps_start: float = 0.99,
                 eps_end: float = 0.01,
                 eps_decay: float = 0.2,
                 gamma: float = 0.9999,
                 cache_obj: Union[CacheManager, None] = None,
                 render_mode=None,
                 index_type: str = 'target_time',
                 deterministic: bool = True,
                 multiprocessing: bool = False,
                 ):

        self.multiprocessing = multiprocessing

        BinanceEnvBase.count.value += 1

        self.idnum = int(BinanceEnvBase.count.value)
        self.observation_type = observation_type
        self.data_processor_obj = IndicatorProcessor(**data_processor_kwargs)

        self.index_type = index_type
        self.deterministic = deterministic

        if lookback_window is None:
            self.lookback_timeframes: int = 0
        elif isinstance(lookback_window, int):
            self.lookback_timeframes = lookback_window
        elif isinstance(lookback_window, str):
            self.lookback_timeframes = int(
                get_timeframe_bins(lookback_window) // get_timeframe_bins(self.data_processor_obj.timeframe))

        self.max_hold_timeframes = int(
            get_timeframe_bins(max_hold_timeframes) // get_timeframe_bins(self.data_processor_obj.timeframe))
        self.invalid_actions = invalid_actions
        self.penalty_value = penalty_value
        self.stable_cache_data_n = stable_cache_data_n
        self.reuse_data_prob = reuse_data_prob
        self.eval_reuse_prob = eval_reuse_prob
        # logger.info(f"{self.__class__.__name__} #{self.idnum}: MAX_TIMESTEPS = {self.max_timesteps}")

        self.use_period = use_period
        self.pnl_stop = pnl_stop
        self.log_interval = log_interval
        self.action_type = action_type
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_threshold = 1.0
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.gamma_return = 0.

        self.timecount: int = 0

        self.reward = 0.

        self.target = TargetCash(symbol='USDT',
                                 initial_cash=target_balance,
                                 minimum_trade=target_minimum_trade,
                                 maximum_trade=target_maximum_trade,
                                 scale_decay=target_scale_decay,
                                 use_period=self.use_period)

        self.asset = Asset(symbol='BTC',
                           commission=.001,  # 0.1000%
                           minimum_trade=0.00001,
                           target_obj=self.target,
                           initial_balance=(0., 0., 0.),
                           scale_decay=100)

        self.verbose = verbose
        timedelta = get_nearest_timeframe(
            self.lookback_timeframes * get_timeframe_bins(self.data_processor_obj.timeframe) + get_timeframe_bins('2d'))
        self.ohlcv_df, self.indicators_df = self.data_processor_obj.get_ohlcv_and_indicators_sample(timedelta=timedelta,
                                                                                                    index_type=self.index_type)

        self.__get_obs_func = None
        self._take_action_func = None

        self.CM = cache_obj
        if self.use_period == 'train':
            # using external cache_manager for multiprocessing or multithreading
            if cache_obj is None:
                self.CM = cache_manager_obj
            self._take_action_func = self._take_action_train
        else:
            # separated CacheManager for eval environment
            if cache_obj is None:
                self.CM = eval_cache_manager_obj
            self._take_action_func = self._take_action_test
            self.reuse_data_prob = eval_reuse_prob

        self.action_space_obj = None
        self.obs_lookback = deque(maxlen=self.lookback_timeframes)
        self._warmup_func = lambda *args: None
        self._last_lookback_timecount: int = 0
        self.observation_space = self.get_observation_space(observation_type=observation_type)
        self.action_space = self.get_action_space(action_type=action_type)  # {0, 1, 2}

        # self._action_buy_hold_sell = ["Buy", "Sell", "Hold"]

        self.initial_total_assets = self.initial_cash + (self.asset.balance.size * self.asset.balance.price)

        # self.action_bin_obj = ActionsBins([-1, 1], 3)
        self.actions_lst: list = []
        self.np_random = None
        self.seed = self.get_seed(seed + self.idnum)

        self.actions_lst: list = []
        self.min_coin_trade = self.asset.orders.minimum_trade + (
                self.asset.orders.commission * self.asset.orders.minimum_trade)
        self.total_timesteps = total_timesteps
        self.action_symbol = str()
        self.order_closed = False
        self.reward_step = .0

        self.previous_total_assets: float = 0.

        self.order_pnl: float = .0
        self.all_actions = np.asarray(list(range(3)))
        self.invalid_action_counter: int = 0
        self.episode_reward: float = 0.
        self.key_list: list = []
        self.first_epsilon = 0.
        self.epsilon: float = 1.0
        self.dones: bool = False

        self.render_path_filename = None
        self.render_mode = render_mode
        if self.render_mode is not None:
            self.render_df: pd.DataFrame = pd.DataFrame(index=self.ohlcv_df.index,
                                                        data=0.,
                                                        columns=['price', 'action', 'amount', 'pnl', 'total'])
            self.last_render_df: Union[pd.DataFrame, None] = None

        self.total_reward: float = 0.
        self.previous_pnl = float(self.pnl)

    @property
    def total_timesteps_counter(self):
        return mp_timesteps_counter.value

    @total_timesteps_counter.setter
    def total_timesteps_counter(self, value):
        mp_timesteps_counter.value = value

    @property
    def total_episodes_counter(self):
        return mp_episodes_counter.value

    @total_episodes_counter.setter
    def total_episodes_counter(self, value):
        mp_episodes_counter.value = value

    def __del__(self):
        BinanceEnvBase.count.value -= 1

    @property
    def cash(self):
        return self.target.cash

    @cash.setter
    def cash(self, value):
        self.target.cash = value

    @property
    def initial_cash(self):
        return self.target.initial_cash

    def get_observation_space(self, observation_type='indicators'):
        if observation_type == 'indicators_assets':
            space_obj = IndicatorsAndAssetsSpace(self.indicators_df.shape[1], 2)
            self.__get_obs_func = self._get_assets_indicators_obs
            # if self.asset.balance.size == .0:
            #     self.asset.balance = (1e-7, 56000.)
        elif observation_type == 'assets_close_indicators':
            space_obj = AssetsCloseIndicatorsSpace(self.indicators_df.shape[1], 5)
            self.__get_obs_func = self._get_assets_close_indicators_obs
        elif observation_type == 'lookback_assets_close_indicators':
            space_obj = LookbackAssetsCloseIndicatorsSpace(ind_num=self.indicators_df.shape[1],
                                                           assets_data=5,
                                                           lookback=self.lookback_timeframes)
            self.__get_obs_func = self._get_lookback_assets_close_indicators_obs
            # filling deque stack
            self._warmup_func = self._lookback_warmup
        elif observation_type == 'lookback_dict':
            space_obj = LookbackDictOHLCAssetsIndicatorsSpace(ind_num=self.indicators_df.shape[1],
                                                              assets_num=1,
                                                              lookback=self.lookback_timeframes)

            self.__get_obs_func = self._get_lookback_dict_obs
            # filling deque stack
            self._warmup_func = self._lookback_warmup

        elif observation_type == 'indicators_pnl':
            space_obj = IndicatorsAndPNLSpace(self.indicators_df.shape[1], 1)
            self.__get_obs_func = self._get_pnl_indicators_obs
        elif observation_type == 'indicators_close':
            space_obj = IndicatorsSpace(self.indicators_df.shape[1] + 1)
            self.__get_obs_func = self._get_indicators_close_obs
        elif observation_type == 'indicators':
            space_obj = IndicatorsSpace(self.indicators_df.shape[1])
            self.__get_obs_func = self._get_indicators_obs
        # elif observation_type == 'assets_close_indicators_action_masks':
        #     space_obj = AssetsCloseIndicatorsSpace(self.indicators_df.shape[1], 8)
        #     self.__get_obs_func = self._get_assets_close_indicators_action_masks_obs
        #     # if self.asset.balance.size == .0:
        #     #     self.asset.balance = (1e-7, 56000.)
        # elif observation_type == 'idx_assets_close_indicators_action_masks':
        #     space_obj = AssetsCloseIndicatorsSpace(self.indicators_df.shape[1], 2 + 1 + 4)
        #     self.__get_obs_func = self._get_idx_assets_close_indicators_action_masks_obs
        #     # if self.asset.balance.size == .0:
        #     #     self.asset.balance = (1e-7, 56000.)
        # elif observation_type == 'assets_close_action_masks_indicators':
        #     space_obj = AssetsCloseIndicatorsSpace(self.indicators_df.shape[1], 2 + 4)
        #     self.__get_obs_func = self._get_assets_close_action_masks_indicators_obs
        else:
            sys.exit(f'Error: Unknown observation type {observation_type}!')
        observation_space = space_obj.observation_space
        self.name = f'{self.name}_{space_obj.name}'
        return observation_space

    def get_action_space(self, action_type='discrete'):
        if action_type == 'discrete':
            self.action_space_obj = DiscreteActionSpace(3)
        elif action_type == 'box':
            self.action_space_obj = BoxActionSpace(n_action=3)
        elif action_type == 'box_4':
            self.action_space_obj = BoxActionSpace(n_action=4)
        elif action_type == 'box1_1_3':
            self.action_space_obj = BoxExtActionSpace(n_action=3)
        elif action_type == 'box1_1_4':
            self.action_space_obj = BoxExtActionSpace(n_action=4)
        elif action_type == 'binbox':
            self.action_space_obj = BinBoxActionSpace(n_action=3, low=-1, high=1)
        elif action_type == 'sell_buy_hold_amount':
            self.action_space_obj = SellBuyHoldAmount()
        else:
            sys.exit(f'Error: Unknown action type {action_type}!')
        action_space = self.action_space_obj.action_space
        self.name = f'{self.name}_{self.action_space_obj.name}'
        return action_space

    def recalc_epsilon(self):
        self.first_epsilon = (self.total_timesteps_counter / self.total_timesteps) * (
                self.eps_start - self.eps_end) + self.eps_end
        self.epsilon = 1 - self.first_epsilon
        # self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
        #     -1. * self.total_timesteps_counter / (self.total_timesteps * self.eps_decay))

    @property
    def price(self) -> float:
        return self.ohlcv_df.iloc[self.timecount]['close']

    @property
    def total_assets(self) -> float:
        return self.target.cash + (self.asset.balance.size * self.price)

    @property
    def pnl(self) -> float:
        return (self.total_assets / self.initial_total_assets) - 1.

    @property
    def current_balance(self) -> str:
        current_balance = (
            f"CURRENT Assets => {self.target.symbol}: {self.target.cash:.1f}, {self.asset.symbol}: "
            f"{self.asset.balance.size:.5f}\tTotal assets: {self.total_assets:.1f} {self.target.symbol}")
        return current_balance

    def _get_obs(self) -> np.ndarray:
        return self.__get_obs_func()

    def _get_assets_indicators_obs(self) -> np.ndarray:
        target = new_logarithmic_scaler(self.cash)
        # coin = logarithmic10_scaler(self.coin_balance)
        coin = new_logarithmic_scaler(self.asset.balance.size * self.price)
        # coin_orders_cost = logarithmic10_scaler(self.coin_orders_cost)
        return np.asarray(
            np.concatenate([[target, coin], self.indicators_df.iloc[self.timecount].values]),
            dtype=np.float32)

    def _get_assets_close_indicators_obs(self) -> np.ndarray:
        scaled_price = self.target.scaler(self.price)
        obs = np.concatenate([[self.target.scaled_cash],
                              self.asset.balance.scaled_arr,
                              [self.asset.balance.size * scaled_price, scaled_price]],
                             dtype=np.float32)
        return np.concatenate([obs, self.indicators_df.iloc[self.timecount].values]).astype(np.float32)

    def _get_lookback_assets_close_indicators_obs(self) -> np.ndarray:
        self.obs_lookback.append(self._get_assets_close_indicators_obs())
        return np.asarray(self.obs_lookback).astype(np.float32).flatten()

    def _get_dict_assets_close_indicators_obs(self) -> np.ndarray:
        scaled_ohlc = self.target.scaler(
            self.ohlcv_df.iloc[self.timecount][['open', 'high', 'low', 'close']].values).astype(np.float32)
        assets = np.concatenate(
            [[self.target.scaled_cash], self.asset.balance.scaled_arr, [self.asset.balance.size * scaled_ohlc[3]]],
            dtype=np.float32)
        indicators = self.indicators_df.iloc[self.timecount].values.astype(np.float32)
        return np.concatenate([assets, scaled_ohlc, indicators]).astype(np.float32)

    def _get_lookback_dict_obs(self) -> dict:
        self.obs_lookback.append(self._get_dict_assets_close_indicators_obs())
        _obs = np.asarray(self.obs_lookback).transpose()
        return dict({'assets': _obs[:5, -1].flatten(), 'ohlc': _obs[5:9, :], 'indicators': _obs[9:, :]})

    # def _get_assets_close_indicators_action_masks_obs(self) -> np.ndarray:
    #     # target = new_logarithmic_scaler(self.cash)
    #     # close = new_logarithmic_scaler(self.price)
    #     # asset_balance = new_logarithmic_scaler(self.asset.balance_arr)
    #     # asset_current_cost = new_logarithmic_scaler(self.asset.balance.size * self.price)
    #     action_masks = self._get_action_masks().astype(np.float32)
    #     obs = new_logarithmic_scaler(np.asarray(
    #         np.concatenate([[self.cash],
    #                         self.asset.balance.arr,
    #                         [self.asset.balance.size * self.price, self.price]]),
    #         dtype=np.float32))
    #     return np.concatenate([obs, self.indicators_df.iloc[self.timecount].values, action_masks]).astype(np.float32)

    # def _get_idx_assets_close_indicators_action_masks_obs(self) -> np.ndarray:
    #     idx = self.timecount * 1e-6
    #     target = new_logarithmic_scaler(self.cash)
    #     # coin = logarithmic10_scaler(self.coin_balance)
    #     close = new_logarithmic_scaler(self.price)
    #     # coin_orders_cost = logarithmic10_scaler(self.coin_orders_cost)
    #     coin = new_logarithmic_scaler(self.asset.balance.size)
    #     action_masks = self._get_action_masks().astype(np.float32)
    #     return np.asarray(
    #         np.concatenate([[idx, target, coin, close], self.indicators_df.iloc[self.timecount].values, action_masks]),
    #         dtype=np.float32)

    # def _get_assets_close_action_masks_indicators_obs(self) -> np.ndarray:
    #
    #     target = new_logarithmic_scaler(self.cash)
    #     # coin = logarithmic10_scaler(self.coin_balance)
    #     asset_balance = new_logarithmic_scaler(self.asset.balance.arr)
    #     asset_current_cost = new_logarithmic_scaler(self.asset.balance.size * self.price)
    #     # coin = new_logarithmic_scaler(self.asset.balance.size * self.price)
    #     close = new_logarithmic_scaler(self.price)
    #     # coin_orders_cost = logarithmic10_scaler(self.coin_orders_cost)
    #     action_masks = self._get_action_masks().astype(np.float32)
    #     return np.asarray(
    #         np.concatenate([[target],
    #                         asset_balance,
    #                         [asset_current_cost, close],
    #                         action_masks,
    #                         self.indicators_df.iloc[self.timecount].values]),
    #         dtype=np.float32)

    def _get_pnl_indicators_obs(self) -> np.ndarray:
        return np.asarray(np.concatenate([[self.pnl], self.indicators_df.iloc[self.timecount].values]),
                          dtype=np.float32)

    def _get_indicators_close_obs(self) -> np.ndarray:
        close = new_logarithmic_scaler(self.price)
        return np.asarray(np.concatenate([self.indicators_df.iloc[self.timecount].values, [close]]),
                          dtype=np.float32)

    def _get_indicators_obs(self) -> np.ndarray:
        return np.asarray(self.indicators_df.iloc[self.timecount].values, dtype=np.float32)

    def _get_action_masks(self) -> np.ndarray:
        return np.array(
            [self.cash / self.price >= self.min_coin_trade,
             self.asset.balance.size >= self.min_coin_trade,
             self.not_max_holding_time(self.max_hold_timeframes)])

    def _get_info(self) -> dict:
        return {"action_masks": self._get_action_masks(), "pnl": self.pnl}

    def not_max_holding_time(self, max_hold_timeframes) -> bool:
        check = True
        if len(self.actions_lst) > self.max_hold_timeframes:  # Hold
            # Penalty actions for just holding
            if np.all(np.array(self.actions_lst[-max_hold_timeframes:]) - 1):
                check = False
        return check

    def _take_action(self, action, amount) -> tuple:
        old_target_balance = float(self.cash)
        old_coin_balance = float(self.asset.balance.size)
        action_commission = .0
        # action_symbol = self.target.symbol
        order_cash = 0.
        msg = str()
        size = 0.

        """buy or sell stock"""
        if action == 0:  # Buy
            self.action_symbol = f'{self.target.symbol}->{self.asset.symbol}'
            max_amount = (self.cash / self.price) / (1. + self.asset.orders.commission)
            size = min(max(max(self.asset.minimum_trade, self.target.minimum_trade / self.price), amount * max_amount),
                       max_amount)
            self.asset.orders.buy(size, self.price)
            action_commission = self.asset.orders.book[-1].order_commission
            order_cash = self.asset.orders.book[-1].order_cash

        elif action == 1:  # Sell
            self.action_symbol = f'{self.asset.symbol}->{self.target.symbol}'
            size = min(max(self.min_coin_trade, amount * self.asset.balance.size), self.asset.balance.size)
            self.asset.orders.sell(size, self.price)
            action_commission = self.asset.orders.book[-1].order_commission
            order_cash = self.asset.orders.book[-1].order_cash

        if self.verbose == 2:
            if self.timecount % self.log_interval == 0:
                ohlcv = (
                    f"{self.timecount}\t {self.ohlcv_df.index[self.timecount]} \t"
                    f"open: \t{self.ohlcv_df.iloc[self.timecount]['open']:.2f} \t"
                    f"high: \t{self.ohlcv_df.iloc[self.timecount]['high']:.2f} \t"
                    f"low: \t{self.ohlcv_df.iloc[self.timecount]['low']:.2f} \t"
                    f"close: \t{self.ohlcv_df.iloc[self.timecount]['close']:.2f} \t"
                    f"volume: \t{self.ohlcv_df.iloc[self.timecount]['volume']:.2f}")

                old_balance = (
                    f"OLD Assets =>\t{self.target.symbol}: {old_target_balance:.4f}, "
                    f"{self.asset.symbol}: {old_coin_balance:.4f}"
                    f"\tTotal assets(old): "
                    f"{(old_target_balance + (old_coin_balance * self.price)):.1f} {self.target.symbol}")
                msg = (f"{ohlcv}\n{msg}"
                       f"\tAction num: {action}\t{old_balance} "
                       f"\tACTION => {actions_reversed_dict[action]}: size:{size:.4f}({order_cash:.2f}) "
                       f"{self.action_symbol}, commission: {action_commission:.2f}"
                       f"\t{self.current_balance}\tprofit {self.reward_step:.2f}\tPNL {self.pnl:.5f}")
                logger.info(msg)

        return action, amount

    def _take_action_train(self, action, amount):
        return self._take_action(action, amount)

    def _take_action_test(self, action, amount) -> tuple:
        return self._take_action(action, amount)

    def get_valid_actions(self, action_masks) -> list:
        return self.all_actions[action_masks]

    def step(self, action):
        # truncated = False
        # terminated = False
        # masked_action = 0
        action_penalty = False
        amount = 1.
        info = self._get_info()
        if self.use_period == 'train':
            self.reward_step = -self.penalty_value
        else:
            self.reward_step = .0

        if self.action_type == 'box':
            action = self.action_space_obj.convert2action(action, None)
        #     action = self.action_space_obj.convert2action(action, info['action_masks'])
        #     if self.eps_threshold < random.random():
        #         action = self.action_space_obj.convert2action(action, info['action_masks'])
        #     else:
        #         action = self.action_space_obj.convert2action(action, None)
        elif self.action_type == 'box1_1':
            masked_action, masked_amount = self.action_space_obj.convert2action(action, info['action_masks'])
            action, amount = self.action_space_obj.convert2action(action, None)
            if masked_action != action:
                amount = 0.
                self.invalid_action_counter += 1
                if self.use_period == 'train':
                    action_penalty = True
                    if action == 0:
                        self.reward_step += -5e-4
                    elif action == 1:
                        self.reward_step += -5e-4
                    else:
                        self.reward_step += -5e-6
            else:
                if self.use_period == 'train':
                    self.reward_step += 1e-5

        elif self.action_type in ['sell_buy_amount', 'binbox']:
            valid_actions = self.get_valid_actions(info['action_masks'])
            action, amount = self.action_space_obj.convert2action(action[0])
            if action not in valid_actions:
                amount = 0.
                self.invalid_action_counter += 1
                if self.use_period == 'train':
                    action_penalty = True
                    if action in [0, 1]:
                        self.reward_step += -5e-4
                    else:
                        self.reward_step += -5e-5
            else:
                if self.use_period == 'train':
                    self.reward_step += 1e-5

        elif self.action_type == 'discrete':
            valid_actions = self.get_valid_actions(info['action_masks'])
            if action not in valid_actions:
                amount = 0.
                self.invalid_action_counter += 1
                if self.use_period == 'train':
                    action_penalty = True
                    if action in [0, 1]:
                        self.reward_step += -5 * self.penalty_value * 10
                    else:
                        self.reward_step += -5 * self.penalty_value
            else:
                if self.use_period == 'train':
                    self.reward_step += self.penalty_value

        truncated = bool(self.invalid_action_counter >= self.invalid_actions)
        self.actions_lst.append(action)
        self._take_action_func(action, amount)

        # TODO  change amount to lot real size
        if self.render_mode is not None:
            self._render(self.timecount, self.price, action, amount, self.pnl, self.total_assets)
        # self.reward_step = self.calc_sharpe_ratio(risk_free_rate=0.02)

        observation = self._get_obs()

        self.total_reward = self.total_assets - self.initial_total_assets

        if self.use_period == 'train':
            if not action_penalty:
                self.reward_step += (self.pnl - self.previous_pnl)
            self.gamma_return = self.gamma_return * self.gamma + self.reward_step
        else:
            self.reward_step += (self.pnl - self.previous_pnl)

        self.previous_total_assets = float(self.total_assets)

        terminated = bool(self.pnl < self.pnl_stop)
        self.timecount += 1

        if self.timecount == self.ohlcv_df.shape[0]:
            terminated = True

        if terminated or truncated:
            self.timecount -= 1
            self.dones = True
            if self.use_period == 'train':
                self.reward_step += self.gamma_return

        self.episode_reward += self.reward_step
        self.previous_pnl = self.pnl

        return observation, self.reward_step, terminated, truncated, info

    def _render(self, timecount, price, action, amount, pnl, total):
        self.render_df.iloc[timecount] = price, action, amount, pnl, total

    def render(self):
        return self.render_df.iloc[self.timecount].values

    def log_reset_msg(self):
        values, counts = np.unique(self.actions_lst, return_counts=True)
        actions_counted: dict = dict(zip(values, counts))
        msg = (f"{self.__class__.__name__} #{self.idnum} {self.use_period}: "
               f"Ep.length(shape): {self.timecount}({self.ohlcv_df.shape[0]}), cache: {len(self.CM.cache):03d}/"
               f"inv_act#: {self.invalid_action_counter:03d}/Ep.reward: {self.episode_reward:.4f}"
               # f"\tepsilon: {self.epsilon:.6f}"
               f"\tprofit {self.total_reward:.1f}"
               # f"\teps_reward {self.eps_reward:.5f}"
               f"\tAssets: {self.current_balance}\tPNL "
               f"{self.pnl:.5f}\t{actions_counted}\treset")
        logger.info(msg)

    def _warmup(self):
        self._warmup_func()

    def _lookback_warmup(self):
        for ix in range(self.lookback_timeframes):
            _ = self.__get_obs_func()
            if self.render_mode is not None:
                self._render(self.timecount, self.price, actions_4_dict['Hold'], 0, self.pnl, self.total_assets)
            self.timecount += 1

    def calc_sharpe_ratio(self, risk_free_rate=0.02) -> float:
        # risk_free_rate = 0.02  # Example risk-free rate
        # std_deviation = self.render_df[:self.timecount + 1]['total'].std()
        # print(std_deviation)
        std_deviation = np.log1p(self.ohlcv_df.iloc[:self.timecount + 1]['close']).diff().std()
        # std_deviation = self.ohlcv_df.iloc[:self.timecount + 1]['close'].pct_change(fill_method='ffill').std()
        # return ((self.total_assets - self.previous_total_assets) * (
        #         1 - risk_free_rate)) / std_deviation if std_deviation != 0 else 0
        # returns_mean = self.render_df.iloc[:self.timecount + 1]['pnl'].mean()
        # sharpe_ratio = returns_mean / std_deviation if std_deviation != 0 else 0
        # returns = self.pnl - self.previous_pnl
        sharpe_ratio = (self.pnl - self.previous_pnl) / std_deviation if std_deviation != 0 else 0
        # print(self.timecount, f'{sharpe_ratio}', f'{returns_mean}', f'{std_deviation}')
        # if self.verbose > 1:
        #     logger.info(f'{self.timecount}, sr: {sharpe_ratio}, ret: {returns}, std: {std_deviation}')
        return sharpe_ratio

    def get_last_render_df(self):
        return self.last_render_df

    def reset(self, seed=None, options=None):
        if self.render_mode is not None:
            self.last_render_df = self.render_df.copy(deep=True)
        with mlp_mutex:
            self.total_timesteps_counter += (self.timecount - self.lookback_timeframes)
        if self.verbose:
            self.log_reset_msg()
        # self.target_balance = float(self.initial_target_balance)
        # self.coin_balance = float(self.initial_coin_balance)
        self.timecount: int = 0
        self.reward = 0.
        self.reward_step = 0.
        self.actions_lst: list = []
        self.invalid_action_counter = 0
        self.gamma_return = 0.
        self.recalc_epsilon()  # recalculate epsilon
        self.dones = False
        stable_cache = max(25.,
                           self.stable_cache_data_n * self.first_epsilon) if self.use_period == 'train' else self.stable_cache_data_n

        if self.reuse_data_prob > self.np_random.random() and len(self.CM.cache) >= stable_cache:
            if not self.key_list:
                self.key_list = list(self.CM.cache.keys())
                self.np_random.shuffle(self.key_list)
            self.ohlcv_df, self.indicators_df = self.CM.cache[self.key_list[0]]
            rnd_start = self.np_random.integers(
                max(1, int(200 * self.first_epsilon))) if self.use_period == 'train' else 0
            self.ohlcv_df = self.ohlcv_df.iloc[rnd_start:].copy(deep=True)
            self.indicators_df = self.indicators_df.iloc[rnd_start:].copy(deep=True)
            self.key_list = self.key_list[1:]
        else:
            self.ohlcv_df, self.indicators_df = self.data_processor_obj.get_random_ohlcv_and_indicators(
                index_type=self.index_type, period_type=self.use_period)

            if self.ohlcv_df.shape[0] != self.indicators_df.shape[0]:
                msg = (f"{self.__class__.__name__} #{self.idnum}: ohlcv_df.shape = {self.ohlcv_df.shape}, "
                       f"indicators_df.shape = {self.indicators_df.shape}")
                logger.debug(msg)
                sys.exit('Error: Check data_processor, length of data is not equal!')

            cm_key = tuple((self.ohlcv_df.index[0], self.ohlcv_df.index[-1]))
            self.CM.update_cache(key=cm_key, value=(self.ohlcv_df, self.indicators_df))

        if self.use_period == 'train':
            size = self.np_random.random() / 2
            cost = self.price * size
            price = self.price
        else:
            size, cost, price = .0, .0, .0

        self.asset.reset((size, cost, price))

        self.initial_total_assets = self.initial_cash + (
                self.asset.initial_balance.size * self.asset.initial_balance.price)

        if self.render_mode is not None:
            self.render_df = pd.DataFrame(index=self.ohlcv_df.index, data=0.,
                                          columns=['price', 'action', 'amount', 'pnl', 'total'])

        self.order_closed = False
        self.episode_reward = .0
        self.gamma_return = .0
        if 'lookback' in self.observation_type:
            self._warmup()
        observation = self._get_obs()
        info = self._get_info()
        self.previous_pnl = float(self.pnl)
        self.previous_total_assets: float = self.total_assets

        return observation, info

    def close(self):
        if self.verbose:
            msg = (f"Ep.length: {self.timecount}\treward {self.reward:.2f}\tAssets: "
                   f"{self.current_balance}\tPNL {self.pnl:.6f}\tclose")
            logger.info(msg)

    def get_seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    def set_render_output(self, path_filename: str):
        self.render_path_filename = path_filename

    def render_all(self, df: pd.DataFrame, show_amount=True):
        fig, ax1 = plt.subplots(figsize=(30, 17))
        scale_factor = 20
        ax2 = ax1.twinx()

        ax1.set_axisbelow(True)
        ax1.minorticks_on()
        # Turn on the minor TICKS, which are required for the minor GRID
        # Customize the major grid
        ax1.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        # Customize the minor grid
        ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        # ax1.legend(fontsize='large', loc='upper left')

        # df.plot(y="total", ax=ax, use_index=True, style='--', color='lightgrey')
        # df.plot(y="price", ax=ax, use_index=True, secondary_y=True, color='black')

        line1, = ax1.plot(df.index, df['price'], c='cyan', label='Price ("Close")')
        line2, = ax2.plot(df.index, df['total'], c='grey', label='Total assets', alpha=0.5)
        # labels_handles = [line1, line2,]
        buy_kwargs = dict(alpha=0.6, label='Buy')
        sell_kwargs = dict(alpha=0.6, label='Sell')
        close_kwargs = dict(alpha=0.6, label='Close(all)')
        for idx in df.index.tolist():
            if (df.loc[idx]['action'] == actions_4_dict['Buy']) & (df.loc[idx]['amount'] > 0):
                marker_buy, = ax1.plot(idx, df.loc[idx]['price'] - 5 * scale_factor, 'g^', **buy_kwargs)
                if buy_kwargs.get('label') is not None:
                    del buy_kwargs['label']
                if show_amount:
                    ax1.text(idx, df.loc[idx]['price'] - 10 * scale_factor, f"{df.loc[idx]['amount']:.3f}", c='green',
                             fontsize=7, horizontalalignment='center', verticalalignment='center')

            elif (df.loc[idx]['action'] == actions_4_dict['Sell']) & (df.loc[idx]['amount'] > 0):
                marker_sell, = ax1.plot(idx, df.loc[idx]['price'] + 5 * scale_factor, 'rv', **sell_kwargs)
                if sell_kwargs.get('label') is not None:
                    del sell_kwargs['label']
                if show_amount:
                    ax1.text(idx, df.loc[idx]['price'] + 10 * scale_factor, f"{df.loc[idx]['amount']:.3f}", c='red',
                             fontsize=7, horizontalalignment='center', verticalalignment='center')
            elif (df.loc[idx]['action'] == actions_4_dict['Close']) & (df.loc[idx]['amount'] > 0):
                marker_close, = ax1.plot(idx, df.loc[idx]['price'] + 5 * scale_factor, 'mv', **close_kwargs)
                if close_kwargs.get('label') is not None:
                    del close_kwargs['label']
                if show_amount:
                    ax1.text(idx, df.loc[idx]['price'] + 10 * scale_factor, f"{df.loc[idx]['amount']:.3f}", c='magenta',
                             fontsize=7, horizontalalignment='center', verticalalignment='center')

        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles=handles, labels=labels, loc='best', fontsize='large')

        if self.render_path_filename is not None:
            plt.savefig(f"{self.render_path_filename}.png",
                        dpi=450,
                        facecolor='w',
                        edgecolor='w',
                        orientation='portrait',
                        format=None,
                        transparent=False,
                        bbox_inches=None,
                        pad_inches=0.1,
                        metadata=None
                        )
            plt.close()
        else:
            plt.show()
            # plt.close()


class BinanceEnvCash(BinanceEnvBase):
    # count: int = 0
    name = 'BinanceEnvCash'

    def __init__(self,
                 data_processor_kwargs: Union[dict, None],
                 target_balance: float = 100_000.,
                 target_minimum_trade: float = 5.,
                 target_maximum_trade: float = 100.,
                 target_scale_decay: int = 1_000_000,
                 coin_balance: float = 0.,
                 pnl_stop: float = -0.5,
                 verbose: int = 0,
                 log_interval=500,
                 observation_type='indicators',
                 action_type='discrete',
                 use_period='train',
                 stable_cache_data_n=30,
                 reuse_data_prob=0.5,
                 eval_reuse_prob=0.1,
                 seed=41,
                 lookback_window: Union[str, int, None] = None,
                 max_hold_timeframes='72h',
                 penalty_value=10,
                 invalid_actions=60,
                 total_timesteps: int = 3_000_000,
                 eps_start=0.95,
                 eps_end=0.01,
                 eps_decay=0.2,
                 gamma=0.99,
                 cache_obj: Union[CacheManager, None] = None,
                 render_mode: Union[str, None] = None,
                 index_type: str = 'target_time',
                 deterministic: bool = True,
                 multiprocessing: bool = False, ):
        super().__init__(data_processor_kwargs, target_balance, target_minimum_trade, target_maximum_trade,
                         target_scale_decay, coin_balance, pnl_stop, verbose, log_interval, observation_type,
                         action_type, use_period, stable_cache_data_n, reuse_data_prob,
                         eval_reuse_prob, seed, lookback_window, max_hold_timeframes, penalty_value,
                         invalid_actions, total_timesteps, eps_start, eps_end, eps_decay, gamma, cache_obj, render_mode,
                         index_type, deterministic, multiprocessing)

        # BinanceEnvCash.count += 1

        self.idnum = int(BinanceEnvCash.count.value)

        self.buy_and_hold_start_size = self.asset.balance.size + (self.initial_cash / self.price) * (
                1 - self.asset.orders.commission)
        self.previous_price = self.price
        self.previous_buy_and_hold_pnl: float = 0.
        self.last_sell_order_pnl: float = 0.
        self.previous_balance = str()
        self.size_lst: list = []

    @property
    def buy_and_hold_pnl(self) -> float:
        return (self.buy_and_hold_start_size * self.price) / self.initial_total_assets - 1

    def _take_action(self, action, amount) -> tuple:
        old_target_balance = float(self.cash)
        old_coin_balance = float(self.asset.balance.size)
        action_commission = .0
        order_cash: float = 0.
        # action_symbol = self.target.symbol
        msg = str()
        size = 0.
        order_profit = 0.

        """buy or sell stock"""
        if action == 0:  # Buy
            self.action_symbol = f'{self.target.symbol}->{self.asset.symbol}'
            max_size = (self.cash / self.price) / (1. + self.asset.orders.commission)
            min_trade = max(self.asset.minimum_trade, self.target.minimum_trade / self.price)
            max_trade = min(max_size if max_size > min_trade else 0., self.target.maximum_trade / self.price)
            # max_trade = max_size if max_size > min_trade else 0.
            size = min(max(min_trade, amount), max_trade)
            if size != 0.:
                self.asset.orders.buy(size, self.price)
                action_commission = self.asset.orders.book[-1].order_commission
                order_cash = self.asset.orders.book[-1].order_cash
                # self.reward_step = self.penalty_value
                # self.reward_step = (self.previous_buy_and_hold_pnl - self.buy_and_hold_pnl) * self.first_epsilon ** 2
                """ looking for negative situation, we need to have less self.pnl then previous or same"""
                # self.reward_step = (self.previous_pnl - self.pnl)/self.pnl
                # self.reward_step = -((self.pnl - self.previous_pnl) + (action_commission / self.initial_total_assets))
                self.reward_step = self.previous_pnl - self.pnl
            # else:
            #     action = 2

        elif action == 1:  # Sell
            self.action_symbol = f'{self.asset.symbol}->{self.target.symbol}'
            min_trade = max(self.min_coin_trade,
                            (self.target.minimum_trade / self.price) * (1. + self.asset.orders.commission))
            max_trade = min(self.asset.balance.size if self.asset.balance.size > min_trade else 0.,
                            self.target.maximum_trade / self.price)
            # max_trade = self.asset.balance.size if self.asset.balance.size > min_trade else 0.
            size = min(max(min_trade, amount), max_trade)

            if size != 0:
                self.asset.orders.sell(size, self.price)
                action_commission = self.asset.orders.book[-1].order_commission
                order_cash = self.asset.orders.book[-1].order_cash
                order_profit = order_cash - self.asset.orders.book[-1].size * self.asset.balance.price

                """ Sell action reward """
                self.reward_step = order_profit / self.initial_total_assets
                # self.reward_step = (self.pnl - self.previous_pnl) + (action_commission / self.initial_total_assets)

                # self.reward_step += (order_profit / self.initial_total_assets)
                # self.last_sell_order_pnl = float(self.pnl)
            # else:
            #     action = 2

        # elif action == 3:  # Close all
        #     self.action_symbol = f'{self.asset.symbol}->{self.target.symbol}'
        #     size = self.asset.balance.size if self.asset.balance.size > self.min_coin_trade else 0.
        #     if size != 0:
        #         self.asset.orders.sell(size, self.price)
        #         action_commission = self.asset.orders.book[-1].order_commission
        #         order_cash = self.asset.orders.book[-1].order_cash
        #         order_profit = order_cash - self.asset.orders.book[-1].size * self.asset.balance.price
        #
        #         """ Close action reward """
        #         self.reward_step += (order_profit / self.initial_total_assets)
        #         # self.last_sell_order_pnl = float(self.pnl)
        #     # else:
        #     #     action = 2

        if self.verbose == 2:
            if self.timecount % self.log_interval == 0:
                ohlcv = (
                    f"{self.timecount}\t {self.ohlcv_df.index[self.timecount]} \t"
                    f"open: \t{self.ohlcv_df.iloc[self.timecount]['open']:.2f} \t"
                    f"high: \t{self.ohlcv_df.iloc[self.timecount]['high']:.2f} \t"
                    f"low: \t{self.ohlcv_df.iloc[self.timecount]['low']:.2f} \t"
                    f"close: \t{self.ohlcv_df.iloc[self.timecount]['close']:.2f} \t"
                    f"volume: \t{self.ohlcv_df.iloc[self.timecount]['volume']:.2f}")

                old_balance = (
                    f"OLD Assets =>\t{self.target.symbol}: {old_target_balance:.4f}, "
                    f"{self.asset.symbol}: {old_coin_balance:.4f}"
                    f"\tTotal assets(old): "
                    f"{(old_target_balance + (old_coin_balance * self.price)):.1f} {self.asset.symbol}")
                msg = (f"{ohlcv}\n{msg}"
                       f"\tAction num: {action}\t{old_balance} "
                       f"\tACTION => {actions_4_reversed_dict[action]}: size:{size:.4f}({order_cash:.2f})(a:{amount:.2f} "
                       f"{self.action_symbol}, commission: {action_commission:.2f}"
                       f"\t{self.current_balance}\tprofit {order_profit:.4f}\tPNL {self.pnl:.4f}"
                       f"\treward:{self.reward_step:.5f}")
                logger.info(msg)

        return action, size

    def step(self, action):
        info = self._get_info()
        self.reward_step = .0
        # self.reward_step = (self.pnl - self.previous_pnl) - (
        #         (self.buy_and_hold_pnl - self.previous_buy_and_hold_pnl) * (1 + np.clip(self.first_epsilon,
        #                                                                                 a_min=0.,
        #                                                                                 a_max=0.5)))
        # """ base step reward """
        # self.reward_step = (self.pnl - self.previous_pnl) - (self.buy_and_hold_pnl - self.previous_buy_and_hold_pnl) * (
        #         1 + self.first_epsilon ** 2)

        # excess_return = self.pnl - self.previous_pnl
        # std_deviation = np.log(self.ohlcv_df.iloc[:self.timecount + 1]['close']).diff().std()
        # sharpe_ratio = excess_return / std_deviation if std_deviation != 0 else 0
        # print(self.timecount, f'{excess_return}', std_deviation, sharpe_ratio,  self.calc_sharpe_ratio(risk_free_rate=0.02))
        # self.reward_step = (self.pnl - self.previous_pnl)
        # self.reward_step = -self.penalty_value

        action, amount = self.action_space_obj.convert2action(action, None)
        # action, amount = self.action_space_obj.convert2action(action, info['action_masks'])

        real_action, real_size = self._take_action_func(action, amount)

        if self.render_mode is not None:
            self._render(self.timecount, self.price, real_action, real_size, self.pnl, self.total_assets)

        # if self.timecount - self.lookback_timeframes > self.ohlcv_df.shape[0] // 2:
        #     self.reward_step += self.calc_sharpe_ratio()

        self.actions_lst.append(real_action)
        self.size_lst.append(real_size)

        observation = self._get_obs()

        truncated = bool(self.invalid_action_counter >= self.invalid_actions)
        terminated = bool(self.pnl < self.pnl_stop)

        """ don't move. use it before increasing timecount """
        """ ----------------------------------------------- """
        self.total_reward = self.total_assets - self.initial_total_assets
        self.previous_pnl = float(self.pnl)
        self.previous_buy_and_hold_pnl = float(self.buy_and_hold_pnl)
        self.previous_total_assets = float(self.total_assets)
        self.previous_price = self.price
        self.previous_balance = str(self.current_balance)
        """ ----------------------------------------------- """
        self.timecount += 1
        """ ----------------------------------------------- """

        if self.timecount == self.ohlcv_df.shape[0]:
            terminated = True

        self.reward_step = self.reward_step * 10

        if terminated or truncated:
            self.dones = True
            """ using the current pnl (as previous_pnl) """
            if self.previous_pnl == 0.:
                last_reward = (self.pnl_stop - self.previous_buy_and_hold_pnl) * self.first_epsilon ** 2
            elif self.previous_pnl > 0.:
                last_reward = 0.5 + self.previous_pnl
                # if self.previous_pnl - self.previous_buy_and_hold_pnl < 0.:
                #     last_reward = self.previous_pnl - self.previous_buy_and_hold_pnl * (1 + self.first_epsilon)
                # else:
                #     last_reward = self.previous_pnl
            else:  # self.pnl < 0.
                last_reward = -0.5 - abs(self.previous_pnl)
            self.reward_step += (last_reward * 10)

        self.episode_reward += self.reward_step

        return observation, self.reward_step, terminated, truncated, info

    def log_reset_msg(self):
        values, counts = np.unique(self.actions_lst, return_counts=True)
        actions_counted: dict = dict(zip(values, counts))
        msg = (f"{self.__class__.__name__} #{self.idnum} {self.use_period}: "
               f"Ep.len: {self.timecount}({self.ohlcv_df.shape[0]}), cache: {len(self.CM.cache):03d}/"
               f"In.t.asset: {self.initial_total_assets:.0f} "
               # f"inv_act#: {self.invalid_action_counter:03d}/"
               f"Ep.reward: {self.episode_reward:.4f}"
               f"\tepsilon: {self.epsilon:.5f}"
               f"\tprofit {self.total_reward:.1f}"
               # f"\steps_reward {self.eps_reward:.5f}"
               f"\tAssets: {self.previous_balance}\tPNL {self.previous_pnl:.5f} BH_PNL {self.previous_buy_and_hold_pnl:.5f}"
               f"\t{actions_counted} \t#{self.total_episodes_counter:05d}")
        logger.info(msg)

    def reset(self, **kwargs):
        self.total_episodes_counter += 1 if self.use_period == 'train' else 0
        observation, info = super().reset(**kwargs)
        self.size_lst = []
        self.last_sell_order_pnl = 0.
        self.buy_and_hold_start_size = self.asset.balance.size + (self.initial_cash / self.price) * (
                1 - self.asset.orders.commission)
        self.previous_buy_and_hold_pnl = 0.
        self.previous_balance = str()
        return observation, info

    def __del__(self):
        BinanceEnvCash.count.value -= 1
