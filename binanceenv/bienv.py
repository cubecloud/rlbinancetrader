import sys
import math
import random
import logging
import gymnasium
import numpy as np
import numba
from numba import jit
from typing import Union

# from gymnasium.spaces import Discrete
# from gymnasium.spaces import Box
from gymnasium.utils import seeding
# from collections import OrderedDict

from binanceenv.cache import CacheManager
from binanceenv.cache import cache_manager_obj
from binanceenv.cache import eval_cache_manager_obj

from dbbinance.fetcher.datautils import get_timeframe_bins
from dbbinance.fetcher.datautils import get_nearest_timeframe
from dbbinance.fetcher.cachemanager import mlp_mutex

from collections import deque
from datawizard.dataprocessor import IndicatorProcessor
# from datawizard.dataprocessor import Constants
from binanceenv.spaces import *
from binanceenv.orderbook import TargetCash
from binanceenv.orderbook import Asset

__version__ = 0.037

logger = logging.getLogger()


# @jit(nopython=True)
# def new_logarithmic_scaler(value):
#     if value < 0:
#         _temp = -(1 / (1 + (math.e ** -(np.log10(abs(value) + 1)))) - 0.5) / 0.5
#     else:
#         _temp = (1 / (1 + (math.e ** -(np.log10(value + 1)))) - 0.5) / 0.5
#     return _temp

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


class BinanceEnvBase(gymnasium.Env):
    count = 0
    name = 'BinanceEnvBase'
    total_timesteps_counter: int = 1
    total_episodes_counter: int = -1

    def __init__(self,
                 data_processor_kwargs: Union[dict, None],
                 target_balance: float = 100_000.,
                 target_minimum_trade: float = 5.,
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
                 # TODO inplement max lot size
                 max_lot_size: float = 0.25,
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
                 deterministic: bool = True):

        BinanceEnvBase.count += 1

        self.idnum = int(BinanceEnvBase.count)
        self.observation_type = observation_type
        self.data_processor_obj = IndicatorProcessor(**data_processor_kwargs)
        self.max_lot_size = max_lot_size
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
        self.target = TargetCash(symbol='USDT', initial_cash=target_balance, minimum_trade=target_minimum_trade)

        self.target = TargetCash(symbol='USDT',
                                 initial_cash=target_balance,
                                 minimum_trade=target_minimum_trade,
                                 use_period=self.use_period)

        self.asset = Asset(symbol='BTC',
                           commission=.002,
                           minimum_trade=0.00001,
                           target_obj=self.target,
                           initial_balance=(0., 0., 0.))

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
        self.old_total_assets: float = 0.
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
        self.total_reward: float = 0.
        self.previous_pnl: float = .0
        self.order_pnl: float = .0
        self.all_actions = np.asarray(list(range(3)))
        self.invalid_action_counter: int = 0
        self.episode_reward: float = 0.
        self.key_list: list = []
        self.epsilon: float = 1.0
        self.dones: bool = False

    def __del__(self):
        BinanceEnvBase.count -= 1

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
            self._warmup_func = self._lookback_warmup
            # filling deque stack
        elif observation_type == 'assets_close_indicators_action_masks':
            space_obj = AssetsCloseIndicatorsSpace(self.indicators_df.shape[1], 8)
            self.__get_obs_func = self._get_assets_close_indicators_action_masks_obs
            # if self.asset.balance.size == .0:
            #     self.asset.balance = (1e-7, 56000.)
        elif observation_type == 'idx_assets_close_indicators_action_masks':
            space_obj = AssetsCloseIndicatorsSpace(self.indicators_df.shape[1], 2 + 1 + 4)
            self.__get_obs_func = self._get_idx_assets_close_indicators_action_masks_obs
            # if self.asset.balance.size == .0:
            #     self.asset.balance = (1e-7, 56000.)
        elif observation_type == 'assets_close_action_masks_indicators':
            space_obj = AssetsCloseIndicatorsSpace(self.indicators_df.shape[1], 2 + 4)
            self.__get_obs_func = self._get_assets_close_action_masks_indicators_obs
        elif observation_type == 'indicators_pnl':
            space_obj = IndicatorsAndPNLSpace(self.indicators_df.shape[1], 1)
            self.__get_obs_func = self._get_pnl_indicators_obs
        elif observation_type == 'indicators_close':
            space_obj = IndicatorsSpace(self.indicators_df.shape[1] + 1)
            self.__get_obs_func = self._get_indicators_close_obs
        elif observation_type == 'indicators':
            space_obj = IndicatorsSpace(self.indicators_df.shape[1])
            self.__get_obs_func = self._get_indicators_obs
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
        elif action_type == 'box1_1_3':
            self.action_space_obj = BoxExtActionSpace(n_action=3)
        elif action_type == 'binbox':
            self.action_space_obj = BinBoxActionSpace(n_action=3, low=-1, high=1)
        elif action_type == 'two_actions':
            self.action_space_obj = TwoActionsSpace(low=-1, high=1)
        # elif action_type == 'sell_buy_hold_amount':
        #     self.action_space_obj = SellBuyHoldAmount(actions=3)
        else:
            sys.exit(f'Error: Unknown action type {action_type}!')
        action_space = self.action_space_obj.action_space
        self.name = f'{self.name}_{self.action_space_obj.name}'
        return action_space

    def recalc_epsilon(self):
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1. * self.total_timesteps_counter / (self.total_timesteps * self.eps_decay))

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
        obs = new_logarithmic_scaler(np.asarray(
            np.concatenate([[self.cash],
                            self.asset.balance.arr,
                            [self.asset.balance.size * self.price, self.price]]),
            dtype=np.float32))
        return np.concatenate([obs, self.indicators_df.iloc[self.timecount].values]).astype(np.float32)

    def _get_lookback_assets_close_indicators_obs(self) -> np.ndarray:
        self.obs_lookback.append(self._get_assets_close_indicators_obs())
        # self._last_lookback_timecount = self.timecount
        return np.asarray(self.obs_lookback).astype(np.float32).flatten()

    def _get_assets_close_indicators_action_masks_obs(self) -> np.ndarray:
        # target = new_logarithmic_scaler(self.cash)
        # close = new_logarithmic_scaler(self.price)
        # asset_balance = new_logarithmic_scaler(self.asset.balance_arr)
        # asset_current_cost = new_logarithmic_scaler(self.asset.balance.size * self.price)
        action_masks = self._get_action_masks().astype(np.float32)
        obs = new_logarithmic_scaler(np.asarray(
            np.concatenate([[self.cash],
                            self.asset.balance.arr,
                            [self.asset.balance.size * self.price, self.price]]),
            dtype=np.float32))
        return np.concatenate([obs, self.indicators_df.iloc[self.timecount].values, action_masks]).astype(np.float32)

    def _get_idx_assets_close_indicators_action_masks_obs(self) -> np.ndarray:
        idx = self.timecount * 1e-6
        target = new_logarithmic_scaler(self.cash)
        # coin = logarithmic10_scaler(self.coin_balance)
        close = new_logarithmic_scaler(self.price)
        # coin_orders_cost = logarithmic10_scaler(self.coin_orders_cost)
        coin = new_logarithmic_scaler(self.asset.balance.size)
        action_masks = self._get_action_masks().astype(np.float32)
        return np.asarray(
            np.concatenate([[idx, target, coin, close], self.indicators_df.iloc[self.timecount].values, action_masks]),
            dtype=np.float32)

    def _get_assets_close_action_masks_indicators_obs(self) -> np.ndarray:

        target = new_logarithmic_scaler(self.cash)
        # coin = logarithmic10_scaler(self.coin_balance)
        asset_balance = new_logarithmic_scaler(self.asset.balance.arr)
        asset_current_cost = new_logarithmic_scaler(self.asset.balance.size * self.price)
        # coin = new_logarithmic_scaler(self.asset.balance.size * self.price)
        close = new_logarithmic_scaler(self.price)
        # coin_orders_cost = logarithmic10_scaler(self.coin_orders_cost)
        action_masks = self._get_action_masks().astype(np.float32)
        return np.asarray(
            np.concatenate([[target],
                            asset_balance,
                            [asset_current_cost, close],
                            action_masks,
                            self.indicators_df.iloc[self.timecount].values]),
            dtype=np.float32)

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

        observation = self._get_obs()

        self.total_reward = self.total_assets - self.initial_total_assets

        if self.use_period == 'train':
            if not action_penalty:
                self.reward_step += (self.pnl - self.previous_pnl)
            self.gamma_return = self.gamma_return * self.gamma + self.reward_step
        else:
            self.reward_step += (self.pnl - self.previous_pnl)

        self.old_total_assets = float(self.total_assets)
        self.previous_pnl = float(self.pnl)

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
            self.timecount = ix

    def reset(self, seed=None, options=None):
        with mlp_mutex:
            self.total_timesteps_counter += self.timecount
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
        stable_cache = max(5., self.stable_cache_data_n * (
                    1. - self.epsilon)) if self.use_period == 'train' else self.stable_cache_data_n

        if self.reuse_data_prob > self.np_random.random() and len(self.CM.cache) >= stable_cache:
            if not self.key_list:
                self.key_list = list(self.CM.cache.keys())
                self.np_random.shuffle(self.key_list)
            self.ohlcv_df, self.indicators_df = self.CM.cache[self.key_list[0]]
            self.key_list = self.key_list[1:]
        else:
            # self.data_processor_obj.change_train_timeframes_num(self.data_processor_obj.initial_minimum_train_size, self.data_processor_obj.initial_maximum_train_size)
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
            size = self.np_random.random() / 2 * (1 - self.epsilon)
            cost = self.price * size
            price = self.price
        else:
            size, cost, price = .0, .0, .0

        self.asset.reset((size, cost, price))

        self.initial_total_assets = self.initial_cash + (
                self.asset.initial_balance.size * self.asset.initial_balance.price)

        self.order_closed = False
        self.episode_reward: float = .0
        self.gamma_return = .0
        if self.observation_type in ['lookback_assets_close_indicators']:
            self._warmup()
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def close(self):
        if self.verbose:
            if self.verbose:
                msg = (f"Ep.length: {self.timecount}\treward {self.reward:.2f}\tAssets: "
                       f"{self.current_balance}\tPNL {self.pnl:.6f}\tclose")
                logger.info(msg)

    def get_seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed


class BinanceEnvCash(BinanceEnvBase):
    count: int = 0
    name = 'BinanceEnvCash'

    def __init__(self,
                 data_processor_kwargs: Union[dict, None],
                 target_balance: float = 100_000.,
                 target_minimum_trade: float = 5.,
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
                 max_lot_size=0.25,
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
                 deterministic: bool = True):
        super().__init__(data_processor_kwargs, target_balance, target_minimum_trade, coin_balance, pnl_stop, verbose,
                         log_interval, observation_type, action_type, use_period, stable_cache_data_n, reuse_data_prob,
                         eval_reuse_prob, seed, lookback_window, max_lot_size, max_hold_timeframes, penalty_value,
                         invalid_actions, total_timesteps, eps_start, eps_end, eps_decay, gamma, cache_obj, render_mode,
                         index_type, deterministic)

        BinanceEnvCash.count += 1

        self.idnum = int(BinanceEnvCash.count)

        self.order_opened: bool = False
        self.order_closed: bool = False
        self.order_opened_pnl: float = 0.
        self.order_closed_pnl: float = 0.
        self.buy_and_hold_start_size = self.asset.balance.size + (self.initial_cash / self.price) * (
                1 - self.asset.orders.commission)
        self.previous_price = self.price
        self.size_lst: list = []
        self.last_sell_order_pnl: float = 0.

    @property
    def buy_and_hold_pnl(self):
        return (self.buy_and_hold_start_size * self.price) / self.initial_total_assets - 1

    def _take_action(self, action, amount) -> tuple:
        old_target_balance = float(self.cash)
        old_coin_balance = float(self.asset.balance.size)
        action_commission = .0
        order_cash: float = 0.
        # action_symbol = self.target.symbol
        msg = str()
        size = 0.

        """buy or sell stock"""
        if action == 0:  # Buy
            self.action_symbol = f'{self.target.symbol}->{self.asset.symbol}'
            max_size = (self.cash / self.price) / (1. + self.asset.orders.commission)
            min_trade = max(self.asset.minimum_trade, self.target.minimum_trade / self.price)
            # size = min(max(min_trade, amount), max_size if min_trade < max_size else 0.)
            size = min_trade if min_trade < max_size else 0.
            if size != 0.:
                self.asset.orders.buy(size, self.price)
                action_commission = self.asset.orders.book[-1].order_commission
                order_cash = self.asset.orders.book[-1].order_cash

        elif action == 1:  # Sell
            self.action_symbol = f'{self.asset.symbol}->{self.target.symbol}'
            min_trade = max(self.min_coin_trade, self.target.minimum_trade / self.price)
            # size = min(max(min_trade, amount),
            #            self.asset.balance.size if min_trade < self.asset.balance.size else 0.)
            size = min_trade if min_trade < self.asset.balance.size else 0.

            if size != 0:
                self.asset.orders.sell(size, self.price)
                action_commission = self.asset.orders.book[-1].order_commission
                order_cash = self.asset.orders.book[-1].order_cash
                self.reward_step = order_cash - self.asset.orders.book[-1].size * self.asset.balance.price
                self.reward_step = (self.reward_step / self.initial_total_assets)
                # self.last_sell_order_pnl = float(self.pnl)

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
                       f"\tACTION => {actions_reversed_dict[action]}: size:{size:.4f}({order_cash:.2f}) "
                       f"{self.action_symbol}, commission: {action_commission:.2f}"
                       f"\t{self.current_balance}\tprofit {self.reward_step:.2f}\tPNL {self.pnl:.4f}")
                logger.info(msg)

        return action, size

    def step(self, action):
        info = self._get_info()
        # self.reward_step = .0
        self.reward_step = -self.penalty_value

        action, amount = self.action_space_obj.convert2action(action, None)
        # action, amount = self.action_space_obj.convert2action(action, info['action_masks'])
        # valid_actions = self.get_valid_actions(info['action_masks'])
        # if action not in valid_actions:
        #     action = 2
        #     amount = 0.
        #     # self.reward_step += -self.penalty_value

        # if masked_action != action:
        #     # amount = 0.
        #     self.invalid_action_counter += 1
        #     # self.reward_step += -self.penalty_value
        #     action = 2
        #     amount = 0

        real_action, real_size = self._take_action_func(action, amount)
        self.actions_lst.append(real_action)
        self.size_lst.append(real_size)
        # self.reward_step = ((self.pnl - self.previous_pnl) + 1) / (self.buy_and_hold_pnl + 1) - self.reward_step

        observation = self._get_obs()

        truncated = bool(self.invalid_action_counter >= self.invalid_actions)
        terminated = bool(self.pnl < self.pnl_stop)

        # don't move. must be before increasing timecount
        self.total_reward = self.total_assets - self.initial_total_assets
        self.timecount += 1

        if self.timecount == self.ohlcv_df.shape[0]:
            terminated = True
            # self.reward_step += (self.asset.balance.size * self.price) - (
            #         self.asset.balance.size * self.asset.balance.price)
            # self.reward_step = new_logarithmic_scaler(self.reward_step)
            # self.gamma_return = self.gamma_return * self.gamma + self.reward_step
            # self.reward_step += self.gamma_return
        if terminated or truncated:
            self.timecount -= 1
            self.dones = True
            # self.reward_step += self.pnl * 100. if self.pnl != 0. else self.pnl_stop * 100.
            self.reward_step += (self.pnl - (self.buy_and_hold_pnl * (
                    1 + (np.sign(self.buy_and_hold_pnl) * (1 - self.epsilon))))) if self.pnl != 0. else (
                    self.pnl_stop - (
                    self.buy_and_hold_pnl * (1 + (np.sign(self.buy_and_hold_pnl) * (1 - self.epsilon)))))

        # self.reward_step = new_logarithmic_scaler(self.reward_step)
        # self.previous_pnl = self.pnl
        # self.previous_price = self.price

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
               f"\tAssets: {self.current_balance}\tPNL {self.pnl:.5f} BH_PNL {self.buy_and_hold_pnl:.5f}"
               f"\t{actions_counted} \t#{self.total_episodes_counter:05d}")
        logger.info(msg)

    def reset(self, **kwargs):
        BinanceEnvCash.total_episodes_counter += 1 if self.use_period == 'train' else 0

        observation, info = super().reset(**kwargs)
        self.size_lst = []
        self.last_sell_order_pnl = 0.
        self.buy_and_hold_start_size = self.asset.balance.size + (self.initial_cash / self.price) * (
                1 - self.asset.orders.commission)
        return observation, info

    def __del__(self):
        BinanceEnvCash.count -= 1


class BinanceEnvMax(BinanceEnvBase):
    count: int = 0
    name = 'BinanceEnvMax'

    def __init__(self,
                 data_processor_kwargs: Union[dict, None],
                 target_balance: float = 100_000.,
                 target_minimum_trade: float = 5.,
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
                 max_lot_size=0.25,
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
                 deterministic: bool = True):
        super().__init__(data_processor_kwargs, target_balance, target_minimum_trade, coin_balance, pnl_stop, verbose,
                         log_interval, observation_type, action_type, use_period, stable_cache_data_n, reuse_data_prob,
                         eval_reuse_prob, seed, lookback_window, max_lot_size, max_hold_timeframes, penalty_value,
                         invalid_actions, total_timesteps, eps_start, eps_end, eps_decay, gamma, cache_obj, render_mode,
                         index_type, deterministic)

        BinanceEnvMax.count += 1

        self.idnum = int(BinanceEnvMax.count)

        self.order_opened: bool = False
        self.order_closed: bool = False
        self.order_opened_pnl: float = 0.
        self.order_closed_pnl: float = 0.
        self.buy_and_hold_start_size = self.asset.balance.size + (self.initial_cash / self.price) * (
                1 - self.asset.orders.commission)
        self.previous_price = self.price
        self.size_lst: list = []
        self.last_sell_order_pnl: float = 0.

    @property
    def buy_and_hold_pnl(self):
        return (self.buy_and_hold_start_size * self.price) / self.initial_total_assets - 1

    def _take_action(self, action, amount) -> tuple:
        old_target_balance = float(self.cash)
        old_coin_balance = float(self.asset.balance.size)
        action_commission = .0
        order_cash: float = 0.
        # action_symbol = self.target.symbol
        msg = str()
        size = 0.

        """buy or sell stock"""
        if action == 0:  # Buy
            self.action_symbol = f'{self.target.symbol}->{self.asset.symbol}'
            max_size = (self.cash / self.price) / (1. + self.asset.orders.commission)
            min_trade = max(self.asset.minimum_trade, self.target.minimum_trade / self.price)
            size = min(max(min_trade, amount), max_size if min_trade < max_size else 0.)
            if size != 0.:
                self.asset.orders.buy(size, self.price)
                action_commission = self.asset.orders.book[-1].order_commission
                order_cash = self.asset.orders.book[-1].order_cash
                # self.order_opened_pnl = self.pnl
                # self.order_opened = True
                # self.order_closed = False
                # self.reward_step += self.penalty_value * 10.
                # self.reward_step = self.pnl - self.buy_and_hold_pnl
            # else:
            #     action = 2

        elif action == 1:  # Sell
            self.action_symbol = f'{self.asset.symbol}->{self.target.symbol}'
            min_trade = max(self.min_coin_trade, self.target.minimum_trade / self.price)
            size = min(max(min_trade, amount),
                       self.asset.balance.size if min_trade < self.asset.balance.size else 0.)
            if size != 0:
                self.asset.orders.sell(size, self.price)
                # self.order_opened = False
                # self.order_closed = True
                action_commission = self.asset.orders.book[-1].order_commission
                order_cash = self.asset.orders.book[-1].order_cash
                self.reward_step = order_cash - self.asset.orders.book[-1].size * self.asset.balance.price
                self.reward_step = ((self.reward_step / self.initial_total_assets) - 1.) * 10.
                self.last_sell_order_pnl = float(self.pnl)
                # self.reward_step = order_cash - self.asset.orders.book[-1].size * self.asset.balance.price
                # self.reward_step = (self.pnl - (
                #         self.reward_step / self.initial_total_assets)) / self.buy_and_hold_pnl
                # self.reward_step += self.pnl - self.buy_and_hold_pnl
            # else:
            #     action = 2

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
                       f"\tACTION => {actions_reversed_dict[action]}: size:{size:.4f}({order_cash:.2f}) "
                       f"{self.action_symbol}, commission: {action_commission:.2f}"
                       f"\t{self.current_balance}\tprofit {self.reward_step:.2f}\tPNL {self.pnl:.4f}")
                logger.info(msg)

        return action, size

    def step(self, action):
        info = self._get_info()
        self.reward_step = .0
        # self.reward_step = -self.penalty_value

        action, amount = self.action_space_obj.convert2action(action, info['action_masks'])
        valid_actions = self.get_valid_actions(info['action_masks'])
        if action not in valid_actions:
            action = 2
            amount = 0.
            # self.reward_step += -self.penalty_value

        # if masked_action != action:
        #     # amount = 0.
        #     self.invalid_action_counter += 1
        #     # self.reward_step += -self.penalty_value
        #     action = 2
        #     amount = 0

        real_action, real_size = self._take_action_func(action, amount)
        self.actions_lst.append(real_action)
        self.size_lst.append(real_size)
        # self.reward_step = ((self.pnl - self.previous_pnl) + 1) / (self.buy_and_hold_pnl + 1) - self.reward_step

        observation = self._get_obs()

        truncated = bool(self.invalid_action_counter >= self.invalid_actions)
        terminated = bool(self.pnl < self.pnl_stop)

        # don't move. must be before increasing timecount
        self.total_reward = self.total_assets - self.initial_total_assets
        self.timecount += 1

        if self.timecount == self.ohlcv_df.shape[0]:
            terminated = True
            # self.reward_step += (self.asset.balance.size * self.price) - (
            #         self.asset.balance.size * self.asset.balance.price)
            # self.reward_step = new_logarithmic_scaler(self.reward_step)
            # self.gamma_return = self.gamma_return * self.gamma + self.reward_step
            # self.reward_step += self.gamma_return
        if terminated or truncated:
            self.timecount -= 1
            self.dones = True
            # self.reward_step += self.pnl * 100. if self.pnl != 0. else self.pnl_stop * 100.
            # self.reward_step += (self.pnl - (self.buy_and_hold_pnl * (
            #         1 + (np.sign(self.buy_and_hold_pnl) * 0.2)))) * 10. if self.pnl != 0. else (self.pnl_stop - (
            #         self.buy_and_hold_pnl * (1 + (np.sign(self.buy_and_hold_pnl) * 0.2)))) * 10.
            self.reward_step += ((self.pnl - self.last_sell_order_pnl) - 1.) * 10. if self.pnl != 0. else (
                                                                                                                  self.pnl_stop - 1.) * 10.

        # self.reward_step = new_logarithmic_scaler(self.reward_step)
        # self.previous_pnl = self.pnl
        # self.previous_price = self.price

        return observation, self.reward_step, terminated, truncated, info

    def log_reset_msg(self):
        values, counts = np.unique(self.actions_lst, return_counts=True)
        actions_counted: dict = dict(zip(values, counts))
        msg = (f"{self.__class__.__name__} #{self.idnum} {self.use_period}: "
               f"Ep.len: {self.timecount}({self.ohlcv_df.shape[0]}), cache: {len(self.CM.cache):03d}/"
               f"In.t.asset: {self.initial_total_assets:.0f} "
               # f"inv_act#: {self.invalid_action_counter:03d}/"
               f"Ep.reward: {self.episode_reward:.4f}"
               # f"\tepsilon: {self.eps_threshold:.6f}"
               f"\tprofit {self.total_reward:.1f}"
               # f"\steps_reward {self.eps_reward:.5f}"
               f"\tAssets: {self.current_balance}\tPNL {self.pnl:.5f} BH_PNL {self.buy_and_hold_pnl:.5f}"
               f"\t{actions_counted} \t#{self.total_episodes_counter:05d}")
        logger.info(msg)

    def reset(self, **kwargs):
        BinanceEnvMax.total_episodes_counter += 1 if self.use_period == 'train' else 0
        observation, info = super().reset(**kwargs)

        self.size_lst = []
        self.last_sell_order_pnl = 0.
        self.buy_and_hold_start_size = self.asset.balance.size + (self.initial_cash / self.price) * (
                1 - self.asset.orders.commission)
        return observation, info

    def __del__(self):
        BinanceEnvMax.count -= 1
