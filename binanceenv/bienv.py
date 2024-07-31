import sys
import math
import random
import logging
import gymnasium
import numpy as np
import numba
from numba import jit
from typing import Union

from gymnasium.spaces import Discrete
from gymnasium.spaces import Box
from gymnasium.utils import seeding
from collections import OrderedDict
from dbbinance.fetcher.cachemanager import CacheManager
from dbbinance.fetcher.datautils import get_timeframe_bins
from datawizard.dataprocessor import IndicatorProcessor
from datawizard.dataprocessor import Constants

__version__ = 0.021

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


class AssetsCloseIndicatorsSpace:
    def __init__(self, ind_num, assets_num):
        # low = np.zeros((assets_num + ind_num,))
        # high = np.ones((assets_num + ind_num,))
        # low[:assets_num] = 0.0
        # high[:assets_num] = 1.0
        self.__observation_space = Box(low=0.0, high=1.0, shape=(ind_num + assets_num + 1,), dtype=np.float32, seed=42)
        # self.__observation_space = Box(low=low, high=high, dtype=np.float32, seed=42)
        self.name = 'assets_close_indicators'

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


class ActionsBins:
    def __init__(self, box, n_actions):
        self.box_range = (np.max(box) - np.min(box))
        self.n_actions = n_actions
        self.step = self.box_range / n_actions
        self.bins = np.arange(box[0], box[1] + 1e-7, step=self.step, dtype=np.float32)
        self.pairs = [(self.bins[ix - 1], self.bins[ix]) for ix in range(1, len(self.bins))]

    @jit(nopython=True)
    def bins_2actions(self, value):
        for ix in range(self.n_actions):
            if np.min(self.pairs[ix]) < value < np.max(self.pairs[ix]):
                break
        return ix


class BinBoxActionSpace:
    def __init__(self, n_action):
        self.__action_space = Box(low=-1., high=1, shape=(1,), dtype=np.float32)
        self.actions_bins_obj = ActionsBins([-1, 1], n_action)
        self.name = 'binbox'

    def convert2action(self, action, masked_actions=None):
        return self.actions_bins_obj.bins_2actions(action)

    @property
    def action_space(self):
        return self.__action_space

    @action_space.setter
    def action_space(self, value):
        self.__action_space = value


# TODO: add amount converter from [-1, 1]
class BoxActionAmountSpace:
    def __init__(self, n_action):
        self.actions_bins_obj = ActionsBins([-1, 1], n_action)
        self.__action_space = Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.name = 'box_action_amount'

    def convert2action(self, action):
        return self.actions_bins_obj.bins_2actions(action[0])

    @property
    def action_space(self):
        return self.__action_space

    @action_space.setter
    def action_space(self, value):
        self.__action_space = value


class BoxActionSpace:
    def __init__(self, n_action):
        self.__action_space = Box(low=0, high=1, shape=(n_action,), dtype=np.float32)
        self.name = 'box'

    def convert2action(self, action, masked_actions=None):
        if masked_actions is None:
            return np.argmax(action)
        else:
            return np.ma.masked_array(action, mask=~masked_actions, fill_value=-np.inf).argmax(axis=0)

    @property
    def action_space(self):
        return self.__action_space

    @action_space.setter
    def action_space(self, value):
        self.__action_space = value


class BoxExtActionSpace:
    def __init__(self, n_action):
        self.__action_space = Box(low=-1., high=1, shape=(n_action,), dtype=np.float32)
        self.name = 'box1_1'

    def convert2action(self, action, masked_actions=None):
        if masked_actions is None:
            return np.argmax(action)
        else:
            return np.ma.masked_array(action, mask=~masked_actions, fill_value=-np.inf).argmax(axis=0)

    @property
    def action_space(self):
        return self.__action_space

    @action_space.setter
    def action_space(self, value):
        self.__action_space = value


@jit(nopython=True)
def logarithmic10_scaler(value):
    if value > 0:
        _temp = 1 / (1 + (math.e ** -(np.log10(value + 1))))  # original version
    else:
        _temp = 0.5 - (1 / (1 + (math.e ** -(np.log10(abs(value) + 1)))) - 0.5)
    return _temp


@jit(nopython=True)
def abs_logarithmic_scaler(value):
    _temp = 1 / (1 + (math.e ** -(np.log10(abs(value) + 1)))) - 0.5
    return _temp


cache_manager_obj = CacheManager(max_memory_gb=1)
eval_cache_manager_obj = CacheManager(max_memory_gb=1)


# class OrderBookMeta:
#     """
#     Class for storing and manipulating with asset orders
#     """
#
#     def __init__(self, assets_qty: int = 2):
#         """
#         Args:
#             assets_qty (int): assets qty (do not add target asset)
#
#         """
#         self.assets_qty = assets_qty
#         """
#         Book structure:
#         1st plane - asset qty
#                     example:    [USDT, BTC]
#                                 [100000, 0]
#         2nd plane - Order #1
#                     [qty,
#         """
#         self.__book = np.zeros((self.assets_qty+1),)
#
#     @property
#     def book(self):
#         return self.__book


class BinanceEnvBase(gymnasium.Env):
    # CM = cache_manager_obj
    # eval_CM = eval_cache_manager_obj

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
                 pnl_stop: float = -0.5,
                 verbose: int = 0,
                 log_interval=500,
                 observation_type='indicators',
                 action_type='discrete',
                 use_period='train',
                 reuse_data_prob=0.5,
                 eval_reuse_prob=0.1,
                 seed=41,
                 max_lot_size=0.25,
                 max_hold_timeframes='72h',
                 penalty_value=10,
                 total_timesteps: int = 3_000_000,
                 eps_start=0.95,
                 eps_end=0.01,
                 eps_decay=0.2,
                 render_mode=None):

        BinanceEnvBase.count += 1

        self.idnum = int(BinanceEnvBase.count)
        self.data_processor_obj = IndicatorProcessor(**data_processor_kwargs)
        self.max_timesteps = self.data_processor_obj.max_timesteps
        self.max_lot_size = max_lot_size

        self.max_hold_timeframes = int(
            get_timeframe_bins(max_hold_timeframes) / get_timeframe_bins(self.data_processor_obj.timeframe))
        self.penalty_value = penalty_value
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

        self.initial_target_balance = target_balance
        self.initial_coin_balance = coin_balance
        self.target_balance = float(self.initial_target_balance)
        self.coin_balance = float(self.initial_coin_balance)

        self.verbose = verbose

        self.ohlcv_df, self.indicators_df = self.data_processor_obj.get_ohlcv_and_indicators_sample(
            index_type='target_time')

        self.__get_obs_func = None
        self._take_action_func = None

        if self.use_period == 'train':
            self.CM = cache_manager_obj
            self._take_action_func = self._take_action_train
        else:
            # separated CacheManager for eval environment
            self.CM = eval_cache_manager_obj
            self._take_action_func = self._take_action_test
            self.reuse_data_prob = eval_reuse_prob

        self.action_space_obj = None
        self.observation_space = self.get_observation_space(observation_type=observation_type)
        self.action_space = self.get_action_space(action_type=action_type)  # {0, 1, 2}

        self._action_buy_hold_sell = ["Buy", "Sell", "Hold"]
        self.old_total_assets: float = 0.
        self.initial_total_assets = self.initial_target_balance + (self.initial_coin_balance * self.price)

        # self.action_bin_obj = ActionsBins([-1, 1], 3)
        self.actions_lst: list = []
        self.seed = seed + self.idnum
        self.coin_orders_values: float = 0
        self.actions_lst: list = []
        self.previous_cm_key = tuple()
        self.min_coin_trade = self.minimum_coin_trade_amount + (self.commission * self.minimum_coin_trade_amount)
        self.total_timesteps = total_timesteps
        self.total_timesteps_counter: int = 1

        self.action_symbol = str()
        self.order_closed = False
        self.reward_step = .0
        self.eps_reward = .0
        self.previous_pnl = .0
        self.order_pnl = .0
        self.reset()

    def __del__(self):
        BinanceEnvBase.count -= 1

    def get_observation_space(self, observation_type='indicators'):
        if observation_type == 'indicators_assets':
            space_obj = IndicatorsAndAssetsSpace(self.indicators_df.shape[1], 2)
            self.__get_obs_func = self._get_assets_indicators_obs
            # if self.coin_balance == .0:
            #     self.coin_balance = 1e-7
        elif observation_type == 'assets_close_indicators':
            space_obj = AssetsCloseIndicatorsSpace(self.indicators_df.shape[1], 2)
            self.__get_obs_func = self._get_assets_close_indicators_obs
            # if self.coin_balance == .0:
            #     self.coin_balance = 1e-7
        elif observation_type == 'assets_close_action_masks_indicators':
            space_obj = AssetsCloseIndicatorsSpace(self.indicators_df.shape[1], 2 + 3)
            self.__get_obs_func = self._get_assets_close_action_masks_indicators_obs
        elif observation_type == 'indicators_pnl':
            space_obj = IndicatorsAndPNLSpace(self.indicators_df.shape[1], 1)
            self.__get_obs_func = self._get_pnl_indicators_obs
        elif observation_type == 'indicators':
            space_obj = IndicatorsSpace(self.indicators_df.shape[1])
            self.__get_obs_func = self._get_indicators_obs
        else:
            sys.exit(f'Error: Unknown observation type {observation_type}!')
        observation_space = space_obj.observation_space
        self.name = f'{self.name}_{space_obj.name}'
        return observation_space

    def get_action_space(self, action_type='discrete'):
        self.action_space_obj = DiscreteActionSpace(3)
        if action_type == 'box':
            self.action_space_obj = BoxActionSpace(n_action=3)
        if action_type == 'box1_1':
            self.action_space_obj = BoxExtActionSpace(n_action=3)
        elif action_type == 'binbox':
            self.action_space_obj = BinBoxActionSpace(n_action=3)
        action_space = self.action_space_obj.action_space
        self.name = f'{self.name}_{self.action_space_obj.name}'
        return action_space

    def calc_eps_treshold(self):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1. * self.total_timesteps_counter / (self.total_timesteps * self.eps_decay))
        return eps_threshold

    @property
    def price(self):
        return self.ohlcv_df.iloc[self.timecount]['close']

    @property
    def pnl(self):
        return (self.total_assets / self.initial_total_assets) - 1.

    @property
    def total_assets(self):
        return self.target_balance + (self.coin_balance * self.price)

    @property
    def current_balance(self) -> str:
        current_balance = (
            f"CURRENT Assets => {self.target_symbol}: {self.target_balance:.2f}, {self.coin_symbol}: "
            f"{self.coin_balance:.6f}\tTotal assets: {self.total_assets:.2f} {self.target_symbol}")
        return current_balance

    # @property
    # def coin_orders_cost(self):
    #     return sum(self.coin_orders_values)

    def _get_obs(self):
        data = self.__get_obs_func()
        logger.debug(data)
        return data

    def _get_assets_indicators_obs(self):
        target = abs_logarithmic_scaler(self.target_balance)
        # coin = logarithmic10_scaler(self.coin_balance)
        coin = abs_logarithmic_scaler(self.coin_balance * self.price)
        # coin_orders_cost = logarithmic10_scaler(self.coin_orders_cost)
        return np.asarray(
            np.concatenate([[target, coin], self.indicators_df.iloc[self.timecount].values]),
            dtype=np.float32)

    def _get_assets_close_indicators_obs(self):
        target = abs_logarithmic_scaler(self.target_balance)
        # coin = logarithmic10_scaler(self.coin_balance)
        coin = abs_logarithmic_scaler(self.coin_balance)
        close = abs_logarithmic_scaler(self.price)
        # coin_orders_cost = logarithmic10_scaler(self.coin_orders_cost)
        return np.asarray(
            np.concatenate([[target, coin, close], self.indicators_df.iloc[self.timecount].values]),
            dtype=np.float32)

    def _get_assets_close_action_masks_indicators_obs(self):
        target = abs_logarithmic_scaler(self.target_balance)
        # coin = logarithmic10_scaler(self.coin_balance)
        coin = abs_logarithmic_scaler(self.coin_balance * self.price)
        close = abs_logarithmic_scaler(self.price)
        # coin_orders_cost = logarithmic10_scaler(self.coin_orders_cost)
        action_masks = self._get_action_masks().astype(np.float32)
        return np.asarray(
            np.concatenate([[target, coin, close], action_masks, self.indicators_df.iloc[self.timecount].values]),
            dtype=np.float32)

    # def _get_complex_obs(self):
    #     target = abs_logarithmic_scaler(self.target_balance)
    #     # coin = logarithmic10_scaler(self.coin_balance)
    #     coin = abs_logarithmic_scaler(self.coin_balance)
    #     close = abs_logarithmic_scaler(self.price)
    #     # coin_orders_cost = logarithmic10_scaler(self.coin_orders_cost)
    #     return np.asarray(
    #         np.concatenate([[target, coin, close],self.indicators_df.iloc[self.timecount].values]),
    #         dtype=np.float32)

    def _get_pnl_indicators_obs(self):
        return np.asarray(np.concatenate([[self.pnl], self.indicators_df.iloc[self.timecount].values]),
                          dtype=np.float32)

    def _get_indicators_obs(self):
        return np.asarray(self.indicators_df.iloc[self.timecount].values, dtype=np.float32)

    def _get_action_masks(self) -> np.ndarray:
        return np.array(
            [self.target_balance / self.price >= self.min_coin_trade, self.coin_balance >= self.min_coin_trade, True])

    def _get_info(self) -> dict:
        return {"action_masks": self._get_action_masks()}

    def buy_order(self, size):
        self.action_symbol = f'{self.target_symbol}->{self.coin_symbol}'
        size_price = size * self.price
        action_commission = size_price * self.commission
        self.target_balance -= (size_price + action_commission)
        self.coin_balance += size
        self.coin_orders_values = size_price
        self.order_closed = False
        self.order_pnl = float(self.pnl)
        return .0

    def sell_order(self, size):
        self.action_symbol = f'{self.coin_symbol}->{self.target_symbol}'
        size_price = size * self.price
        action_commission = size_price * self.commission
        self.target_balance += (size_price - action_commission)
        self.coin_balance -= size
        if self.order_pnl > .0:
            sell_reward = self.pnl - self.order_pnl
            self.order_pnl = .0
        else:
            sell_reward = 0.
        self.coin_orders_values = .0
        self.order_closed = True
        return sell_reward

    def hold_order(self, target_size):
        hold_reward = .0
        if len(self.actions_lst) > self.max_hold_timeframes:  # Hold
            # Penalty actions for just holding
            check_actions = np.array(self.actions_lst[-self.max_hold_timeframes:]) - 1
            if np.all(check_actions):
                if not self.order_closed:
                    self.target_balance -= target_size
                else:
                    self.coin_balance -= (target_size / self.price)
                hold_reward = self.pnl - self.previous_pnl
        return hold_reward

    def _take_action_with_penalty(self, action, amount):
        old_target_balance = float(self.target_balance)
        old_coin_balance = float(self.coin_balance)
        action_commission = .0
        action_symbol = self.target_symbol
        size = .0
        msg = str()


        # stock_action = action[0]
        # """buy or sell stock"""
        # adj = self.day_price[0]
        # if stock_action < 0:
        #     stock_action = max(
        #         0, min(-1 * stock_action, 0.5 * self.total_asset / adj + self.stocks)
        #     )
        #     self.account += adj * stock_action * (1 - self.transaction_fee_percent)
        #     self.stocks -= stock_action
        # elif stock_action > 0:
        #     max_amount = self.account / adj
        #     stock_action = min(stock_action, max_amount)
        #     self.account -= adj * stock_action * (1 + self.transaction_fee_percent)




        if action == 0:  # buy
            size = (self.target_balance / self.price) / (1 + self.commission)
            self.reward_step = self.buy_order(size=size)
            self.eps_reward += self.reward_step

        elif action == 1:  # sell
            size = old_coin_balance
            self.reward_step = self.sell_order(size=size)
            self.eps_reward += self.reward_step

        elif action == 2:  # hold
            size = self.penalty_value
            self.reward_step = self.hold_order(size)

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
                    f"OLD Assets =>\t{self.target_symbol}: {old_target_balance:.4f}, "
                    f"{self.coin_symbol}: {old_coin_balance:.4f}"
                    f"\tTotal assets(old): "
                    f"{(old_target_balance + (old_coin_balance * self.price)):.1f} {self.target_symbol}")
                msg = (f"{ohlcv}\n{msg}"
                       f"\tAction num: {action}\t{old_balance} "
                       f"\tACTION => {self._action_buy_hold_sell[action]}: size:{size:.4f}({amount:.3f}) "
                       f"{action_symbol}, commission: {action_commission:.6f}"
                       f"\t{self.current_balance}\treward {self.reward:.2f}\tPNL {self.pnl:.5f}")
                logger.info(msg)

    def _take_action_train(self, action, amount):
        self._take_action_with_penalty(action, amount)

    def _take_action_test(self, action, amount):
        self._take_action_with_penalty(action, amount)

    def step(self, action):

        amount = 1.
        info = self._get_info()
        self.reward_step = .0
        if self.action_type in ['box1_1', 'box', 'binbox']:
            # action = self.action_space_obj.convert2action(action, None)
            # action = self.action_space_obj.convert2action(action, info['action_masks'])
            if self.eps_threshold < random.random():
                action = self.action_space_obj.convert2action(action, info['action_masks'])
            else:
                action = self.action_space_obj.convert2action(action, None)
        self.actions_lst.append(action)
        self._take_action_func(action, amount)

        observation = self._get_obs()

        self.reward = self.total_assets - self.initial_total_assets

        terminated = bool(self.pnl < self.pnl_stop)
        # self.reward += (self.total_assets - self.old_total_assets)
        self.old_total_assets = float(self.total_assets)
        self.previous_pnl = float(self.pnl)

        # delay_modifier = (self.timecount / self.max_timesteps)
        # discounted_reward = self.reward * delay_modifier
        self.timecount += 1

        if self.timecount == self.ohlcv_df.shape[0]:
            terminated = True
            self.timecount -= 1
            self.reward_step = self.eps_reward
            # if not self.order_closed and self.coin_balance > .0:
            #     size = self.coin_balance
            #     self.reward_step = self.sell_order(size=size)
            #     self.eps_reward += self.reward_step
            #     self.reward_step += self.eps_reward

        return observation, self.reward_step, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total_timesteps_counter += self.timecount

        if self.verbose:
            values, counts = np.unique(self.actions_lst, return_counts=True)
            actions_counted: dict = dict(zip(values, counts))
            msg = (f"{self.__class__.__name__} #{self.idnum} {self.use_period}: Episode length: {self.timecount},"
                   f"\tcache_len: {len(self.CM.cache)}"
                   f"\tepsilon: {self.eps_threshold:.6f}"
                   f"\treward {self.reward:.2f}"
                   f"\teps_reward {self.eps_reward:.5f}"
                   f"\tAssets: {self.current_balance}\tPNL "
                   f"{self.pnl:.6f}\t{actions_counted}\treset")
            logger.info(msg)

        self.target_balance = float(self.initial_target_balance)
        self.coin_balance = float(self.initial_coin_balance)
        self.timecount: int = 0
        self.reward = 0.
        self.reward_step = 0.
        self.actions_lst: list = []

        if self.use_period == 'test':
            if len(self.CM.cache) >= 50 and self.reuse_data_prob < 1.0:
                self.reuse_data_prob = 0.95
        else:
            self.eps_threshold = self.calc_eps_treshold()

        if self.reuse_data_prob > random.random() and len(self.CM.cache) >= 20:
            while True:
                cm_key = random.sample(list(self.CM.cache.keys()), 1)[0]
                if cm_key != self.previous_cm_key:
                    break
            self.ohlcv_df, self.indicators_df = self.CM.cache[cm_key]
            self.previous_cm_key = cm_key
        else:
            self.ohlcv_df, self.indicators_df = self.data_processor_obj.get_random_ohlcv_and_indicators(
                period_type=self.use_period)
            if self.ohlcv_df.shape[0] != self.indicators_df.shape[0]:
                msg = (f"{self.__class__.__name__} #{self.idnum}: ohlcv_df.shape = {self.ohlcv_df.shape}, "
                       f"indicators_df.shape = {self.indicators_df.shape}")
                logger.debug(msg)
                sys.exit('Error: Check data_processor, length of data is not equal!')

            cm_key = tuple((self.ohlcv_df.index[0], self.ohlcv_df.index[-1]))
            self.CM.update_cache(key=cm_key, value=(self.ohlcv_df, self.indicators_df))
            self.previous_cm_key = cm_key

        self.initial_total_assets = self.initial_target_balance + (self.initial_coin_balance * self.price)
        self.coin_orders_values = .0
        self.order_closed = False
        self.eps_reward = .0

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
