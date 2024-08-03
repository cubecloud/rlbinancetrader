import sys
import math
import random
import logging
import gymnasium
import numpy as np
import numba
from numba import jit
# from typing import Union

# from gymnasium.spaces import Discrete
# from gymnasium.spaces import Box
from gymnasium.utils import seeding
# from collections import OrderedDict
from dbbinance.fetcher.cachemanager import CacheManager
from dbbinance.fetcher.datautils import get_timeframe_bins
from datawizard.dataprocessor import IndicatorProcessor
# from datawizard.dataprocessor import Constants
from binanceenv.spaces import *

__version__ = 0.027

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


cache_manager_obj = CacheManager(max_memory_gb=1)
eval_cache_manager_obj = CacheManager(max_memory_gb=1)


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
                 invalid_actions=60,
                 total_timesteps: int = 3_000_000,
                 eps_start=0.95,
                 eps_end=0.01,
                 eps_decay=0.2,
                 gamma=0.99,
                 render_mode=None):

        BinanceEnvBase.count += 1

        self.idnum = int(BinanceEnvBase.count)
        self.data_processor_obj = IndicatorProcessor(**data_processor_kwargs)
        self.max_timesteps = self.data_processor_obj.max_timesteps
        self.max_lot_size = max_lot_size

        self.max_hold_timeframes = int(
            get_timeframe_bins(max_hold_timeframes) / get_timeframe_bins(self.data_processor_obj.timeframe))
        self.invalid_actions = invalid_actions
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
        self.gamma = gamma
        self.gamma_return = 0.

        self.timecount: int = 0
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

        # self._action_buy_hold_sell = ["Buy", "Sell", "Hold"]
        self.old_total_assets: float = 0.
        self.initial_total_assets = self.initial_target_balance + (self.initial_coin_balance * self.price)

        # self.action_bin_obj = ActionsBins([-1, 1], 3)
        self.actions_lst: list = []
        self.seed = self.get_seed(seed) + self.idnum
        self.coin_orders_values: float = .0
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
        self.all_actions = np.asarray(list(range(3)))
        self.invalid_action_counter = 0
        self.reset()

    def __del__(self):
        BinanceEnvBase.count -= 1

    def get_observation_space(self, observation_type='indicators'):
        if observation_type == 'indicators_assets':
            space_obj = IndicatorsAndAssetsSpace(self.indicators_df.shape[1], 2)
            self.__get_obs_func = self._get_assets_indicators_obs
            if self.coin_balance == .0:
                self.coin_balance = 1e-7
        elif observation_type == 'assets_close_indicators':
            space_obj = AssetsCloseIndicatorsSpace(self.indicators_df.shape[1], 2)
            self.__get_obs_func = self._get_assets_close_indicators_obs
            if self.coin_balance == .0:
                self.coin_balance = 1e-7
        elif observation_type == 'idx_assets_close_indicators_action_masks':
            space_obj = AssetsCloseIndicatorsSpace(self.indicators_df.shape[1], 2 + 1 + 3)
            self.__get_obs_func = self._get_idx_assets_close_indicators_action_masks_obs
            if self.coin_balance == .0:
                self.coin_balance = 1e-7
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
        if action_type == 'discrete':
            self.action_space_obj = DiscreteActionSpace(3)
        elif action_type == 'box':
            self.action_space_obj = BoxActionSpace(n_action=3)
        elif action_type == 'box1_1':
            self.action_space_obj = BoxExtActionSpace(n_action=3)
        elif action_type == 'binbox':
            self.action_space_obj = BinBoxActionSpace(n_action=3, low=-1, high=1)
        elif action_type == 'sell_buy_amount':
            self.action_space_obj = SellBuyAmount(assets_qty=1)
        else:
            sys.exit(f'Error: Unknown action type {action_type}!')
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
            f"CURRENT Assets => {self.target_symbol}: {self.target_balance:.1f}, {self.coin_symbol}: "
            f"{self.coin_balance:.6f}\tTotal assets: {self.total_assets:.1f} {self.target_symbol}")
        return current_balance

    # @property
    # def coin_orders_cost(self):
    #     return sum(self.coin_orders_values)

    def _get_obs(self):
        data = self.__get_obs_func()
        logger.debug(data)
        return data

    def _get_assets_indicators_obs(self):
        target = new_logarithmic_scaler(self.target_balance)
        # coin = logarithmic10_scaler(self.coin_balance)
        coin = new_logarithmic_scaler(self.coin_balance * self.price)
        # coin_orders_cost = logarithmic10_scaler(self.coin_orders_cost)
        return np.asarray(
            np.concatenate([[target, coin], self.indicators_df.iloc[self.timecount].values]),
            dtype=np.float32)

    def _get_assets_close_indicators_obs(self):
        target = new_logarithmic_scaler(self.target_balance)
        # coin = logarithmic10_scaler(self.coin_balance)
        close = new_logarithmic_scaler(self.price)
        # coin_orders_cost = logarithmic10_scaler(self.coin_orders_cost)
        coin = new_logarithmic_scaler(self.coin_balance)
        return np.asarray(
            np.concatenate([[target, close], self.indicators_df.iloc[self.timecount].values, [coin]]),
            dtype=np.float32)

    def _get_idx_assets_close_indicators_action_masks_obs(self):
        idx = self.timecount * 1e-6
        target = new_logarithmic_scaler(self.target_balance)
        # coin = logarithmic10_scaler(self.coin_balance)
        close = new_logarithmic_scaler(self.price)
        # coin_orders_cost = logarithmic10_scaler(self.coin_orders_cost)
        coin = new_logarithmic_scaler(self.coin_balance)
        action_masks = self._get_action_masks().astype(np.float32)
        return np.asarray(
            np.concatenate([[idx, target, coin, close], self.indicators_df.iloc[self.timecount].values, action_masks]),
            dtype=np.float32)

    def _get_assets_close_action_masks_indicators_obs(self):
        target = new_logarithmic_scaler(self.target_balance)
        # coin = logarithmic10_scaler(self.coin_balance)
        coin = new_logarithmic_scaler(self.coin_balance * self.price)
        close = new_logarithmic_scaler(self.price)
        # coin_orders_cost = logarithmic10_scaler(self.coin_orders_cost)
        action_masks = self._get_action_masks().astype(np.float32)
        return np.asarray(
            np.concatenate([[target, coin, close], action_masks, self.indicators_df.iloc[self.timecount].values]),
            dtype=np.float32)

    def _get_pnl_indicators_obs(self):
        return np.asarray(np.concatenate([[self.pnl], self.indicators_df.iloc[self.timecount].values]),
                          dtype=np.float32)

    def _get_indicators_obs(self):
        return np.asarray(self.indicators_df.iloc[self.timecount].values, dtype=np.float32)

    def _get_action_masks(self) -> np.ndarray:
        return np.array(
            [self.target_balance / self.price >= self.min_coin_trade,
             self.coin_balance >= self.min_coin_trade,
             self.not_max_holding_time(self.max_hold_timeframes)])

    def _get_info(self) -> dict:
        return {"action_masks": self._get_action_masks()}

    # def buy_order(self, size):
    #     self.action_symbol = f'{self.target_symbol}->{self.coin_symbol}'
    #     size_price = size * self.price
    #     action_commission = size_price * self.commission
    #     self.target_balance -= (size_price + action_commission)
    #     self.coin_balance += size
    #     self.coin_orders_values = size_price
    #     self.order_closed = False
    #     self.order_pnl = float(self.pnl)
    #     return .0

    # def sell_order(self, size):
    #     self.action_symbol = f'{self.coin_symbol}->{self.target_symbol}'
    #     size_price = size * self.price
    #     action_commission = size_price * self.commission
    #     self.target_balance += (size_price - action_commission)
    #     self.coin_balance -= size
    #     # if self.order_pnl > .0:
    #     #     sell_reward = self.pnl - self.order_pnl
    #     #     self.order_pnl = .0
    #     # else:
    #     #     sell_reward = 0.
    #     self.coin_orders_values = .0
    #     self.order_closed = True
    #     return sell_reward

    def not_max_holding_time(self, max_hold_timeframes) -> bool:
        check = True
        if len(self.actions_lst) > self.max_hold_timeframes:  # Hold
            # Penalty actions for just holding
            if np.all(np.array(self.actions_lst[-max_hold_timeframes:]) - 1):
                check = False
        return check

    def _take_action(self, action, amount):
        old_target_balance = float(self.target_balance)
        old_coin_balance = float(self.coin_balance)
        action_commission = .0
        action_symbol = self.target_symbol
        msg = str()
        size = 0.

        """buy or sell stock"""
        if action == 0:  # Buy
            self.action_symbol = f'{self.target_symbol}->{self.coin_symbol}'
            max_amount = (self.target_balance / self.price) / (1. + self.commission)
            size = min(max(self.minimum_coin_trade_amount, amount * max_amount), max_amount)
            size_price = self.price * size
            action_commission = size_price * self.commission
            self.target_balance -= (size_price + action_commission)
            self.coin_balance += size

        elif action == 1:  # Sell
            self.action_symbol = f'{self.coin_symbol}->{self.target_symbol}'
            size = min(max(self.min_coin_trade, amount * self.coin_balance), self.coin_balance)
            size_price = self.price * size
            action_commission = size_price * self.commission
            self.target_balance += (size_price - action_commission)
            self.coin_balance -= size

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
                       f"\tACTION => {actions_dict[action]}: size:{size:.4f}({amount:.3f}) "
                       f"{action_symbol}, commission: {action_commission:.6f}"
                       f"\t{self.current_balance}\tprofit {self.reward:.2f}\tPNL {self.pnl:.5f}")
                logger.info(msg)

    def _take_action_train(self, action, amount):
        self._take_action(action, amount)

    def _take_action_test(self, action, amount):
        self._take_action(action, amount)

    def get_valid_actions(self, action_masks) -> list:
        return self.all_actions[action_masks]

    def step(self, action):
        truncated = False
        terminated = False
        masked_action = 0
        action_penalty = False
        amount = 1.
        info = self._get_info()
        if self.use_period == 'train':
            self.reward_step = -2e-4
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

        elif self.action_type in ['sell_buy_amount', 'binbox']:
            action, amount = self.action_space_obj.convert2action(action[0])

        if masked_action != action:
            action = masked_action
            self.invalid_action_counter += 1
            if self.use_period == 'train':
                action_penalty = True
                self.reward_step += -3e-4
        else:
            if self.use_period == 'train':
                self.reward_step += 2e-4

        # valid_actions = self.get_valid_actions(info['action_masks'])
        # if action not in valid_actions:
        #     if self.use_period == 'train':
        #         # amount = .0
        #         action = self.np_random.choice(valid_actions, 1)[0]
        #         if action in [0, 1]:
        #             action_penalty = False
        #             # self.reward_step -= 1e-5
        #         else:
        #             action_penalty = True
        #             # self.reward_step -= 2e-5
        #         # truncated = bool(self.invalid_action_counter >= self.invalid_actions)
        #     else:
        #         action = 2  # hold
        #     self.invalid_action_counter += 1
        # else:
        #     if self.use_period == 'train':
        #         if action in [0, 1]:
        #             self.reward_step += 1e-5
        #         else:
        #             self.reward_step += 1e-7

        truncated = bool(self.invalid_action_counter >= self.invalid_actions)
        self.actions_lst.append(action)
        self._take_action_func(action, amount)

        observation = self._get_obs()

        self.reward = self.total_assets - self.initial_total_assets
        if not action_penalty:
            self.reward_step += self.pnl - self.previous_pnl
        # else:
        #     # penalty for invalid action
        #     self.reward_step += -2e-5
        self.old_total_assets = float(self.total_assets)
        self.previous_pnl = float(self.pnl)

        self.gamma_return = self.gamma_return * self.gamma + self.reward_step

        terminated = bool(self.pnl < self.pnl_stop)
        self.timecount += 1
        if self.timecount == self.ohlcv_df.shape[0]:
            terminated = True
            if self.use_period == 'train':
                self.reward_step = self.gamma_return
            self.timecount -= 1

        return observation, self.reward_step, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is None:
            self.seed = self.get_seed()
        super().reset(seed=seed)
        self.total_timesteps_counter += self.timecount

        if self.verbose:
            values, counts = np.unique(self.actions_lst, return_counts=True)
            actions_counted: dict = dict(zip(values, counts))
            msg = (f"{self.__class__.__name__} #{self.idnum} {self.use_period}: "
                   f"Ep.length(shape): {self.timecount}({self.ohlcv_df.shape[0]}), cache: {len(self.CM.cache):03d}/"
                   f"inv_act#: {self.invalid_action_counter:03d}/g_return: {self.gamma_return:.4f}"
                   # f"\tepsilon: {self.eps_threshold:.6f}"
                   f"\tprofit {self.reward:.1f}"
                   # f"\teps_reward {self.eps_reward:.5f}"
                   f"\tAssets: {self.current_balance}\tPNL "
                   f"{self.pnl:.5f}\t{actions_counted}\treset")
            logger.info(msg)

        self.target_balance = float(self.initial_target_balance)
        self.coin_balance = float(self.initial_coin_balance)
        self.timecount: int = 0
        self.reward = 0.
        self.reward_step = 0.
        self.actions_lst: list = []
        self.invalid_action_counter = 0
        self.gamma_return = 0.

        if self.use_period == 'test':
            if len(self.CM.cache) >= 50 and self.reuse_data_prob < 1.0:
                self.reuse_data_prob = 1.0
        # else:
        #     self.eps_threshold = self.calc_eps_treshold()

        if self.reuse_data_prob > self.np_random.random() and len(self.CM.cache) >= 20:
            while True:
                cm_key = tuple(self.np_random.choice(list(self.CM.cache.keys()), 1)[0])
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
        self.gamma_return = .0

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def close(self):
        if self.verbose:
            if self.verbose:
                msg = (f"Episode length: {self.timecount}\treward {self.reward:.2f}\tAssets: "
                       f"{self.current_balance}\tPNL {self.pnl:.6f}\tclose")
                logger.info(msg)

    def get_seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed
