from abc import ABC, abstractmethod

from numpy import ndarray

from binanceenv.orderbook import TradesBook, Asset, Trade
from typing import Optional, List
import numpy as np
import pandas as pd
import math

__version__ = 0.017


def get_sigmoid_results(max_len: int = 600, normalization_window: int = 96) -> np.array:
    """
    calculate sigmoid results with parameters to calculate sigmoid values.

    Args:
        max_len (int): The maximum length of x values.
        normalization_window (int, optional): Normalization window for scaling x. Defaults to 96.
    Returns:
        np.ndarray
    """

    x_values = np.arange(0, max_len + 1, dtype=np.float32)
    sigmoid_results = 1 / (1 + np.exp(-x_values / normalization_window))
    return sigmoid_results


class AnyPeriod:

    def __init__(self, normalization_window: int):
        self.entry_datetime = None
        self.exit_datetime = None
        self.entry_price = None
        self.exit_price = None
        self.size: float = 0.0
        self.reward: List[float] = []
        self.sigmoid_results: np.array = get_sigmoid_results(max_len=normalization_window * 6,
                                                             normalization_window=normalization_window)
        self.gamma_reward: float = 0.0

    @property
    def normalized_steps(self) -> float:
        return self.sigmoid_results[len(self.reward)]

    def reset(self):
        self.entry_datetime = None
        self.exit_datetime = None
        self.entry_price = None
        self.exit_price = None
        self.size: float = 0.0
        self.reward: List[float] = []
        self.gamma_reward: float = 0.0


class WaitPeriod(AnyPeriod):
    pass


class HoldPeriod(AnyPeriod):
    pass


class RewardsBase(ABC):
    def __init__(self,
                 asset: Asset,
                 gamma=0.93,
                 loss_threshold: float = 0.0087,
                 profit_threshold: float = 0.011,
                 normalization_window: int = 12,
                 use_final_reward: bool = False):
        self.__asset = asset
        self.gamma = gamma
        self.loss_threshold = loss_threshold
        self.profit_threshold = profit_threshold
        self.window_size = normalization_window
        self.use_final_reward = use_final_reward
        self.__trades: TradesBook = asset.trades
        self.ohlcv_df: Optional[pd.DataFrame] = None
        self.raw_rewards: List[float] = []

        # Local initialization of TALib, for multiprocessing
        self.talib = None
        self.current_wait_period = WaitPeriod(self.window_size)
        self.current_hold_period = HoldPeriod(self.window_size)

    def _init_lib(self):
        # Called only once per instance, in the child process
        if self.talib is None:
            import talib
            self.talib = talib

    def pnl(self, profit):
        return profit / (self.__asset.target.initial_cash + (
                self.__asset.initial_balance.size * self.__asset.initial_balance.price))

    def win_rate_score(self, weight=0.55):
        win_rate = self.__trades.win_rate
        if win_rate > 0.:
            if win_rate > 0.55:
                score = np.exp(win_rate) * weight
            else:
                score = np.exp(win_rate) * (weight ** 2)
        else:
            score = -np.exp(-win_rate) * weight
        return score * 0.001

    def pnl_score(self, buy_and_hold_pnl: float, weight: float = 0.8):
        pnl = self.pnl(self.__trades.profit)
        if pnl > .0:
            if pnl > buy_and_hold_pnl:
                score = np.exp(pnl)
            else:
                score = np.exp(pnl) * weight
        else:
            score = -np.exp(-pnl) * (weight ** 2)
        return score * 0.01

    def get_normalized_reward(self, window_size):
        if len(self.raw_rewards) <= 2:
            # Not enough data points yet; return the last raw reward
            return self.raw_rewards[-1]

        # Take the tail of the rewards with size equal to window_size
        # Calculate mean and standard deviation using numpy
        mean = np.mean(self.raw_rewards[-window_size:])
        std = np.std(self.raw_rewards[-window_size:])

        # Normalize the last reward (or current if needed)
        last_reward = self.raw_rewards[-1]
        normalized = (last_reward - mean) / (std + 1e-7)  # Add small epsilon to avoid division by zero

        return normalized

    def final_reward(self, buy_and_hold_pnl) -> float:
        _final_reward = 0.0
        _reward_constant = 1e-5

        if self.use_final_reward:
            """
            id no trades and B&H PNL <=0 
            we return penalized reward 
            """
            if self.__trades.trades_qty == 0:
                return -_reward_constant * 1000

            #   calculating mean reward
            # _final_reward = 0. if not len(self.raw_rewards) else np.mean(self.raw_rewards)
            episode_relative_pnl = self.pnl(self.__trades.profit)
            return 0. if not len(self.raw_rewards) else episode_relative_pnl / len(self.raw_rewards)

            # """
            # Changed final reward calculations (was +0.5):
            # # PNL > B&H   => +0.0125
            # # PNL > 0.    => +0.0125
            # PNL > B&H   => +0.00001 else -0.00001
            # PNL > 0.    => +0.00001 else -0.00001
            # """
            # if episode_relative_pnl > .0:
            #     _final_reward += _reward_constant * 10
            #     if episode_relative_pnl > buy_and_hold_pnl:
            #         _final_reward += _reward_constant * 10
            #
            # else:
            #     _final_reward -= _reward_constant * 10
            #     if episode_relative_pnl <= buy_and_hold_pnl:
            #         _final_reward -= _reward_constant * 10
            #
            # if self.__trades.profit_rate > 0.65:
            #     _final_reward += _reward_constant * 5
            # #   if profit negative and profit_rate negative too
            # elif self.__trades.profit_rate <= -0.75:
            #     _final_reward -= _reward_constant * 5

        return _final_reward

    def size(self, price, cash) -> float:
        """
        Calculate size based on cash and price
        Returns:
            size (float)
        """
        max_size = (cash / price) / (1. + self.__asset.orders.commission)
        min_trade = max(self.__asset.minimum_trade, self.__asset.target.minimum_trade / price)
        # max_trade = min(max_size if max_size > min_trade else 0., self.target.maximum_trade / self.price)
        max_trade = max_size if max_size > min_trade else 0.
        size = max(min_trade, max_trade)
        return size

    def trade_loss_drawdown_weight(self, prices: pd.Series, penalty_per_step: float = 0.01) -> float:
        """Calculates the maximum drawdown relative to the highest achieved price with additional penalty per step after the peak.

        If there are no data points after the maximum price,
        it assumes that there's no further drawdown and returns zero.

        Args:
            prices (pd.Series): Series of prices for analysis.
            penalty_per_step (float): Penalty percentage added for each step after the maximum price.

        Returns:
            float: weight
        """
        # Find the index of the maximum value
        index_max = prices.values.argmax()

        # Check if the maximum price is the last element in the series
        if index_max == prices.shape[0] - 1:
            # Maximum price is at the end, so assume no drawdown
            return 0.0

        # Calculate the number of steps after the maximum price
        steps_after_peak = prices.shape[0] - index_max - 1

        # Apply penalty based on the number of steps after the peak
        penalty_amount = steps_after_peak * penalty_per_step

        # Get the minimum price after the maximum
        min_after_max = prices.iloc[index_max + 1:].min()
        max_price = prices.iloc[index_max]

        # Calculate the drawdown relative to the maximum
        drawdown = (max_price - min_after_max) / max_price

        # Multiply drawdown by (1 + penalty_amount) to increase it proportionally
        increased_drawdown = drawdown * (1 + penalty_amount)

        return max(0.0, increased_drawdown - self.loss_threshold)

    def wait_potential_loss(self, prices: pd.Series) -> float:
        """Calculates the potential maximum loss relative to the highest achieved price.

        Args:
            prices (pd.Series): Series of prices for analysis.

        Returns:
            float: weight
         """
        max_price = prices.max()
        potential_loss = (max_price - prices[-1]) / max_price  # Potential loss considering current price as highest
        return max(0, potential_loss - self.loss_threshold)  # Ensure non-negative weight

    def trade_period_weight(self) -> float:
        """
        Calculate the weight for a single trade based on drawdown metrics.
        The weight is always non-zero and provides meaningful feedback for every trade
        """

        # Extract trade data
        trade = self.__trades.last_trade
        entry_datetime = trade.entry_datetime
        exit_datetime = trade.exit_datetime

        """ Get the trade prices and calculate score based on drawdown from entry_price """
        trade_prices = self._trade_prices(entry_datetime, exit_datetime)
        weight = self.trade_loss_drawdown_weight(trade_prices)
        return weight

    def wait_period_weight(self) -> float:
        """
        Calculates the potential maximum loss (weight) relative to the highest and lowest achieved price

        Returns:
            float: weight
        """
        # Extract trade data
        entry_datetime = self.current_wait_period.entry_datetime
        exit_datetime = self.current_wait_period.exit_datetime

        # If entry and exit dates are equal, consider no loss
        if entry_datetime == exit_datetime:
            return 0.0  # Minimize weight if dates are identical

        trade_prices = self._trade_prices(entry_datetime, exit_datetime)

        min_price = trade_prices.min()
        max_price = trade_prices.max()
        entry_price = self.current_wait_period.entry_price

        # Calculate potential loss assuming current price reached its maximum
        loss_from_high = (max_price - entry_price) / entry_price

        # If minimum price is lower than entry price, calculate loss from it
        if min_price < entry_price:
            loss_from_low = (entry_price - min_price) / entry_price
        else:
            loss_from_low = 0.0  # No potential loss from low prices

        # Take the maximum of two possible losses
        max_potential_loss = max(loss_from_high, loss_from_low)

        return max(0, max_potential_loss - self.loss_threshold)  # Ensure non-negative weight

    def _trade_prices(self, entry_datetime, exit_datetime):
        return self.ohlcv_df['close'].loc[entry_datetime:exit_datetime].copy()

    def closed_trade_reward(self, timecount) -> float:
        """
        Calculate the reward for a closed trade by combining relative_pnl and the trade weight.
        The weight to adjust the reward based on the trade's risk-adjusted performance.
        """
        #   if current_hold_period NOT empty -> add exit data to object
        if self.current_hold_period.reward:
            self.current_hold_period.exit_datetime = self.__asset.orders.last_order.order_datetime
            self.current_hold_period.exit_price = self.__asset.orders.last_order.price
            # TODO rewrite for multi assets trading (must updates each timestep)
            self.current_hold_period.size = self.__asset.orders.last_order.size

        # Get the trade weight from the trade_weight method
        weight = self.trade_period_weight()

        # Get the PnL of the last closed trade
        relative_pnl = self.pnl(self.__trades.last_trade.profit)  # PnL can be positive or negative

        if relative_pnl >= 0:
            action_reward = relative_pnl * (1 - weight)
        else:
            action_reward = relative_pnl * (1 + weight)

        # hold_actions_length = len(self.current_hold_period.reward) + 1
        self.current_hold_period.reset()

        self.raw_rewards.append(action_reward)

        return action_reward

    def hold_action_reward(self, timecount) -> float:
        #   if current_hold_period empty -> add starting data to object
        if not self.current_hold_period.reward:
            self.current_hold_period.entry_datetime = self.ohlcv_df.index[timecount].to_pydatetime()
            self.current_hold_period.entry_price = self.ohlcv_df.iloc[timecount]['close']
            # TODO rewrite for multi assets trading (must updates each timestep)
            self.current_hold_period.size = self.__asset.orders.last_order.size

        price = self.ohlcv_df.iloc[timecount]['close']
        previous_price = self.ohlcv_df.iloc[timecount - 1]['close']
        action_reward = ((
                                 price - previous_price) * self.__asset.orders.last_order.size) / self.__asset.initial_total_in_cash

        self.current_hold_period.reward.append(action_reward)
        # self.current_hold_period.gamma_reward = self.current_hold_period.gamma_reward * self.gamma + action_reward

        self.raw_rewards.append(action_reward)
        # return action_reward
        return 0.
        # return self.current_hold_period.gamma_reward

    def buy_action_reward(self, timecount) -> float:
        #   if current_wait_period reward NOT empty -> add exit data to object
        if self.current_wait_period.reward:
            self.current_wait_period.exit_datetime = self.__asset.orders.last_order.order_datetime
            self.current_wait_period.exit_price = self.__asset.orders.last_order.price
        else:
            return 1e-6

        start_price = self.current_wait_period.entry_price
        end_price = self.current_wait_period.exit_price
        #   wait_reward append to raw_rewards list with self._wait_action_reward
        _ = self._wait_action_reward(timecount)

        wait_period_pnl = ((start_price - end_price) * self.current_wait_period.size) / self.__asset.initial_total_in_cash

        weight = self.wait_period_weight()

        if abs(wait_period_pnl) > self.loss_threshold:
            if wait_period_pnl >= 0:
                action_reward = wait_period_pnl * (1 - weight)
            else:
                action_reward = wait_period_pnl * (1 + weight)
        else:
            action_reward = wait_period_pnl * 1.1

        # wait_actions_length = len(self.current_wait_period.reward)

        self.current_wait_period.reset()

        return action_reward

    def _wait_action_reward(self,
                            timecount: int,
                            momentum_threshold: float = 0.87,
                            perc_threshold: float = 0.0087) -> float:
        """
        Wait_action_reward function, for calculate wait action reward
        Args:
            timecount (int):                timecount
            momentum_threshold (float):     momentum threshold
            perc_threshold (float):         percentage threshold

        Returns:
            float
        """
        #   if current_wait_period reward empty -> add starting data to object

        if not self.current_wait_period.reward:
            self.current_wait_period.entry_datetime = self.ohlcv_df.index[timecount].to_pydatetime()
            self.current_wait_period.entry_price = self.ohlcv_df.iloc[timecount]['close']
            # TODO rewrite for multi assets trading (must updates each timestep)
            self.current_wait_period.size = self.size(self.ohlcv_df.iloc[timecount]['close'], self.__asset.target.cash)

        # Get precomputed TA-Lib values
        # atr = self.ohlcv_df.iloc[timecount]['atr14']
        # momentum = self.ohlcv_df.iloc[timecount]['momentum14']
        #
        # # Calculate trend conditions
        # price_volatility = atr / self.ohlcv_df.iloc[timecount]['close']
        # flat_market = (price_volatility < perc_threshold) & (
        #         abs(momentum) < momentum_threshold)  # 0.87% momentum threshold

        size = self.current_wait_period.size
        price = self.ohlcv_df.iloc[timecount]['close']
        previous_price = self.ohlcv_df.iloc[timecount - 1]['close']
        # if flat_market:
        #     # action_reward = 0.105 * size * (perc_threshold - price_volatility)
        #     action_reward = abs(((previous_price - price) * size) / self.__asset.initial_total_in_cash)
        # else:
        #     action_reward = ((previous_price - price) * size) / self.__asset.initial_total_in_cash
        action_reward = ((previous_price - price) * size) / self.__asset.initial_total_in_cash

        self.current_wait_period.reward.append(action_reward)
        # self.current_wait_period.gamma_reward = self.current_wait_period.gamma_reward * self.gamma + action_reward

        self.raw_rewards.append(action_reward)
        return action_reward
        # return self.current_wait_period.gamma_reward

    def wait_action_reward(self,
                           timecount: int,
                           momentum_threshold: float = 0.87,
                           perc_threshold: float = 0.0087) -> float:
        """
        Wrapper for _wait_action_reward function
        Args:
            timecount (int):                timecount
            momentum_threshold (float):     momentum threshold
            perc_threshold (float):         percentage threshold

        Returns:
            float
        """
        action_reward = self._wait_action_reward(timecount, momentum_threshold, perc_threshold)
        # return action_reward
        return 0.

    def reset(self, ohlcv_df):
        # self._init_lib()
        #
        # self.ohlcv_df = ohlcv_df.copy()
        # # Precompute TA-Lib indicators
        # # Calculate ATR (14-period)
        # self.ohlcv_df['atr14'] = self.talib.ATR(
        #     self.ohlcv_df['high'],
        #     self.ohlcv_df['low'],
        #     self.ohlcv_df['close'],
        #     timeperiod=14
        # )
        #
        # # Calculate Momentum (14-period ROC)
        # self.ohlcv_df['momentum14'] = self.talib.ROC(self.ohlcv_df['close'], timeperiod=14)
        #
        # # Fill NaN values created by indicators
        # self.ohlcv_df.fillna(method='bfill', inplace=True)
        self.current_wait_period.reset()
        self.current_hold_period.reset()
        self.raw_rewards.clear()


class Rewards(RewardsBase):
    pass


if __name__ == "__main__":
    pass
