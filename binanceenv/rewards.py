from abc import ABC, abstractmethod

from binanceenv.orderbook import TradesBook, Asset, Trade
from typing import Optional, List, Union
import numpy as np
import pandas as pd

__version__ = 0.032


class RollingRewardNormalizer:
    """Class for normalizing rewards using a running average and standard deviation."""

    def __init__(self, epsilon: float = 1e-8, dtype=np.float64) -> None:
        """
        Initializes the reward normalizer object.

        Args:
            epsilon: A small number to ensure numerical stability.
        """
        self.dtype = dtype
        self.running_mean: dtype = 0.0  # Running mean of rewards.
        self.running_var: dtype = 1.0  # Running variance of rewards, initialized to be positive.
        self.epsilon = epsilon
        self.count = self.epsilon  # Count of observed rewards, starts with epsilon to avoid division by zero.

    def __call__(self, reward) -> Union[np.float64, float]:
        """
        Returns normalized value

        Args:
            reward: The new reward value.

        Returns:
            np.float64 or float: The normalized reward.
        """

        self.update(reward)
        return self.normalize(reward)

    def update(self, reward) -> None:
        """
        Updates the running mean and variance based on the new reward.

        Args:
            reward: The new reward value.
        """
        self.count += 1
        old_mean = self.running_mean  # Save the old mean before updating
        delta = reward - old_mean  # Calculate delta from the old mean
        self.running_mean += delta / self.count  # Update the mean
        self.running_var += delta * (reward - old_mean)  # Update the variance correctly

    def normalize(self, reward) -> Union[np.float64, float]:
        """
        Normalizes the given reward based on current mean and variance.

        Args:
            reward: The reward to be normalized.

        Returns:
            float: The normalized reward.
        """
        if self.running_var > 1e-8:
            std_dev = np.sqrt(self.running_var / (self.count - 1))  # Standard deviation.
            return (reward - self.running_mean) / std_dev
        else:
            return reward

    def reset(self):
        self.running_mean = 0.0  # Running mean of rewards.
        self.running_var = 1.0  # Running variance of rewards, initialized to be positive.
        self.count = self.epsilon  # Count of observed rewards, starts with epsilon to avoid division by zero.


def get_sigmoid_results(max_len: int = 3000, normalization_window: int = 96) -> np.array:
    """
    calculate sigmoid results with parameters to calculate sigmoid values.

    Args:
        max_len (int): The maximum length of x values.
        normalization_window (int, optional): Normalization window for scaling x. Defaults to 96.
    Returns:
        np.ndarray
    """

    x_values = np.arange(0, max_len + 1, dtype=np.float64)
    sigmoid_results = (1 / (1 + np.exp(-x_values / normalization_window)))
    return sigmoid_results


class AnyPeriod:

    def __init__(self, normalization_window: int):
        self.entry_datetime = None
        self.exit_datetime = None
        self.entry_price = None
        self.exit_price = None
        self.size: float = 0.0
        self.reward: List[float] = []
        self.sigmoid_results: np.array = get_sigmoid_results(max_len=10000,
                                                             normalization_window=normalization_window)
        self.gamma_reward: float = 0.0
        self.theta = 0.9

    @property
    def normalized_steps(self) -> float:
        return self.sigmoid_results[self.period_len]

    @property
    def period_len(self):
        return self.reward.__len__()

    def __len__(self):
        return self.reward.__len__()

    def calc_timecounted_gamma_reward(self, reward: float, gamma: float = 0.92, end_period: bool = False) -> float:
        self.gamma_reward = self.gamma_reward * gamma + reward
        # accumulating reward for 4 steps with current period
        if end_period:
            return reward
        else:
            return self.gamma_reward
        # if self.period_len % 4 == 0:
        #     # return self.gamma_reward * self.theta ** (self.period_len / timeframes_24h)
        #     return self.gamma_reward
        # else:
        #     return 1e-7
        # # return self.gamma_reward

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


class TerminalConditions(dict):
    def __getattr__(self, item):
        return self[item]


class RewardsBase(ABC):
    def __init__(self,
                 asset: Asset,
                 gamma=0.93,
                 loss_threshold: float = 0.0087,
                 profit_threshold: float = 0.011,
                 normalization_window: int = 12,
                 use_final_reward: bool = False,
                 timeframes_24h: int = 96):
        self.__asset = asset
        self.gamma = gamma
        self.loss_threshold = loss_threshold
        self.profit_threshold = profit_threshold
        self.window_size = normalization_window
        self.use_final_reward = use_final_reward
        self.timeframes_24 = timeframes_24h
        self.__trades: TradesBook = asset.trades
        self.ohlcv_df: Optional[pd.DataFrame] = None
        self.raw_rewards: List[float] = []
        self.reward_constant = 1e-6
        self.rrn_obj = RollingRewardNormalizer()
        self.term_cond = TerminalConditions()
        self.term_cond.stop = False
        self.term_cond.win_rate = False
        self.zero_trade_reward: float = 0.0
        # Local initialization of TALib, for multiprocessing
        # self.talib = None
        self.current_wait_period = WaitPeriod(self.window_size)
        self.current_hold_period = HoldPeriod(self.window_size)

    @property
    def episode_relative_pnl(self) -> float:
        return self.pnl(self.__trades.profit)

    # def _init_lib(self):
    #     # Called only once per instance, in the child process
    #     if self.talib is None:
    #         import talib
    #         self.talib = talib

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

    def get_normalized_reward(self):
        return self.rrn_obj(self.raw_rewards[-1])

    def get_rolling_average_reward(self, window_size):
        if len(self.raw_rewards) <= 2:
            # Not enough data points yet; return the last raw reward
            return self.raw_rewards[-1]

        # Take the tail of the rewards with size equal to window_size
        # Calculate mean using numpy
        return np.mean(self.raw_rewards[-window_size:])

    def is_terminate_condition(self, pnl_stop):
        #   start working from window_size (usually qty of current timeframe for 24h)
        if len(self.raw_rewards) > self.window_size:
            self.term_cond.stop = self.episode_relative_pnl < pnl_stop
        return self.term_cond.stop or self.term_cond.win_rate

    def final_reward(self, buy_and_hold_pnl) -> float:
        _final_reward = 0.0

        if self.use_final_reward:

            # self.term_cond.win_rate = (self.__asset.trades.trades_qty > 2 and self.__asset.trades.win_rate < 60)
            """
            if stop signal cos of pnl_stop -> return small penalty as final reward
            """
            if self.term_cond.stop:
                _final_reward += -self.reward_constant * 2
                self.raw_rewards[-1] = _final_reward
                # return self.get_normalized_reward()
                return self.raw_rewards[-1]

            """
            if no trades and B&H PNL <=0 we return big penalty incremented overtime in each environment locally 
            """
            if len(self.raw_rewards) > self.window_size and self.__trades.trades_qty < 2:
                # increment zero_trade_reward each time then we don't have any trade
                self.zero_trade_reward += -self.reward_constant
                _final_reward += self.zero_trade_reward

            # if self.__asset.trades.trades_qty > 2:
            #     if self.__asset.trades.win_rate > 90:
            #         _final_reward += self.reward_constant * 8
            #     elif self.__asset.trades.win_rate > 80:
            #         _final_reward += self.reward_constant * 4
            # elif self.__asset.trades.win_rate > 70:
            #     _final_reward += self.reward_constant * 2
            # else:
            #     _final_reward += -self.reward_constant

            if self.__asset.trades.trades_qty > 2:
                # positive reward for trading more
                _final_reward += self.reward_constant
                if self.__asset.trades.win_rate < 60:
                    _final_reward += -self.reward_constant
                elif self.__asset.trades.win_rate > 90:
                    _final_reward += self.reward_constant * 16
                elif self.__asset.trades.win_rate > 80:
                    _final_reward += self.reward_constant * 8
                elif self.__asset.trades.win_rate > 70:
                    _final_reward += self.reward_constant * 4
                elif self.__asset.trades.win_rate > 60:
                    _final_reward += self.reward_constant * 2
            #     elif self.__asset.trades.win_rate >= 60:
            #         _final_reward += self.reward_constant * 2
            # else:
            #     _final_reward += -self.reward_constant * 4

            if self.episode_relative_pnl > .0:
                _final_reward += self.reward_constant
            else:
                _final_reward += -self.reward_constant

            #     else:
            #         _final_reward += -self.reward_constant
            # else:
            #     _final_reward += -self.reward_constant
            #     if self.episode_relative_pnl > buy_and_hold_pnl:
            #         _final_reward += self.reward_constant * 4
            #     else:
            #         _final_reward += -self.reward_constant

            self.raw_rewards[-1] += _final_reward
        # return self.get_normalized_reward()
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

        # return max(0.0, increased_drawdown - self.loss_threshold)
        return max(0.0, increased_drawdown)

    def wait_potential_loss(self, prices: pd.Series) -> float:
        """Calculates the potential maximum loss relative to the highest achieved price.

        Args:
            prices (pd.Series): Series of prices for analysis.

        Returns:
            float: weight
         """
        max_price = prices.max()
        potential_loss = (max_price - prices[-1]) / max_price  # Potential loss considering current price as highest
        # return max(0, potential_loss - self.loss_threshold)  # Ensure non-negative weight
        return max(0, potential_loss)  # Ensure non-negative weight

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

        return max(0, max_potential_loss)  # Ensure non-negative weight

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

        # price = self.ohlcv_df.iloc[timecount]['close'] + 1e-8
        # previous_price = self.ohlcv_df.iloc[timecount - 1]['close']
        # action_reward = ((
        #                          price - previous_price) * self.__asset.orders.last_order.size) / self.__asset.initial_total_in_cash

        # Get the trade weight from the trade_weight method
        # weight = self.trade_period_weight()

        # Get the PnL of the last closed trade
        # relative_pnl = self.pnl(self.__trades.last_trade.profit)  # PnL can be positive or negative
        action_reward = self.pnl(self.__trades.last_trade.profit)  # PnL can be positive or negative

        # if relative_pnl >= 0:
        #     action_reward = relative_pnl * (1 - weight)
        # else:
        #     action_reward = relative_pnl * (1 + weight)

        self.raw_rewards.append(action_reward)
        timecounted_close_reward = self.current_hold_period.calc_timecounted_gamma_reward(action_reward,
                                                                                          gamma=self.gamma,
                                                                                          end_period=True)
        self.current_hold_period.reset()
        # assert timecounted_close_reward != 0., 'Error: "Close" reward equal zero'
        return timecounted_close_reward

        # return action_reward
        # return self.get_normalized_reward()

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
        self.raw_rewards.append(action_reward)

        # return action_reward
        # return 0.0
        # return self.get_normalized_reward()
        return self.current_hold_period.calc_timecounted_gamma_reward(action_reward, gamma=self.gamma)

    def buy_action_reward(self, timecount) -> float:
        #   if current_wait_period reward NOT empty -> add exit data to object
        if self.current_wait_period.reward:
            self.current_wait_period.exit_datetime = self.__asset.orders.last_order.order_datetime
            self.current_wait_period.exit_price = self.__asset.orders.last_order.price
            # self.current_wait_period.size = self.size(self.ohlcv_df.iloc[timecount]['close'], self.__asset.target.cash)
            previous_price = self.current_wait_period.entry_price
            price = self.current_wait_period.exit_price
            # action_reward = -((
            #                          start_price - end_price) * self.current_wait_period.size) / self.__asset.initial_total_in_cash
            # action_reward = ((
            #                          start_price - end_price) * self.current_wait_period.size) / self.__asset.initial_total_in_cash
        else:
            # penalize buy action w/o wait period
            self.current_wait_period.size = self.size(self.ohlcv_df.iloc[timecount]['close'], self.__asset.target.cash)
            price = self.ohlcv_df.iloc[timecount]['close']
            previous_price = self.ohlcv_df.iloc[timecount - 1]['close'] + 1e-8
        # price = self.ohlcv_df.iloc[timecount]['close']
        # previous_price = self.ohlcv_df.iloc[timecount - 1]['close'] + 1e-8
        # # action_reward = -((previous_price - price) * size) / self.__asset.initial_total_in_cashS

        size = self.current_wait_period.size
        action_reward = ((previous_price - price) * size) / self.__asset.initial_total_in_cash
        #   wait_reward append to raw_rewards list with self._wait_action_reward
        # _ = self._wait_action_reward(timecount)

        # weight = self.wait_period_weight()

        # if wait period less than zero -> better cos we have local minimum for going up
        # if wait_period_pnl < 0:
        #     action_reward = wait_period_pnl * (1 + weight)  # increase negative reward by weight
        # else:
        #     action_reward = wait_period_pnl * (1 - weight)  # decrease positive reward by weight

        # if wait_period_pnl > -self.loss_threshold:
        #     if wait_period_pnl >= 0:
        #         action_reward = wait_period_pnl * (1 - weight)
        #     else:
        #         action_reward = wait_period_pnl * (1 + weight)
        # else:
        #     action_reward = wait_period_pnl * 1.1

        # wait_actions_length = len(self.current_wait_period.reward)
        self.current_wait_period.reward.append(action_reward)
        self.raw_rewards.append(action_reward)

        timecounted_buy_reward = self.current_wait_period.calc_timecounted_gamma_reward(action_reward,
                                                                                        gamma=self.gamma,
                                                                                        end_period=True)
        self.current_wait_period.reset()
        # assert timecounted_buy_reward != 0., 'Error: "BUY" reward equal zero'
        return timecounted_buy_reward
        # return action_reward
        # return self.get_normalized_reward()

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
        # action_reward = -((previous_price - price) * size) / self.__asset.initial_total_in_cash
        action_reward = ((previous_price - price) * size) / self.__asset.initial_total_in_cash

        self.current_wait_period.reward.append(action_reward)
        self.raw_rewards.append(action_reward)

        # return action_reward
        # return self.get_normalized_reward()
        return self.current_wait_period.calc_timecounted_gamma_reward(action_reward, gamma=self.gamma)

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
        return action_reward
        # return 0.0

    def reset(self, ohlcv_df):
        # self._init_lib()
        #
        self.ohlcv_df = ohlcv_df.copy()
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
        self.rrn_obj.reset()
        self.term_cond.stop = False
        self.term_cond.win_rate = False


class Rewards(RewardsBase):
    pass


if __name__ == "__main__":
    pass
