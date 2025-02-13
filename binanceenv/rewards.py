from abc import ABC, abstractmethod
from binanceenv.orderbook import TradesBook, Asset, Trade
from typing import Optional, List
import numpy as np
import pandas as pd

__version__ = 0.009


class WaitPeriod:
    def __init__(self):
        self.entry_datetime = None
        self.exit_datetime = None
        self.entry_price = None
        self.exit_price = None
        self.wait_reward: List[float] = []
        self.size: float = 0.0

    def reset(self):
        self.entry_datetime = None
        self.exit_datetime = None
        self.entry_price = None
        self.exit_price = None
        self.size: float = 0.0
        self.wait_reward.clear()


class RewardsBase(ABC):
    def __init__(self, asset: Asset,
                 loss_threshold: float = 0.0089,
                 profit_threshold: float = 0.011):
        self.__asset = asset
        self.loss_threshold = loss_threshold
        self.profit_threshold = profit_threshold
        self.__trades: TradesBook = asset.trades
        self.ohlcv_df: Optional[pd.DataFrame] = None

        # Local initialization of TALib, for multiprocessing
        self.talib = None
        self.current_wait_period = WaitPeriod()

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

    def final_reward(self, buy_and_hold_pnl):
        # pnl_score = self.pnl_score(buy_and_hold_pnl)
        # win_rate_score = self.win_rate_score()
        # _final_reward = win_rate_score + pnl_score
        pass
        # return _final_reward

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

    def trade_loss_drawdown_weight(self, prices: pd.Series) -> float:
        """Calculates the maximum drawdown relative to the highest achieved price.

        Args:
            prices (pd.Series): Series of prices for analysis.

        Returns:
            float: weight
        """
        max_price = prices.max()
        drawdown = (max_price - prices.min()) / max_price
        return max(0.0, drawdown - self.loss_threshold)  # ensure non-negative weight

    def trade_profit_max_weight(self, prices: pd.Series) -> float:
        """Calculates the maximum profit relative to the lowest achieved price.

        Args:
            prices (pd.Series): Series of prices for analysis.

        Returns:
            float: weight
        """
        min_price = prices.min()
        max_profit = (prices.max() - min_price) / min_price
        return max(0, max_profit - self.profit_threshold)  # Ensure non-negative weight

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

    # @staticmethod
    # def trade_loss_drawdown_weight(prices: pd.Series, loss_threshold: float = 0.089) -> float:
    #     """Calculates the maximum drawdown relative to the highest achieved price.
    #
    #     Args:
    #         prices (pd.Series): Series of prices for analysis.
    #         loss_threshold (float): Loss threshold at which we switch to calculating losses from the entry price.
    #
    #     Returns:
    #         float: weight
    #     """
    #     max_drawdown = 0.0
    #     max_loss = 0.0
    #     entry_price = prices.iloc[0]
    #     high_price = entry_price
    #
    #     for price in prices:
    #         if price > high_price:
    #             high_price = price
    #
    #         current_loss = (price - entry_price) / entry_price
    #         drawdown = (high_price - price) / high_price
    #         if drawdown > max_drawdown and drawdown > -loss_threshold:
    #             max_drawdown = drawdown
    #
    #         if current_loss < max_loss and current_loss < -loss_threshold:
    #             max_loss = current_loss
    #
    #     return max_loss if max_drawdown < max_loss else max_drawdown

    def trade_period_weight(self):
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
        if entry_datetime == exit_datetime:
            return 1e-6

        trade_prices = self._trade_prices(entry_datetime, exit_datetime)

        min_price = trade_prices.min()
        max_price = trade_prices.max()
        entry_price = self.current_wait_period.entry_price

        # Calculate the potential loss considering current price as highest
        potential_loss1 = (max_price - entry_price) / max_price
        potential_loss2 = 1e-6

        # If we have bought in minimum price then calculate potential loss from there
        if min_price < entry_price:
            potential_loss2 = (entry_price - min_price) / min_price

        # Take the maximum of both potential losses
        max_potential_loss = max(potential_loss1, potential_loss2)

        return max(0, max_potential_loss - self.loss_threshold)  # Ensure non-negative weight

    def _trade_prices(self, entry_datetime, exit_datetime):
        return self.ohlcv_df['close'].loc[entry_datetime:exit_datetime].copy()

    def closed_trade_reward(self):
        """
        Calculate the reward for a closed trade by combining relative_pnl and the trade weight.
        The weight to adjust the reward based on the trade's risk-adjusted performance.
        """
        # Get the trade weight from the trade_weight method
        weight = self.trade_period_weight()

        # Get the PnL of the last closed trade
        relative_pnl = self.pnl(self.__trades.last_trade.profit)  # PnL can be positive or negative

        if relative_pnl >= 0:
            reward = relative_pnl * (1 - weight)
        else:
            reward = relative_pnl * (1 + weight)
        return reward

    def buy_action_reward(self):
        if self.current_wait_period.wait_reward:
            self.current_wait_period.exit_datetime = self.__trades.last_trade.entry_datetime
            self.current_wait_period.exit_price = self.__trades.last_trade.entry_price
        else:
            return 1e-6

        weight = self.wait_period_weight()
        relative_pnl = (
                                   self.current_wait_period.entry_price - self.current_wait_period.exit_price) / self.__asset.initial_total_in_cash
        if relative_pnl >= 0:
            reward = relative_pnl * (1 - weight)
        else:
            reward = relative_pnl * (1 + weight)

        self.current_wait_period.reset()
        return reward

    def wait_action_reward(self, timecount, momentum_threshold=0.87, perc_threshold=0.0087):
        if not self.current_wait_period.wait_reward:
            self.current_wait_period.entry_datetime = self.ohlcv_df.index[timecount].to_pydatetime()
            self.current_wait_period.entry_price = self.ohlcv_df.iloc[timecount]['close']
            # TODO rewrite for multiassets trading (must updates each timestep)
            self.current_wait_period.size = self.size(self.ohlcv_df.iloc[timecount]['close'], self.__asset.target.cash)

        # Get precomputed TA-Lib values
        atr = self.ohlcv_df.iloc[timecount]['atr14']
        momentum = self.ohlcv_df.iloc[timecount]['momentum14']

        # Calculate trend conditions
        price_volatility = atr / self.ohlcv_df.iloc[timecount]['close']
        flat_market = (price_volatility < perc_threshold) & (
                abs(momentum) < momentum_threshold)  # 0.87% momentum threshold

        if flat_market:
            # Reward based on volatility suppression
            reward = 0.3 * (perc_threshold - price_volatility)
        else:
            size = self.current_wait_period.size
            reward = ((self.ohlcv_df.iloc[timecount - 1]['close'] - self.ohlcv_df.iloc[timecount]['close']) * size) / (
                self.__asset.initial_total_in_cash)

        self.current_wait_period.wait_reward.append(reward)
        return reward

    def reset(self, ohlcv_df):
        self._init_lib()

        self.ohlcv_df = ohlcv_df.copy()
        # Precompute TA-Lib indicators
        # Calculate ATR (14-period)
        self.ohlcv_df['atr14'] = self.talib.ATR(
            self.ohlcv_df['high'],
            self.ohlcv_df['low'],
            self.ohlcv_df['close'],
            timeperiod=14
        )

        # Calculate Momentum (14-period ROC)
        self.ohlcv_df['momentum14'] = self.talib.ROC(self.ohlcv_df['close'], timeperiod=14)

        # Fill NaN values created by indicators
        self.ohlcv_df.fillna(method='bfill', inplace=True)
        self.current_wait_period.reset()


class Rewards(RewardsBase):
    pass


if __name__ == "__main__":
    pass
