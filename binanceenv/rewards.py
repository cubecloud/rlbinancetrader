from abc import ABC, abstractmethod
from binanceenv.orderbook import TradesBook, Asset, Trade
from typing import Optional, List
import numpy as np
import pandas as pd

__version__ = 0.009


class RewardsBase(ABC):
    def __init__(self, asset: Asset):
        self.__asset = asset
        self.__trades: TradesBook = asset.trades
        self.ohlcv_df: Optional[pd.DataFrame] = None

        self.__wait_reward: List[float] = []
        """ 
        Various weights for calculations 
        """
        # Trade score weights
        self.risk_weight = 0.02
        self.sharpe_ratio_weight = 0.3
        self.drawdown_weight = 0.3

        # Local initialization of TALib, for multiprocessing
        self.talib = None

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
        pnl_score = self.pnl_score(buy_and_hold_pnl)
        win_rate_score = self.win_rate_score()
        _final_reward = win_rate_score + pnl_score
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

    def trade_score(self):
        """
        Calculate the score for a single trade based on risk-adjusted metrics.
        The score is always non-zero and provides meaningful feedback for every trade.
        """
        # Return default score for partially closed trades
        if self.__trades.last_trade.status == 'partly':
            return 1.0  # Default score for partially closed trades

        # Extract trade data
        trade = self.__trades.last_trade
        start_time = trade.entry_datetime
        end_time = trade.exit_datetime
        exit_price = trade.exit_price
        last_price = self.ohlcv_df['close'].loc[end_time]

        # Verify price consistency with tolerance for floating point errors
        if not np.isclose(exit_price, last_price, atol=1e-6):
            raise ValueError(f"Exit price {exit_price} does not match last price {last_price}")

        # Calculate price extremes and drawdown
        trade_prices = self.ohlcv_df['close'].loc[start_time:end_time].copy()
        high_price = trade_prices.max()
        low_price = trade_prices.min()
        max_drawdown_pct = ((high_price - low_price) / high_price) * 100  # Percentage drawdown

        # Calculate risk-adjusted metrics
        if trade_prices.shape[0] > 3:
            # Volatility (standard deviation of returns)
            trade_returns = trade_prices.pct_change().dropna()
            volatility = trade_returns.std() if not trade_returns.empty else 0.0
            volatility = max(volatility, 1e-6)  # Ensure non-zero
            # Sharpe Ratio (mean return divided by volatility)
            if trade_returns.empty:
                sharpe_ratio = 0.0
            else:
                mean_return = trade_returns.mean()
                sharpe_ratio = mean_return / volatility
        else:
            return 1.0

        # Risk score (inverse of drawdown with smoothing)
        risk_score = 1 / (max_drawdown_pct + 1e-6)

        # Combine components with weights
        score = (
                self.risk_weight * risk_score +
                self.sharpe_ratio_weight * sharpe_ratio +
                self.drawdown_weight * max_drawdown_pct
        )

        return abs(score)

    def closed_trade_reward(self):
        """
        Calculate the reward for a closed trade by combining PnL and the trade score.
        The score acts as a weight to adjust the reward based on the trade's risk-adjusted performance.
        """
        # Get the trade score (weight) from the trade_score method
        score = self.trade_score()

        # Ensure the score is non-negative to avoid flipping the sign of the reward
        if score < 0:
            raise ValueError("Trade score must be non-negative.")

        # Get the PnL of the last closed trade
        pnl = self.pnl(self.__trades.last_trade.profit)  # PnL can be positive or negative

        # Calculate the reward by scaling the PnL with the trade score
        # reward = pnl * score
        reward = pnl * score

        return reward

    def buy_action_reward(self):
        reward = sum(self.__wait_reward)
        self.__wait_reward.clear()
        return reward

    def wait_action_reward(self, timecount, momentum_threshold=0.87, perc_threshold=0.005):
        # Get precomputed TA-Lib values
        atr = self.ohlcv_df.iloc[timecount]['atr14']
        momentum = self.ohlcv_df.iloc[timecount]['momentum14']

        # Calculate trend conditions
        price_volatility = atr / self.ohlcv_df.iloc[timecount]['close']
        flat_market = (price_volatility < perc_threshold) & (
                abs(momentum) < momentum_threshold)  # 0.87% momentum threshold

        if flat_market:
            # Reward based on volatility suppression
            reward = 0.015 * (perc_threshold - price_volatility)
        else:
            size = self.size(self.ohlcv_df.iloc[timecount]['close'], self.__asset.target.cash)
            reward = ((self.ohlcv_df.iloc[timecount - 1]['close'] - self.ohlcv_df.iloc[timecount]['close']) * size) / (
                        self.__asset.initial_total_in_cash)

        self.__wait_reward.append(reward)
        return reward

    def reset(self, ohlcv_df):
        self._init_lib()

        # Clear wait_reward buffer
        self.__wait_reward.clear()

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


class Rewards(RewardsBase):
    pass


if __name__ == "__main__":
    pass
