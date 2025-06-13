import numpy as np
import pandas as pd
from binanceenv.orderbook import TradesBook

from typing import Callable, Union, Dict, Tuple

__version__ = 0.0003


def get_trade_metrics(trades_book, df: pd.DataFrame) -> Dict:
    """Calculate comprehensive trading metrics"""
    metrics = {
        'Start': pd.NaT,
        'End': pd.NaT,
        'Duration': pd.Timedelta(0),
        'Exposure Time [%]': 0.0,
        'Equity Final [$]': df['total'].iloc[-1] if not df.empty else 0.0,
        'Equity Peak [$]': df['total'].max() if not df.empty else 0.0,
        'Return [%]': 0.0,
        'Buy & Hold Return [%]': 0.0,
        'Return (Ann.) [%]': 0.0,
        'Volatility (Ann.) [%]': 0.0,
        'Sharpe Ratio': 0.0,
        'Sortino Ratio': 0.0,
        'Calmar Ratio': 0.0,
        'Max. Drawdown [%]': 0.0,
        'Avg. Drawdown [%]': 0.0,
        'Max. Drawdown Duration': pd.Timedelta(0),
        'Avg. Drawdown Duration': pd.Timedelta(0),
        '# Trades': 0,
        'Win Rate [%]': 0.0,
        'Profit Rate [%]': 0.0,
        'Best Trade [%]': 0.0,
        'Worst Trade [%]': 0.0,
        'Avg. Trade [%]': 0.0,
        'Max. Trade Duration': pd.Timedelta(0),
        'Avg. Trade Duration': pd.Timedelta(0),
        'Profit Factor': 0.0,
        'Expectancy [%]': 0.0,
        'SQN': 0.0
    }

    # Returns
    initial_equity = df['total'].iloc[0]
    final_equity = metrics['Equity Final [$]']
    metrics['Return [%]'] = (final_equity / initial_equity - 1) * 100

    # Buy & Hold return
    initial_price = df['close'].iloc[0]
    final_price = df['close'].iloc[-1]
    metrics['Buy & Hold Return [%]'] = (final_price / initial_price - 1) * 100

    # Annualized return
    timeframe_minutes = (df.index[-1] - df.index[0]).total_seconds() / 60 / len(df)
    periods_per_year = (365 * 24 * 60) / timeframe_minutes
    metrics['Return (Ann.) [%]'] = ((final_equity / initial_equity) ** (periods_per_year / len(df)) - 1) * 100

    # Volatility
    returns = np.log(df['total'] / df['total'].shift(1)).dropna()
    metrics['Volatility (Ann.) [%]'] = returns.std() * np.sqrt(periods_per_year) * 100

    # Drawdown calculations
    df['running_max'] = df['total'].cummax()
    df['drawdown'] = (df['total'] - df['running_max']) / df['running_max']
    drawdowns = df[df['drawdown'] < 0]['drawdown']

    if not drawdowns.empty:
        metrics.update({
            'Max. Drawdown [%]': drawdowns.min() * 100,
            'Avg. Drawdown [%]': drawdowns.mean() * 100,
            'Max. Drawdown Duration': drawdowns.idxmin() - df['running_max'].idxmax(),
            'Avg. Drawdown Duration': pd.Timedelta(seconds=drawdowns.count() * timeframe_minutes * 60 / len(df))
        })

    # Risk-adjusted ratios
    risk_free_rate = 0.0
    sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(periods_per_year)
    downside_returns = returns[returns < 0]
    sortino_ratio = (returns.mean() - risk_free_rate) / downside_returns.std() * np.sqrt(
        periods_per_year) if downside_returns.std() != 0 else 0.0

    # Calmar Ratio calculation
    calmar_ratio = 0.0
    if metrics['Max. Drawdown [%]'] != 0:
        annual_return_decimal = metrics['Return (Ann.) [%]'] / 100
        max_dd_decimal = abs(metrics['Max. Drawdown [%]'] / 100)
        calmar_ratio = annual_return_decimal / max_dd_decimal

    metrics.update({
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio
    })

    if not trades_book.book:
        metrics['# Trades'] = 0
        return metrics

    # Time metrics
    all_trades = [t for t in trades_book.book if t.closed]
    if not all_trades:
        return metrics

    # start = min(t.entry_datetime for t in all_trades)
    # end = max(t.exit_datetime for t in all_trades)
    start = df.index[0]
    end = df.index[-1]
    total_duration = end - start

    metrics.update({
        'Start': start,
        'End': end,
        'Duration': total_duration
    })

    # Exposure time
    exposure_time = sum((t.exit_datetime - t.entry_datetime for t in all_trades), pd.Timedelta(0))
    metrics['Exposure Time [%]'] = (exposure_time.total_seconds() /
                                    total_duration.total_seconds()) * 100

    # Trade statistics
    trade_returns = []
    trade_durations = []
    wins = []
    losses = []
    volume = 0.0

    for trade in all_trades:
        entry = trade.entry_price * trade.size
        exit = trade.exit_price * trade.size
        trade_return = (exit - entry) / entry * 100
        duration = trade.exit_datetime - trade.entry_datetime

        trade_returns.append(trade_return)
        trade_durations.append(duration)
        volume += abs(trade.profit)

        if trade_return > 0:
            wins.append(trade_return)
        else:
            losses.append(trade_return)

    metrics['# Trades'] = len(all_trades)

    if trade_returns:
        metrics.update({
            'Win Rate [%]': len(wins) / len(trade_returns) * 100,
            'Profit Rate [%]': trades_book.profit / volume * 100,
            'Best Trade [%]': max(trade_returns) if trade_returns else 0.0,
            'Worst Trade [%]': min(trade_returns) if trade_returns else 0.0,
            'Avg. Trade [%]': np.mean(trade_returns),
            'Max. Trade Duration': max(trade_durations),
            'Avg. Trade Duration': sum(trade_durations, pd.Timedelta(0)) / len(trade_durations),
            'Profit Factor': abs(sum(wins)) / abs(sum(losses)) if losses else 0.0,
            'Expectancy [%]': (np.mean(wins) * len(wins) - np.mean(losses) * len(losses)) / len(
                trade_returns) if trade_returns else 0.0,
            'SQN': (np.mean(trade_returns) / np.std(trade_returns)) * np.sqrt(len(trade_returns)) if len(
                trade_returns) > 1 else 0.0
        })

    return metrics


def get_trade_metrics_1(trades: TradesBook) -> dict:
    """Extract metrics from TradesBook"""
    if not trades.trades_qty:
        return {
            'win_rate': 0.0,
            'total_profit': 0.0,
            'active_period': (None, None),
            'num_trades': 0,
            'avg_trade_duration': pd.Timedelta(0),
            'profit_factor': 0.0
        }

    # Win Rate
    win_rate = trades.win_rate * 100  # Convert to percentage

    # Total Profit
    total_profit = trades.profit

    # Active Period
    all_trades = [t for t in trades.book if t.closed]
    if all_trades:
        start = min(t.orders.open_order.order_datetime for t in all_trades)
        end = max(t.orders.close_order.order_datetime for t in all_trades)
    else:
        start = end = None

    # Trade duration statistics
    durations = [
        t.orders.close_order.order_datetime - t.orders.open_order.order_datetime
        for t in all_trades
    ]
    avg_duration = pd.Timedelta(0) if not durations else sum(durations, pd.Timedelta(0)) / len(durations)

    # Profit Factor
    gross_profit = sum(t.profit for t in all_trades if t.profit > 0)
    gross_loss = abs(sum(t.profit for t in all_trades if t.profit < 0))
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0.0

    return {
        'win_rate': win_rate,
        'total_profit': total_profit,
        'active_period': (start, end),
        'num_trades': len(all_trades),
        'avg_trade_duration': avg_duration,
        'profit_factor': profit_factor
    }


def calculate_sharpe_ratio(portfolio_values: pd.Series,
                           timeframe_minutes: int,
                           risk_free_rate=0.005) -> float:
    """
    Calculate realistic annualized Sharpe Ratio for crypto.

    Args:
        portfolio_values: Series of portfolio values
        timeframe_minutes: Timeframe resolution in minutes
        risk_free_rate: Annualized risk-free rate (default 0%)

    Returns:
        Annualized Sharpe Ratio (realistic range 0-3)
    """
    if len(portfolio_values) < 2:
        return 0.0

    # Calculate log returns
    log_returns = np.log(portfolio_values / portfolio_values.shift(1)).dropna()

    if log_returns.std() == 0:
        return 0.0

    # Convert risk-free rate to timeframe rate
    annual_minutes = 365 * 24 * 60
    rf_per_period = (1 + risk_free_rate) ** (timeframe_minutes / annual_minutes) - 1

    # Annualization factor
    annualization = np.sqrt(annual_minutes / timeframe_minutes)

    # Calculate Sharpe Ratio
    excess_returns = log_returns - rf_per_period
    sharpe_ratio = excess_returns.mean() / log_returns.std()
    sharpe_ratio = sharpe_ratio * annualization
    return sharpe_ratio


def calculate_win_rate(df: pd.DataFrame) -> Tuple[int, float]:
    """
    Calculate the win rate of trades.

    Args:
        df: DataFrame containing 'action' and 'pnl' columns.

    Returns:
        float: Win rate as a percentage (0-100).
    """
    # Identify trades (buy/sell pairs)
    trades = []
    in_trade = False
    entry_pnl = 1.0

    for i, row in df.iterrows():
        if row['action'] == 0:  # Buy
            in_trade = True
            entry_pnl = row['pnl']
        elif row['action'] == 1 and in_trade:  # Sell
            trades.append((entry_pnl, row['pnl']))
            in_trade = False

    # Calculate win rate
    if not trades:
        return 0, 0.0

    winning_trades = sum(1 for entry_pnl, exit_pnl in trades if exit_pnl - entry_pnl > 0)
    trades_num = len(trades)
    win_rate = (winning_trades / len(trades)) * 100

    return trades_num, win_rate
