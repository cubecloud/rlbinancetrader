import copy
import numpy as np
import pandas as pd
from dbbinance.fetcher import Constants
from dbbinance.fetcher.datautils import get_timeframe_bins
from rllab.labtools import get_lookback_timeframes
from typing import Union

__version__ = 0.006


def prepare_vwap(ohlcv_df, rolling_window):
    """
    Calculate the Volume Weighted Average Price (VWAP) for each row in the given ohlcv_df DataFrame.

    VWAP is calculated as the sum of the product of the average price and the volume for each row,
    divided by the sum of the volume for each row.

    Parameters
    ----------
    ohlcv_df : pandas.DataFrame
        A DataFrame containing the OHLCV data.
    rolling_window : int
        The number of rows to use for the rolling calculation of the VWAP.

    Returns
    -------
    vwap : pandas.Series
        A Series containing the VWAP for each row in the given ohlcv_df DataFrame.
    """
    return (((ohlcv_df['high'] + ohlcv_df['low'] + ohlcv_df['close']) / 3) * (ohlcv_df['volume']).rolling(
        window=rolling_window, min_periods=1).sum()) / (
        ohlcv_df['volume'].rolling(window=rolling_window, min_periods=1).sum())


def prepare_ret_obs(ohlcv_df, rolling_window):
    """
    Prepare OHLCV observation data for BinanceEnvCash.

    This includes:

        - Calculate the Volume Weighted Average Price (VWAP) for each row
        - Calculate the return of the VWAP
        - Convert the OHLCV data to log scale
        - Calculate some additional features, such as the returns of the high and low prices,
          and the high-low spread

    Returns
    -------
    pandas.DataFrame
        The prepared OHLCV observation data

    Notes
    -----
    This function is used internally to prepare the observation data.
    """
    obs_ohlcv_df = ohlcv_df.copy(deep=True)
    obs_ohlcv_df['vwap'] = prepare_vwap(ohlcv_df=obs_ohlcv_df, rolling_window=rolling_window)
    obs_ohlcv_df = np.log(obs_ohlcv_df[['open', 'high', 'low', 'close', 'vwap']])
    obs_ohlcv_df['vwap_ret'] = obs_ohlcv_df.vwap.pct_change()
    obs_ohlcv_df['x1'] = obs_ohlcv_df.close.pct_change()
    obs_ohlcv_df['x2'] = obs_ohlcv_df.high.pct_change()
    obs_ohlcv_df['x3'] = obs_ohlcv_df.low.pct_change()
    obs_ohlcv_df['x4'] = (obs_ohlcv_df['high'] - obs_ohlcv_df['close']) / obs_ohlcv_df['close']
    obs_ohlcv_df['x5'] = (obs_ohlcv_df['close'] - obs_ohlcv_df['low']) / obs_ohlcv_df['close']
    return obs_ohlcv_df[['x1', 'x2', 'x3', 'x4', 'x5', 'vwap_ret']]


def calculate_volatility(ohlcv_df, window: Union[int, str] = '24h', method='parkinson',
                         annualize=True, timeframe='15m'):
    """
    Calculate volatility for any timeframe (e.g., 15-minute crypto data).

    Parameters:
        ohlcv_df (pd.DataFrame): OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        window (int, str): Rolling window period (default: 30)
        method (str): 'close_to_close' or 'parkinson' (default: 'close_to_close')
        annualize (bool): Annualize volatility (default: False)
        timeframe (str): Time frequency (e.g., '15m', '1h', 'D')

    Returns:
        pd.DataFrame: Original DataFrame with added 'volatility' column
    """
    df = ohlcv_df.copy()

    window_timeframes = get_lookback_timeframes(window, timeframe)

    # Calculate volatility
    if method == 'close_to_close':
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_returns'].rolling(window=window_timeframes).std()
    elif method == 'parkinson':
        df['hl_ratio'] = np.log(df['high'] / df['low'])
        df['hl_squared'] = df['hl_ratio'] ** 2
        df['volatility'] = np.sqrt(
            (1 / (4 * np.log(2) * window_timeframes)) * df['hl_squared'].rolling(window=window_timeframes).sum())
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Annualized volatility for crypto (24/7 markets)
    if annualize:
        td_minutes = get_timeframe_bins(timeframe)
        periods_per_year = 525600 / td_minutes  # 365 days * 24h * 60min / interval
        annualization_factor = np.sqrt(periods_per_year)
        df['volatility'] *= annualization_factor

    return df


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Generate sample 15-minute OHLCV data
    np.random.seed(42)
    length = 10000
    dates = pd.date_range(start="2023-01-01", periods=10000, freq='15min')
    df = pd.DataFrame({
        'open': np.cumprod(1 + np.random.normal(0.0001, 0.01, length)) + 100,
        'high': np.random.uniform(1.001, 1.02, length) * 100,
        'low': np.random.uniform(0.98, 0.999, length) * 100,
        'close': np.cumprod(1 + np.random.normal(0.0001, 0.01, length)) + 100,
        'volume': np.random.randint(1000, 5000, length)
    }, index=dates)

    window = '1d'
    # Calculate annualized Parkinson volatility (15m timeframe)
    vol_df = calculate_volatility(
        df,
        window=window,  # ~1 day of 15m bars (100 periods)
        method='parkinson',
        annualize=True,
        timeframe='15m'
    )

    print(
        f"Parkinson. Latest Annualized Volatility (rolling_window 24h, timeframe 15m): {vol_df['volatility'].iloc[-1]:.2%}")

    vol_df[['close', 'volatility']].plot(subplots=True, figsize=(12, 6))
    plt.show()

    vol_df = calculate_volatility(
        df,
        window=window,  # ~1 day of 15m bars (100 periods)
        method='close_to_close',
        annualize=True,
        timeframe='15m'
    )
    print(
        f"Close-to-Close. Latest Annualized Volatility (rolling_window 24h, timeframe 15m): {vol_df['volatility'].iloc[-1]:.2%}")
    vol_df[['close', 'volatility']].plot(subplots=True, figsize=(12, 6))
    plt.show()
