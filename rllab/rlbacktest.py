import pandas as pd
from backtester.core import Back
from backtester import Strategy
from typing import Type

__version__ = 0.004


def rlbacktest(data_df: pd.DataFrame, strategy: Type[Strategy], start_cache: float, commission: float,
               path_filename: str):
    # data_df[['open', 'high', 'low', 'close']] = (data_df[['open', 'high', 'low', 'close']] / 1e6)
    # data_df[['volume', 'amount']] = (data_df[['volume', 'amount']] * 1e6)
    # data_df['volume'] = (data_df['volume'] * 1e6)
    bt = Back(data_df,
              strategy,
              cash=100_000,
              # cash=start_cache,
              commission=commission,
              trade_on_close=True,
              )
    _stats = bt.run(cash=start_cache, commission=commission)
    print(_stats)
    bt.plot(plot_volume=True,
            relative_equity=True,
            open_browser=False,
            filename=path_filename
            )
