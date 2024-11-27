import talib
import numpy as np
import pandas as pd
from datawizard.powertrend import *
from backtesting import Strategy
from backtesting.lib import SignalStrategy, TrailingStrategy


class BaseStrategy(Strategy):
    def _check_params(self, params):
        for k, v in params.items():
            setattr(self, k, v)
        return params


class SimpleSignalStrategy(SignalStrategy):
    def init(self):
        super().init()
        self.set_signal(self.data['Signal'] == 1, self.data['Signal'] == -1)


def SMA(array, n):
    """Simple moving average"""
    return pd.Series(array).rolling(n).mean()


class AgroStrategy(Strategy):
    # price_delta = .09  # 0.4%
    size = .333
    name = 'Agro'
    stop_loss = 0.07
    take_profit = 0.145
    default_buy_size = 0.333
    default_sell_size = 0.333
    monotonic = False
    previous_signal = 0
    monotonic_counter = 0
    monotonic_threshold = 14
    # _power_trend = 0.049
    power_trend = 0.049

    def init(self):
        # Plot signal from model
        self.signal = self.I(lambda x: x, self.data.Signal, name='Signal')
        # power_trend = 0.049
        # Plot trend for inspection
        self.p_trend_df = CalcTrendDF(self.data.df, self.power_trend)()
        self.P_trend_new = self.I(lambda x: x, self.p_trend_df, name=f"P_trend_{self.power_trend}")

    # @property
    # def power_trend(self):
    #     return self._power_trend
    #
    # @power_trend.setter
    # def power_trend(self, _p_trend):
    #     self._power_trend =_p_trend
    #     self.p_trend_df = CalcTrendDF(self.data.df, self.power_trend)()
    #     self.P_trend_new = self.I(lambda x: x, self.p_trend_df, name=f"P_trend_{self.power_trend}")

    def next(self):
        high, low, close = self.data.High, self.data.Low, self.data.Close
        current_time = self.data.index[-1]

        price = self.data.Close[-1]
        current_signal = self.signal[-1]

        if self.previous_signal == current_signal:
            self.monotonic = True
            self.monotonic_counter += 1
        else:
            self.monotonic = False
            self.monotonic_counter = 0
            self.previous_signal = current_signal
            # If our forecast is upwards and we don't already hold a long position
            # place a long order for 20% of available account equity. Vice versa for short.
            # Also set target take-profit and stop-loss prices to be one price_delta
            # away from the current closing price.

            long_sl = price - (price * self.stop_loss)
            long_tp = price + (price * self.take_profit)

            if current_signal == 1:
                if not self.monotonic or self.monotonic_counter == self.monotonic_threshold:
                    self.buy(size=self.default_buy_size, sl=long_sl, tp=long_tp)
            elif current_signal == -1:
                if not self.monotonic or self.monotonic_counter == self.monotonic_threshold:
                    for trade in self.trades:
                        if trade.is_long:
                            trade.close(portion=self.default_buy_size)
                    # self.position.close(portion=self.default_buy_size)

        # Additionally, set aggressive stop-loss on trades that have been open
        # for more than two days
        for trade in self.trades:
            if current_time - trade.entry_time > pd.Timedelta('7 days'):
                if trade.is_long:
                    trade.sl = max(trade.sl, low)
                    # trade.tp = max(trade.tp, high)


class TestStrategy(SignalStrategy,
                   # TrailingStrategy
                   ):
    name = 'Test'
    default_buy_size = 0.333
    monotonic = False
    previous_signal = 0
    monotonic_counter = 0
    monotonic_normal_len = 24
    size = 1.0
    stop_loss = 0.09
    take_profit = 0.145

    def init(self):
        super().init()
        self.signal = self.I(lambda x: x, self.data.Signal, name='Signal')
        # self.ma10 = self.I(SMA, self.data.Close, 10)
        self.MACD, self.MACDsignal, self.MACDhist = self.I(talib.MACD, self.data.Close, 12, 26, 9)
        self.MOM = self.I(talib.MOM, self.data.Close, 14)
        power_trend = 0.050
        p_trend_df = CalcTrendDF(self.data.df, power_trend)()
        self.P_trend_new = self.I(lambda x: x, p_trend_df, name=f"P_trend_{power_trend}")
        self.B_S = self.I(get_buy_sell_markers, p_trend_df, 3, name="B_S")
        self.entry_size = self.signal * self.default_buy_size
        self.set_signal(entry_size=self.entry_size)
        # self.set_trailing_sl(2)

    def next(self):
        price = self.data.Close[-1]
        current_signal = self.signal[-1]
        if self.previous_signal == current_signal:
            self.monotonic = True
            self.monotonic_counter += 1
        else:
            self.monotonic = False
            self.monotonic_counter = 0
            self.previous_signal = current_signal
            # self.set_signal(entry_size=self.entry_size)

            long_sl = price - (price * self.stop_loss)
            long_tp = price + (price * self.take_profit)

            if self.position.size < 1.0 - self.default_buy_size \
                    and current_signal == 1.0:
                if not self.monotonic:
                    self.buy(size=self.default_buy_size, sl=long_sl, tp=long_tp)
                    # self.set_signal(entry_size=self.entry_size)
                elif self.monotonic and self.monotonic_counter > self.monotonic_normal_len:
                    self.buy(size=self.default_buy_size, sl=long_sl, tp=long_tp)
                    # self.set_signal(entry_size=self.entry_size)
            elif self.position.is_long and current_signal == -1.0:
                if not self.monotonic:
                    self.position.close()
                elif self.monotonic and self.monotonic_counter > self.monotonic_normal_len:
                    self.position.close()

    def __repr__(self):
        return self.name


class TestLSStrategy(SignalStrategy,
                     TrailingStrategy):
    name = 'TestLS'
    # stop_loss = 0.02
    # take_profit = 0.05
    default_buy_size = 0.333
    default_sell_size = 0.333
    monotonic = False
    previous_signal = 0
    monotonic_counter = 0
    monotonic_normal_len = 14
    size = 0.333
    stop_loss = 0.09
    take_profit = 0.12

    def init(self):
        self.signal = self.I(lambda x: x, self.data.Signal, name='Signal')
        entry_size = self.signal * self.default_buy_size
        self.set_signal(entry_size=entry_size)
        # self.set_trailing_sl(2)

    def next(self):
        price = self.data.Close[-1]
        # super().next()
        current_signal = self.signal[-1]
        if self.previous_signal == current_signal:
            self.monotonic = True
            self.monotonic_counter += 1
        else:
            self.monotonic = False
            self.monotonic_counter = 0
            self.previous_signal = current_signal

            long_sl = price - (price * self.stop_loss)
            long_tp = price + (price * self.take_profit)
            short_sl = price + (price * self.stop_loss)
            short_tp = price - (price * self.take_profit)

            if self.position.size < 1.0 - self.default_buy_size \
                    and current_signal == 1.0:
                if not self.monotonic:
                    self.buy(size=self.default_buy_size, sl=long_sl, tp=long_tp)
                    # self.position.entry_price = price
                elif self.monotonic and self.monotonic_counter > self.monotonic_normal_len:
                    self.buy(size=self.default_buy_size, sl=long_sl, tp=long_tp)
                    # self.position.entry_price = price
            elif self.position.size < 1.0 - self.default_sell_size \
                    and current_signal == -1.0:
                if not self.monotonic:
                    self.sell(size=self.default_sell_size, sl=short_sl, tp=short_tp)
                    # self.position.entry_price = price
                elif self.monotonic and self.monotonic_counter > self.monotonic_normal_len:
                    self.sell(size=self.default_sell_size, sl=short_sl, tp=short_tp)
                    # self.position.entry_price = price
            elif self.position.is_long and current_signal == -1.0:
                if not self.monotonic:
                    self.position.close()
                elif self.monotonic and self.monotonic_counter > self.monotonic_normal_len:
                    self.position.close()
            elif self.position.is_short and current_signal == 1.0:
                if not self.monotonic:
                    self.position.close()
                elif self.monotonic and self.monotonic_counter > self.monotonic_normal_len:
                    self.position.close()

    def __repr__(self):
        return self.name


class MyLongStrategy(BaseStrategy):
    name = 'MyLong'
    stop_loss = 0.02
    take_profit = 0.05

    def init(self):
        self.signal = self.I(lambda x: x, self.data.Signal, name='Signal')

    def next(self):
        super().next()
        price = self.data.Close[-1]
        if self.position:
            if self.signal == -1:
                # or self.signal == 0:
                self.position.close()
        else:
            if self.signal == 1:
                # or self.signal == 0:
                sl1 = price - (price * self.stop_loss)
                tp1 = price + (price * self.take_profit)
                self.buy(size=1.0, sl=sl1)
                # self.buy(size=1.0, sl=sl1, tp=tp1)
                # self.buy()
            elif self.signal == -1:
                # or self.signal == 0:
                self.position.close()

    def __repr__(self):
        return self.name


class MyNewLongStrategy(BaseStrategy):
    name = 'MyNewLong'
    stop_loss = 0.075
    # take_profit = 0.05
    # lot_size = 0.10
    # atr_f = 0.2
    # ratio_f = 1.0
    signals_count = 0
    signal_direction = 0

    def init(self):
        self.signal = self.I(lambda x: x, self.data.Signal, name='Signal')

    def next(self):
        # ToDo добавить торговлю ограниченным лотом
        # торгуем по крайней цене закрытия
        price = self.data.Close[-1]

        self.signals_count += 1

        if (self.position.is_long
                and self.signal == -1):
            self.signal_direction = -1
            self.position.close()
            self.signals_count = 0
        elif (self.position.is_long
              and self.signal == 1
              and self.signals_count > 2):
            self.signal_direction = 1
            self.position.close()
            sl1 = price - (price * self.stop_loss)
            halfsize = 0.5
            self.signals_count = 0
            self.buy(size=halfsize, sl=sl1)
        if (self.position.size == 0
                and self.signal == 1):
            self.signal_direction = 1
            sl1 = price - (price * self.stop_loss)
            self.buy(size=1.0, sl=sl1)
            self.signals_count += 1
            # self.buy(sl=0.8 * price)
            self.position.entry_price = price

    def __repr__(self):
        return self.name


class LongStrategy(BaseStrategy):
    name = 'Long'
    power_trend = 0.049

    def init(self):
        self.signal = self.I(lambda x: x, self.data.Signal, name='Signal')
        self.p_trend_df = CalcTrendDF(self.data.df, self.power_trend)()
        self.P_trend_new = self.I(lambda x: x, self.p_trend_df, name=f"P_trend_{self.power_trend}")

    def next(self):
        # ToDo добавить торговлю ограниченным лотом
        # торгуем по крайней цене закрытия
        price = self.data.Close[-1]

        if (self.position.is_long and
                self.signal == -1):
            self.position.close()

        if (self.position.size == 0 and
                self.signal == 1):
            self.buy()
            # self.buy(sl=0.8 * price)
            self.position.entry_price = price

    def __repr__(self):
        return self.name


class LongShortStrategy(BaseStrategy):
    name = 'LongShort'

    def init(self):
        self.signal = self.I(lambda x: x, self.data.Signal, name='Signal')

    def next(self):
        # торгуем по крайней цене закрытия
        price = self.data.Close[-1]

        if (self.position.is_long and
                self.signal == -1):
            self.position.close()

        elif (self.position.is_short and
              self.signal == 1):
            self.position.close()

        elif (self.position.size == 0 and
              self.signal == 1):
            self.buy()
            # self.buy(sl=0.8 * price)
            self.position.entry_price = price

        elif (self.position.size == 0 and
              self.signal == -1):
            self.sell()
