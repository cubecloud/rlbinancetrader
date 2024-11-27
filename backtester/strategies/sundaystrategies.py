

from datawizard.powertrend import *
from backtesting import Strategy


class SundayBase(Strategy):
    """
    Logic explanation:
    if sum of predictions 'plus' or 'minus' channels data greater,
    then threshold for 'plus'/'minus' channel, we put in orders stack 'buy','sell' or 'close' order.
    Orders stack will be processed on 'next' timeframe.
    """
    name = 'Long'
    plus_threshold = 3.68
    minus_threshold = 3.38

    # minor_plus_threshold = 0.51
    # minor_minus_threshold = 0.71

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.channels_dict: dict = {}
        self.orders_stack: list = []

    def _check_params(self, params):
        for k, v in params.items():
            setattr(self, k, v)
        return params

    def init(self):
        prefixes = ['plus', 'minus']
        for col_name in self.data.df.columns:
            for prefix in prefixes:
                if prefix in col_name.lower():
                    setattr(self, col_name.lower(), self.I(lambda x: x, self.data.df[col_name], name=col_name))

    def decision(self) -> list:
        todo_lst: list = []
        minus_indicator = getattr(self, 'minus')[-1]
        plus_indicator = getattr(self, 'plus')[-1]
        if self.position.is_long and minus_indicator > self.minus_threshold:
            todo_lst.extend(['close'])
        if self.position.size == 0 and plus_indicator > self.plus_threshold:
            todo_lst.extend(['buy'])
        return todo_lst

    def set_orders(self, todo: list):
        self.orders_stack.extend(todo)

    def process_orders(self):
        # торгуем по крайней цене закрытия
        current_price = self.data.Close[-1]
        while self.orders_stack:
            todo_order = self.orders_stack.pop(-1)
            if todo_order == 'close':
                self.position.close()
            elif todo_order == 'buy':
                self.position.entry_price = current_price
                self.buy()
            elif todo_order == 'sell':
                self.sell()

    def next(self):
        """
        Logic: the decision about what 2do we get on current timeframe, and setup orders for next timeframes.

        Returns:
            None
        """
        """ Process orders from previous timeframe """
        self.process_orders()
        todo = self.decision()
        if todo:
            self.set_orders(todo)

    def __repr__(self):
        return self.name


class SundayPtrend(SundayBase):
    """
    Logic explanation:
    1. prepare global trend data - power_trend=0.08

    """
    name = 'Ptrend'
    plus_threshold = 2.41
    minus_threshold = 1.67
    power_trend: float = 0.08
    stop_loss: float = 0.038
    take_profit: float = 0.06
    timeframes_period = 248

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.ptrend = None
        self.minus_indicator = None
        self.plus_indicator = None
        self.price = None
        self.previous_trend: int = 0
        self.trend_monotonic_counter: int = 0
        self.trend_monotonic: bool = False
        self.signal_minus_monotonic_counter: int = 0
        self.signal_minus_monotonic: bool = False
        self.previous_minus_signal: float = .0
        self.signal_plus_monotonic_counter: int = 0
        self.signal_plus_monotonic: bool = False
        self.previous_plus_signal: float = .0

    def init(self):
        pred_directions = ['plus', 'minus']
        for col_name in self.data.df.columns:
            for direction in pred_directions:
                if direction == col_name.lower():
                    setattr(self, col_name.lower(), self.I(lambda x: x, self.data.df[col_name], name=col_name))

    def decision(self) -> list:
        todo_lst: list = []
        self.price = self.data.Close[-1]
        self.minus_indicator = getattr(self, 'minus')[-1]
        self.plus_indicator = getattr(self, 'plus')[-1]

        if self.minus_indicator > 0.51:
            if (self.previous_minus_signal > self.minus_threshold) == (self.minus_indicator > self.minus_threshold):
                self.signal_minus_monotonic = True
                self.signal_minus_monotonic_counter += 1
            else:
                self.signal_minus_monotonic = False
                self.signal_minus_monotonic_counter = 0
            self.previous_minus_signal = self.minus_indicator
        else:
            self.previous_minus_signal = self.minus_indicator
            self.signal_minus_monotonic = False
            self.signal_minus_monotonic_counter = 0

        if self.plus_indicator > 0.51:
            if (self.previous_plus_signal > self.plus_threshold) == (self.plus_indicator > self.plus_threshold):
                self.signal_plus_monotonic = True
                self.signal_plus_monotonic_counter += 1
            else:
                self.signal_plus_monotonic = False
                self.signal_plus_monotonic_counter = 0
            self.previous_plus_signal = self.plus_indicator
        else:
            self.previous_plus_signal = self.plus_indicator
            self.signal_plus_monotonic = False
            self.signal_plus_monotonic_counter = 0

        ptrend = CalcTrend()(input_arr=self.data.df[['Open', 'High', 'Low', 'Close']].values,
                             W=self.power_trend)
        current_trend = ptrend[-1]

        indices = np.where(np.flip(ptrend) != current_trend)[0]
        if len(indices) > 0:
            trend_count = indices[0]
        elif len(ptrend) > 0:
            trend_count = len(ptrend)
        else:
            trend_count = 1

        if self.previous_trend == current_trend:
            self.trend_monotonic = True
            self.trend_monotonic_counter += 1
        else:
            self.trend_monotonic = False
            self.trend_monotonic_counter = 0
            trend_count = 0
            self.previous_trend = current_trend

        if self.position.is_long and (self.minus_indicator > self.minus_threshold):
            todo_lst.extend([('close',)])
        elif (self.position.size == 0.0 and (self.plus_indicator > self.plus_threshold) and
              (self.signal_plus_monotonic_counter > 2) and (current_trend == 0.)):
            """ Unknown trend """
            todo_lst.extend([('buy', self.stop_loss, self.take_profit)])
        elif (self.position.size == 0.0 and (self.plus_indicator > self.plus_threshold) and
              (self.signal_plus_monotonic_counter > 2) and (current_trend == 1.)):
            """ Plus trend """
            todo_lst.extend([('buy', self.stop_loss, self.take_profit)])
        elif (self.position.size == 0.0 and (self.plus_indicator > self.plus_threshold) and
              (self.trend_monotonic_counter > 2) and (current_trend == -1.)):
            """ Minus trend """
            todo_lst.extend([('buy', self.stop_loss, self.take_profit)])
        elif (self.position.size == 0.0 and (self.minus_indicator > self.minus_threshold) and
              (self.signal_minus_monotonic_counter > 2) and (current_trend == -1.)):
            """ Minus trend and signal minus """
            todo_lst.extend([('buy', self.stop_loss / 2, self.take_profit / 2)])
            todo_lst.extend([('hold', self.timeframes_period)])
            todo_lst.extend([('close',)])
        elif (self.position.size == 0.0 and (self.minus_indicator > self.minus_threshold) and
              (self.signal_minus_monotonic_counter > 5) and (current_trend == 1.) and (
                      trend_count > self.timeframes_period)):
            """ Plus trend and signal minus """
            todo_lst.extend([('buy', self.stop_loss, self.take_profit)])
            todo_lst.extend([('hold', self.timeframes_period)])
            todo_lst.extend([('close',)])
        return todo_lst

    def process_orders(self):
        # торгуем по крайней цене закрытия
        current_price = self.data.Close[-1]

        while self.orders_stack:
            todo_order = self.orders_stack.pop(0)
            if todo_order[0] == 'close':
                # self.position.close(portion=todo_order[1])
                self.position.close()
            elif todo_order[0] == 'buy':
                self.position.entry_price = current_price
                sl = current_price - (current_price * todo_order[1])
                tp = current_price + (current_price * todo_order[2])
                self.buy(sl=sl, tp=tp)
            elif todo_order[0] == 'sell':
                # self.sell(size=todo_order[1], sl=todo_order[2], tp=todo_order[3])
                sl = current_price - (current_price * todo_order[1])
                tp = current_price + (current_price * todo_order[2])
                self.sell(sl=sl, tp=tp)
            elif todo_order[0] == 'hold':
                if todo_order[1] > 0:
                    self.orders_stack.insert(0, (todo_order[0], todo_order[1] - 1))
                    break


class TwoEyes(SundayBase):
    """
    Logic explanation:
    1. prepare global trend data - power_trend=0.08
    """
    name = 'TwoEyes'
    plus_threshold = 0.58
    minus_threshold = 0.88
    stop_loss: float = 0.02
    take_profit: float = 0.095
    timeframes_period = 108

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.minus_indicator = None
        self.plus_indicator = None
        self.price = None
        self.signal_minus_monotonic_counter: int = 0
        self.signal_minus_monotonic: bool = False
        self.previous_minus_signal: float = .0
        self.signal_plus_monotonic_counter: int = 0
        self.signal_plus_monotonic: bool = False
        self.previous_plus_signal: float = .0

    def init(self):
        pred_directions = ['plus', 'minus']
        for col_name in self.data.df.columns:
            for direction in pred_directions:
                if direction == col_name.lower():
                    setattr(self, col_name.lower(), self.I(lambda x: x, self.data.df[col_name], name=col_name))

    def decision(self) -> list:
        todo_lst: list = []
        self.price = self.data.Close[-1]
        self.minus_indicator = getattr(self, 'minus')[-1]
        self.plus_indicator = getattr(self, 'plus')[-1]

        if self.minus_indicator > 0.51:
            if (self.previous_minus_signal > self.minus_threshold) == (self.minus_indicator > self.minus_threshold):
                self.signal_minus_monotonic = True
                self.signal_minus_monotonic_counter += 1
            else:
                self.signal_minus_monotonic = False
                self.signal_minus_monotonic_counter = 0
            self.previous_minus_signal = self.minus_indicator
        else:
            self.previous_minus_signal = self.minus_indicator
            self.signal_minus_monotonic = False
            self.signal_minus_monotonic_counter = 0

        if self.plus_indicator > 0.51:
            if (self.previous_plus_signal > self.plus_threshold) == (self.plus_indicator > self.plus_threshold):
                self.signal_plus_monotonic = True
                self.signal_plus_monotonic_counter += 1
            else:
                self.signal_plus_monotonic = False
                self.signal_plus_monotonic_counter = 0
            self.previous_plus_signal = self.plus_indicator
        else:
            self.previous_plus_signal = self.plus_indicator
            self.signal_plus_monotonic = False
            self.signal_plus_monotonic_counter = 0

        if self.position.is_long and (self.minus_indicator > self.minus_threshold):
            todo_lst.extend([('close',)])
        elif (self.position.size == 0.0 and (self.plus_indicator > self.plus_threshold) and
              (self.signal_plus_monotonic_counter > 2)):
            todo_lst.extend([('buy', self.stop_loss, self.take_profit)])
        elif (self.position.size == 0.0 and (self.minus_indicator > self.minus_threshold) and
              (self.signal_minus_monotonic_counter > 5)):
            todo_lst.extend([('buy', self.stop_loss, self.take_profit)])
            todo_lst.extend([('hold', self.timeframes_period)])
            todo_lst.extend([('close',)])
        return todo_lst

    def process_orders(self):
        # торгуем по крайней цене закрытия
        current_price = self.data.Close[-1]

        while self.orders_stack:
            todo_order = self.orders_stack.pop(0)
            if todo_order[0] == 'close':
                # self.position.close(portion=todo_order[1])
                self.position.close()
            elif todo_order[0] == 'buy':
                buy_kwargs: dict = {}
                self.position.entry_price = current_price
                sl = current_price - (current_price * todo_order[1])
                tp = current_price + (current_price * todo_order[2])
                buy_kwargs.update({"sl": sl,
                                   "tp": tp})
                self.buy(**buy_kwargs)
            elif todo_order[0] == 'sell':
                sell_kwargs: dict = {}
                self.position.entry_price = current_price
                sl = current_price + (current_price * todo_order[1])
                tp = current_price - (current_price * todo_order[2])
                sell_kwargs.update({"sl": sl,
                                    "tp": tp})
                self.sell(**sell_kwargs)
            elif todo_order[0] == 'hold':
                if todo_order[1] > 0:
                    self.orders_stack.insert(0, (todo_order[0], todo_order[1] - 1))
                    break


class ThreeEyes(TwoEyes):
    """
    Logic explanation:
        like 2 Eyes, but:
        if after buy signal Close going down on next timeframe we are postpone buy for one timeframe again.
    """
    name = 'ThreeEyes'
    plus_threshold = 3.03
    minus_threshold = 1.07
    stop_loss: float = 0.053
    take_profit: float = 0.1
    timeframes_period = 240

    def process_orders(self):
        # торгуем по крайней цене закрытия
        current_price = self.data.Close[-1]
        while self.orders_stack:
            todo_order = self.orders_stack.pop(0)
            if todo_order[0] == 'close':
                # self.position.close(portion=todo_order[1])
                self.position.close()
            elif todo_order[0] == 'buy':
                self.position.entry_price = current_price
                if self.data.Close[-2] < current_price:
                    sl = current_price - (current_price * todo_order[1])
                    tp = current_price + (current_price * todo_order[2])
                    self.buy(sl=sl, tp=tp)
                else:
                    # postpone buy for next process_orders
                    if len(self.orders_stack) > 0:
                        # if we have more orders -> process them
                        if todo_order == self.orders_stack[0]:
                            self.orders_stack.pop(0)
                        self.process_orders()
                    self.orders_stack.append(('buy', self.stop_loss, self.take_profit))
                    break
            elif todo_order[0] == 'sell':
                # self.sell(size=todo_order[1], sl=todo_order[2], tp=todo_order[3])
                sl = current_price - (current_price * todo_order[1])
                tp = current_price + (current_price * todo_order[2])
                self.sell(sl=sl, tp=tp)
            elif todo_order[0] == 'hold':
                if todo_order[1] > 0:
                    self.orders_stack.insert(0, (todo_order[0], todo_order[1] - 1))
                    break


class SixEyes(Strategy):
    """
    Logic explanation:
        1. Choosing 4 channels from models, (2 'plus' channels and 2 'minus' channels),
           and 2 summation channels 'plus' and 'minus'
        2. Using this channels for creating optimized logic, for trading strategy
    """

    name = 'SixEyes'
    plus_threshold = 2.57
    minus_threshold = 1.73
    power_trend: float = 0.075
    stop_loss: float = 0.056
    take_profit: float = 0.1
    timeframes_period = 140
    channels_lst = (0.019, '1h')

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.channels_dict: dict = {}
        self.indicators_list: list = []
        self.orders_stack: list = []
        self.events_data = None

    def _check_params(self, params):
        for k, v in params.items():
            setattr(self, k, v)
        return params

    def prepare_models_channels(self):
        pass

    def init(self):
        prefixes = ['plus', 'minus']
        for col_name in self.data.df.columns:
            for prefix in prefixes:
                if prefix in col_name.lower():
                    setattr(self, col_name.lower(), self.I(lambda x: x, self.data.df[col_name], name=col_name))
                    self.indicators_list.append(col_name.lower())
                    channel_data = self.unpack_channel(col_name.lower())
                    if channel_data is not None:
                        self.channels_dict.update({col_name.lower(): channel_data})

    def unpack_channel(self, channel_name: str):
        splited = channel_name.split('_')
        if len(splited) > 1:
            event, timeframe, power_trend, timeframes_period = splited
            power_trend = float(power_trend)
            timeframes_period = int(timeframes_period)
            result = (event, timeframe, power_trend, timeframes_period)
        else:
            result = None
        return result

    def __repr__(self):
        return self.name


class SimpleSpiderEyes(SundayBase):
    """
    Logic explanation:
    1. prepare global trend data - power_trend=0.076
    2. process each of predicted channels and stack triggers

    """
    name = 'Eyes'

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.triggers_stack: list = []
        self.indicators_list: list = []

    def init(self):
        prefixes = ['plus', 'minus']
        for col_name in self.data.df.columns:
            for prefix in prefixes:
                if prefix in col_name.lower():
                    setattr(self, col_name.lower(), self.I(lambda x: x, self.data.df[col_name], name=col_name))
                    self.indicators_list.append(col_name.lower())
                    data = self.unpack_channel(col_name.lower())
                    if data is not None:
                        self.channels_dict.update({col_name.lower(): data})

    def unpack_channel(self, channel_name: str):
        splited = channel_name.split('_')
        if len(splited) > 1:
            event, timeframe, power_trend, timeframes_period = splited
            power_trend = float(power_trend)
            timeframes_period = int(timeframes_period)
            result = (event, timeframe, power_trend, timeframes_period)
        else:
            result = None
        return result

    def process_orders(self):
        # торгуем по крайней цене закрытия
        current_price = self.data.Close[-1]
        while self.orders_stack:
            todo_order = self.orders_stack.pop(-1)
            if todo_order == 'buy':
                self.position.entry_price = current_price
                self.buy()
            elif todo_order == 'sell':
                self.sell()
            elif todo_order == 'close':
                self.position.close()

    def process_channels(self):
        for channel_name, data in self.channels_dict.items():
            indicator_data = getattr(self, channel_name)
            indicator_max = np.max(indicator_data)
            indicator_threshold = indicator_max / 1.15
            if indicator_max < 0.576 or indicator_data[-1] < indicator_threshold:
                continue
            event, timeframe, power_trend, timeframes_period = data
            if event == 'plus':
                pass
            elif event == 'minus':
                pass

    def decision(self) -> list:
        todo_lst: list = []
        minus_indicator = getattr(self, 'minus')[-1]
        plus_indicator = getattr(self, 'plus')[-1]
        if self.position.is_long and minus_indicator > self.minus_threshold:
            todo_lst.extend(['close'])
        if self.position.size == 0 and plus_indicator > self.plus_threshold:
            todo_lst.extend(['buy'])
        return todo_lst

    def next(self):
        """
        Logic: the decision about what 2do we get on current timeframe, and setup orders for next timeframes.

        Returns:
            None
        """

        """ Process orders from previous timeframe """
        self.process_orders()
        self.process_channels()
        todo = self.decision()
        if todo:
            self.set_orders(todo)
