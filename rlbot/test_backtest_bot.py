import os
import sys
import copy
import logging

import datetime
import backtrader as bt
from collections import deque
from backtrader_binance import BinanceStore
from backtrader_binance.binance_broker import BinanceBroker

# from ConfigBinance.Config import Config  # Configuration file
from dbbinance.config.configpostgresql import ConfigPostgreSQL
from dbbinance.config.configbinance import ConfigBinance

from dbbinance.fetcher import ceil_time, floor_time, get_timedelta_kwargs, Constants
import backtester.indicators.btdbindicators as btdbind
from indicators.rlbtindicator import *

# import yaml
# import pprint
# import pandas as pd

# from basictradedtats import BasicTradeStats
# from backtrader_plotting import Bokeh
# from backtrader_plotting.schemes import Tradimo


__version__ = 0.040

logger = logging.getLogger()


def live_buy_and_oco_bracket(strategy_obj, price, stop_loss_price, take_profit_price, size):
    # Manual brackets
    mainside = strategy_obj.buy(data=data, price=price, size=size, exectype=bt.Order.Limit,
                                transmit=False)
    lowside = strategy_obj.sell(data=data, price=stop_loss_price, size=mainside.size,
                                exectype=bt.Order.StopLimit,
                                transmit=False, parent=mainside)
    highside = strategy_obj.sell(data=data, price=take_profit_price, size=mainside.size,
                                 exectype=BinanceBroker.Order.TakeProfitLimit,
                                 transmit=True, parent=mainside)
    brackets = [mainside, lowside, highside]
    return brackets


def live_buy_sl_tp_bracket(strategy_obj, price, stop_loss_price, take_profit_price, size):
    # Manual brackets
    mainside = strategy_obj.buy(data=data, price=price, size=size, exectype=bt.Order.Limit,
                                transmit=False)
    lowside = strategy_obj.sell(data=data, price=stop_loss_price, size=mainside.size,
                                exectype=bt.Order.StopLimit,
                                transmit=False, parent=mainside)
    highside = strategy_obj.sell(data=data, price=take_profit_price, size=mainside.size,
                                 exectype=BinanceBroker.Order.TakeProfitLimit,
                                 transmit=True, parent=mainside)
    brackets = [mainside, lowside, highside]
    return brackets


def history_buy_sl_tp_bracket(strategy_obj, price, stop_loss_price, take_profit_price, size):
    # Manual brackets
    mainside = strategy_obj.buy(data=data, price=price, size=size, exectype=bt.Order.Limit,
                                transmit=False)
    lowside = strategy_obj.sell(data=data, price=stop_loss_price, size=mainside.size,
                                exectype=bt.Order.Stop,
                                transmit=True, parent=mainside)
    highside = strategy_obj.sell(data=data, price=take_profit_price, size=mainside.size,
                                 exectype=bt.Order.Limit,
                                 transmit=True, parent=mainside)
    brackets = [mainside, lowside, highside]
    return brackets


def live_buy_sl_bracket(strategy_obj, price, stop_loss_price, size):
    # Manual brackets
    mainside = strategy_obj.buy(data=data, price=price, size=size, exectype=bt.Order.Limit,
                                transmit=False)
    lowside = strategy_obj.sell(data=data, price=stop_loss_price, size=mainside.size,
                                exectype=bt.Order.StopLimit,
                                transmit=True, parent=mainside)
    brackets = [mainside, lowside]
    return brackets


def history_buy_sl_bracket(strategy_obj, price, stop_loss_price, size):
    # Manual brackets
    mainside = strategy_obj.buy(data=data, price=price, size=size, exectype=bt.Order.Limit,
                                transmit=False)
    lowside = strategy_obj.sell(data=data, price=stop_loss_price, size=mainside.size,
                                exectype=bt.Order.Stop,
                                transmit=True, parent=mainside)
    brackets = [mainside, lowside]
    return brackets


# Trading System
class AIStrategy(bt.Strategy):
    """
    Live strategy demonstration with SMA, RSI indicators
    """
    params = (  # Default parameters of the trading system
        ('coin_target', ''),
        ('coin_commission', 'BNB'),
        ('timeframe', '15m'),
        ('discretization', '15m'),
        ('end_datetime', None),
        ('not_online', True),
    )
    name = 'AIStrategy'

    def __init__(self):
        """Initialization, adding indicators for each ticker"""
        self.orders = {}  # All orders as a dict, for this particularly trading strategy one ticker is one order
        for d in self.datas:  # Running through all the tickers
            self.orders[d._name] = list()  # There is no order for ticker yet

        # creating indicators for each ticker
        self.sellai: dict = {}
        self.buyai: dict = {}
        self.online_status = (not self.p.not_online)

        self.trade_fee: dict = {}
        self.todo_stack = {}

        self.live_just_started: dict = {}
        self.stop_trading_switch = 0.2

        for i in range(len(self.datas)):
            ticker = list(self.dnames.keys())[i]  # key name is ticker name
            self.buyai[ticker] = btdbind.BuyAI(self.datas[i],
                                               state=self.datas[i]._state,
                                               timeframe=self.p.timeframe,
                                               discretization=self.p.discretization,
                                               end_datetime=self.p.end_datetime
                                               )  # BuyAI indicator
            self.sellai[ticker] = btdbind.SellAI(self.datas[i],
                                                 state=self.datas[i]._state,
                                                 timeframe=self.p.timeframe,
                                                 discretization=self.p.discretization,
                                                 end_datetime=self.p.end_datetime
                                                 )  # SellAI indicator

            self.todo_stack[ticker] = deque()
            self.live_just_started[ticker] = False
            self.init()
            if self.online_status:
                self.online_init(ticker)
            else:
                # Set the commission - 0.075% ... divide by 100 to remove %
                self.trade_fee[ticker] = dict(makerCommission=0.0750 / 100,
                                              takerCommission=0.0750 / 100)

    # def get_cash(self):
    #     _cash = self.broker.getcash()
    #     logger.info(f'{self.__class__.__name__}: Free balance: {_cash} {self.p.coin_target}')
    #     return _cash

    def init(self):
        self.starting_orders_history: dict = {}

        self.starting_coin_target_balance: dict = {}
        self.starting_locked_target_balance: dict = {}
        self.total_starting_coin_target_balance: dict = {}

        self.starting_coin_symbol_balance: dict = {}
        self.starting_locked_symbol_balance: dict = {}
        self.total_starting_coin_symbol_balance: dict = {}

        self.starting_coin_commission_balance: dict = {}
        self.starting_locked_commission_balance: dict = {}
        self.total_starting_coin_commission_balance: dict = {}
        self.total_in_target: dict = {}

        self.starting_open_positions: dict = {}
        self.starting_open_orders: dict = {}

    def online_init(self, ticker):
        self._init_get_ticker_all_orders(ticker)  # get all ticker orders
        self._init_get_ticker_open_orders(ticker)
        self.check_starting_open_orders(ticker=ticker)
        self._init_get_ticker_fee(ticker)
        self._init_get_symbol_balance(ticker)
        self._init_get_target_balance(ticker)
        self._init_get_commission_balance(ticker)
        self._init_get_totals(ticker)
        pass

    def _init_get_symbol_balance(self, ticker):

        coin_symbol = ticker[:-(len(coin_target))]
        #   get coin symbol data
        self.starting_coin_symbol_balance[ticker], self.starting_locked_symbol_balance[
            ticker] = self.broker.get_asset_balance(coin_symbol)
        self.total_starting_coin_symbol_balance[ticker] = self.starting_coin_symbol_balance[ticker] + \
                                                          self.starting_locked_symbol_balance[ticker]

    def _init_get_target_balance(self, ticker):
        #   get coin target data
        self.starting_coin_target_balance[ticker], self.starting_locked_target_balance[
            ticker] = self.broker.get_asset_balance(self.p.coin_target)
        self.total_starting_coin_target_balance[ticker] = self.starting_coin_target_balance[ticker] + \
                                                          self.starting_locked_target_balance[ticker]

    def _init_get_commission_balance(self, ticker):
        #   get coin commission data
        self.starting_coin_commission_balance[ticker], self.starting_locked_commission_balance[
            ticker] = self.broker.get_asset_balance(self.p.coin_commission)
        self.total_starting_coin_commission_balance[ticker] = self.starting_coin_commission_balance[ticker] + \
                                                              self.starting_locked_commission_balance[ticker]

    def _init_get_totals(self, ticker):
        avg_symbol_price = self.broker.get_avg_price(ticker)
        avg_commission_price = self.broker.get_avg_price(self.p.coin_commission + self.p.coin_target)
        self.total_in_target[ticker] = self.total_starting_coin_symbol_balance[ticker] * avg_symbol_price
        self.total_in_target[ticker] += (self.total_starting_coin_commission_balance[ticker] * avg_commission_price)
        self.total_in_target[ticker] += self.total_starting_coin_target_balance[ticker]

    def _init_get_ticker_fee(self, ticker):
        """
        Using Binance broker to get symbol fee
        Args:
            ticker (str): ticker symbol (e.g. BTCUSDT)
        Returns:
            None:
        """
        trade_fee = self.broker.get_trade_fee(ticker)[0]
        self.trade_fee[ticker] = dict(takerCommission=float(trade_fee['takerCommission']),
                                      makerCommission=float(trade_fee['makerCommission']))

    def _init_get_ticker_open_orders(self, ticker):
        """
        Using Binance broker to get open orders
        Args:
            ticker (str): ticker symbol (e.g. BTCUSDT)

        Returns:
            None:
        """
        #   get open orders
        self.starting_open_orders[ticker] = self.broker.get_open_orders(symbol=ticker)

    def _init_get_ticker_all_orders(self, ticker):
        """
        Using Binance broker to get all orders

        Args:
            ticker (str): ticker symbol (e.g. BTCUSDT)

        Returns:
            None:
        """
        #   get all orders (history)
        self.starting_orders_history[ticker] = self.broker.get_all_orders(symbol=ticker)

    def check_starting_open_orders(self, ticker):
        """
        Check open orders

        Args:
            ticker (str): ticker symbol (e.g. BTCUSDT)

        Returns:
            None:
        """
        if self.starting_open_orders[ticker]:
            for open_order in self.starting_open_orders[ticker]:
                side = open_order.get('SIDE', None)
                if side == 'BUY':
                    pass
                elif side == 'SELL':
                    pass
                pass

    def check_live_trading_stop_switch(self, ticker: str, assets_sum_target_coin_now: float):
        """
        Check if trading loss higher than live_trading_stop_switch value -> stop trading and sys.exit()
        """

        if assets_sum_target_coin_now < (self.total_in_target[ticker] / (1 + self.stop_trading_switch)):
            logger.warning(
                f'{self.__class__.__name__}: Trading loss > {self.stop_trading_switch * 100}%. Stopping.'
                f'\nStarting balance: {self.total_in_target}'
                f'\nEnding balance: {assets_sum_target_coin_now}.')
            sys.exit('Stopping strategy... Exit...')

    def next(self):
        """Arrival of a new ticker candle"""
        for data in self.datas:  # Running through all the requested bars of all tickers
            total_target_balance = .0
            ticker = data._name
            status = data._state  # 0 - Live data, 1 - History data, 2 - None

            if status in [0, 1]:
                coin_target = self.p.coin_target

                # Show current info about data, state, ticker and etc
                msg = (f'{self.__class__.__name__}:'
                       f'\t - {ticker} BuyAI: {self.buyai[ticker][0]:.3f}, SellAI: {self.sellai[ticker][0]:.3f}')

                if status:
                    _state = "(1) - History data"
                    _interval = self.p.timeframe
                    coin_target_balance = self.broker.getcash()
                    msg = f'{msg} - Free balance: {coin_target_balance} {coin_target}'
                else:
                    _state = "(0) - Live data"
                    _interval = self.broker._store.get_interval(data._timeframe, data._compression)
                    coin_commission = self.p.coin_commission
                    coin_commission_balance, locked_commission_balance = self.broker.get_asset_balance(
                        coin_commission)
                    coin_target_balance, locked_target_balance = self.broker.get_asset_balance(coin_target)
                    coin_symbol = ticker[:-(len(coin_target))]
                    symbol_balance, locked_symbol_balance = self.broker.get_asset_balance(coin_symbol)
                    total_target_balance += sum([symbol_balance * data.close[0], locked_symbol_balance * data.close[0]])
                    avg_commission_price = self.broker.get_avg_price(self.p.coin_commission + self.p.coin_target)
                    total_target_balance += (coin_commission_balance + locked_commission_balance) * avg_commission_price
                    msg = (f'{msg}\n\t- Free: {coin_target_balance} {coin_target}, '
                           f'Locked: {locked_target_balance} {coin_target} // '
                           f'Free: {symbol_balance} {coin_symbol}, '
                           f'Locked: {locked_symbol_balance} {coin_symbol} // '
                           f'Free: {coin_commission_balance} {coin_commission}, '
                           f'Locked: {locked_commission_balance} {coin_commission}')

                logger.info(
                    f'{self.__class__.__name__}: {bt.num2date(data.datetime[0])} / {ticker} '
                    f'[{_interval}] - Open: {data.open[0]}, High: {data.high[0]}, '
                    f'Low: {data.low[0]}, Close: {data.close[0]}, Volume: {data.volume[0]} - '
                    f'State: {_state}')

                logger.info(msg)

            def live_buy_bracket(price, stop_loss_price, take_profit_price, size):
                # Manual brackets
                mainside = self.buy(data=data, price=price, size=size, exectype=bt.Order.Limit,
                                    transmit=False)
                lowside = self.sell(data=data, price=stop_loss_price, size=mainside.size,
                                    exectype=bt.Order.StopLimit,
                                    transmit=False, parent=mainside)
                highside = self.sell(data=data, price=take_profit_price, size=mainside.size,
                                     exectype=BinanceBroker.Order.TakeProfitLimit,
                                     transmit=True, parent=mainside)
                brackets = [mainside, lowside, highside]
                return brackets

            def history_buy_bracket(price, stop_loss_price, take_profit_price, size):
                # Manual brackets
                mainside = self.buy(data=data, price=price, size=size, exectype=bt.Order.Limit,
                                    transmit=False)
                lowside = self.sell(data=data, price=stop_loss_price, size=mainside.size,
                                    exectype=bt.Order.Stop,
                                    transmit=False, parent=mainside)
                highside = self.sell(data=data, price=take_profit_price, size=mainside.size,
                                     exectype=bt.Order.Limit,
                                     transmit=True, parent=mainside)
                brackets = [mainside, lowside, highside]
                return brackets

            def process_todo_stack():
                counter = 0
                current_datetime = bt.num2date(data.datetime[0])
                while counter < len(self.todo_stack[ticker]):
                    todo_action = self.todo_stack[ticker].popleft()
                    _name, _target_time, _kwargs = todo_action
                    counter += 1
                    """
                    We waiting for target_time to use send orders to Broker
                    """
                    if _target_time == current_datetime:
                        if _name == 'buy':
                            price = data.close[0]  # by closing price
                            size = (coin_target_balance / price) * (1. - (
                                    self.trade_fee[ticker]['makerCommission'] + self.trade_fee[ticker][
                                'takerCommission']))

                            if not status:
                                size = float(self.broker._store.format_quantity(ticker, size))

                            stop_loss_price = price - (price * _kwargs['stop_loss'])
                            log_msg = (
                                f'{self.__class__.__name__}:\t - buy {ticker}, size = {size} at price = {price}, '
                                f'stop_loss ({_kwargs["stop_loss"]}) = {stop_loss_price}')
                            take_profit_price = .0
                            if _kwargs.get('take_profit', None) is not None:
                                take_profit_price = price + (price * _kwargs['take_profit'])
                                log_msg = f'{log_msg}, take_profit ({_kwargs["take_profit"]}) = {take_profit_price}'

                            if ticker == 'BTCUSDT' and size < 0.00001:
                                logger.warning(
                                    f'{self.__class__.__name__}:\n {ticker} Not enough USDT funds - trade skipped')
                                return

                            if status:
                                brackets = history_buy_bracket(price, stop_loss_price, take_profit_price, size)
                            else:
                                brackets = live_buy_bracket(price, stop_loss_price, take_profit_price, size)

                            # if status:
                            #     brackets = history_buy_sl_tp_bracket(self, price, stop_loss_price, take_profit_price, size)
                            # else:
                            #     brackets = live_buy_sl_tp_bracket(self, price, stop_loss_price, take_profit_price, size)

                            self.orders[ticker].extend(brackets)
                            logger.debug(log_msg)

                            if status:
                                for opened_order in brackets:
                                    logger.info(
                                        f'{self.__class__.__name__}:'
                                        f'\t - {ticker} Order {opened_order.ref} {opened_order.getordername()} '
                                        f'{opened_order.getstatusname()} has been submitted')

                        elif _name == 'close':
                            if self.orders[data._name]:
                                for order in self.orders[data._name]:
                                    # If the order is on the exchange (accepted by the broker)
                                    if order and order.status == bt.Order.Accepted:
                                        logger.info(
                                            f'{self.__class__.__name__}:'
                                            f'\t{bt.num2date(data.datetime[0])} '
                                            f'- Cancel the order {order.p.tradeid} to {order.ordtypename()} {ticker}')
                                        # Cancel order at broker
                                        self.cancel(order)
                            logger.info(
                                f'{self.__class__.__name__}:\t - {ticker} SellAI: {_kwargs}')
                            # Close position with `close` order
                            self.orders[data._name].append(self.close())
                            logger.info(f'{self.__class__.__name__}:\t - {ticker} Close it by the market')
                    # if todo_ation target_datetime > current_datetime -> postpone order
                    elif _target_time > current_datetime:
                        self.todo_stack[ticker].append(todo_action)

            process_todo_stack()

            for order in self.orders[data._name]:
                # If the order is not on the exchange (sent to the broker)
                if order and order.status == bt.Order.Submitted:
                    # then we are waiting for the order to be placed on the exchange,
                    # we leave, we do not continue further
                    return

            if not self.getposition(data):  # If there is no position
                if self.buyai[ticker] > .0:
                    order_target_time = ceil_time(self.buyai[ticker].target_time)
                    order_kwargs = copy.deepcopy(self.buyai[ticker].kwargs)
                    action = ('buy', order_target_time, order_kwargs)
                    self.todo_stack[ticker].append(action)
                    logger.info(
                        f'{self.__class__.__name__}:\t - {ticker} BuyAI: {order_target_time} {order_kwargs}')
            else:  # If there is a position
                if self.sellai[ticker] > .0:
                    order_target_time = ceil_time(self.sellai[ticker].target_time)
                    order_kwargs = copy.deepcopy(self.sellai[ticker].kwargs)
                    action = ('close', order_target_time, order_kwargs)
                    self.todo_stack[ticker].append(action)
                    logger.info(
                        f'{self.__class__.__name__}:\t - {ticker} SellAI: {order_target_time} {order_kwargs}')

            if not status:
                total_target_balance += sum([coin_target_balance, locked_target_balance])
                self.check_live_trading_stop_switch(ticker, total_target_balance)
                logger.info(f'{self.__class__.__name__}: '
                            f'Starting balance: {self.total_in_target[ticker]:.5f} '
                            f'/ Current balance: {total_target_balance:.5f} '
                            f'/ PNL: {((total_target_balance / self.total_in_target[ticker]) - 1) * 100:.2f}%')


def notify_order(self, order):
    """Changing the status of the order"""
    # Name of ticker from order
    ticker = order.data._name
    self.log(
        f'Order number {order.ref} {order.getordername()} {order.getstatusname()} '
        f'{"Buy" if order.isbuy() else "Sell"} {ticker} {order.size} @ {order.price}')
    if order.status == bt.Order.Completed:  # If the order is fully executed
        # The order to buy
        if order.isbuy():
            self.log(
                f'Buy {ticker} @{order.executed.price:.2f}, '
                f'Value {order.executed.value:.2f}, Commission {order.executed.comm:.2f}')
        # The order to sell
        else:
            self.log(
                f'Sell {ticker} @{order.executed.price:.2f}, '
                f'Value {order.executed.value:.2f}, Commission {order.executed.comm:.2f}')
        # Remove Completed order from strategy orders list
        self.orders[ticker].remove(order)
    elif order.status == bt.Order.Canceled:
        self.log(
            f'Canceled {ticker} @{order.executed.price:.2f}, '
            f'Value {order.executed.value:.2f}, Commission {order.executed.comm:.2f}')
        # Remove NOT USED Canceled orders from strategy orders list
        self.orders[ticker].remove(order)


def notify_trade(self, trade):
    """Changing the position status"""
    if trade.isclosed:  # If the position is closed
        self.log(
            f'Profit on a closed position {trade.getdataname()}, '
            f'Total/wo {trade.pnl:.2f}/{trade.pnlcomm:.2f} Fee: {trade.commission}')


def log(self, txt, dt=None):
    """Print string with date to the console"""
    # date or date of the current bar
    dt = bt.num2date(self.datas[0].datetime[0]) if not dt else dt
    # Print the date and time with the specified text to the console
    logger.info(
        f'{self.__class__.__name__}: {dt.strftime("%d.%m.%Y %H:%M")}, {txt}')


if __name__ == '__main__':
    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('/home/cubecloud/Python/projects/sunday/live_strategy_test.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    logging.getLogger('binance.streams').setLevel(logging.INFO)
    logging.getLogger('websockets.protocol').setLevel(logging.INFO)
    logging.getLogger('websockets.server').setLevel(logging.INFO)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.INFO)
    logging.getLogger('matplotlib.pyplot').setLevel(logging.INFO)
    logging.getLogger('matplotlib.ticker').setLevel(logging.INFO)
    logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)

    cerebro = bt.Cerebro(quicknotify=True)

    # Setting how much money
    cerebro.broker.setcash(10000)

    # Set the commission - 0.1% ... divide by 100 to remove %
    cerebro.broker.setcommission(commission=0.001, margin=None, )

    coin_target = 'USDT'  # the base ticker in which calculations will be performed
    symbol = 'BTC' + coin_target  # the ticker by which we will receive data in the format <CodeTickerBaseTicker>
    # symbol2 = 'ETH' + coin_target  # the ticker by which we will receive data in the format <CodeTickerBaseTicker>

    LIVE_TRADING = False
    timeframe = '30m'
    discretization = '30m'
    backtest_period = '21w'
    compression = Constants.binsizes.get(timeframe)
    assert compression is not None, f"Error: unknown timeframe {timeframe}"

    store = BinanceStore(
        api_key=ConfigBinance.BINANCE_API_KEY,
        api_secret=ConfigBinance.BINANCE_API_SECRET,
        coin_target=coin_target,
        testnet=False)  # Binance Storage

    if LIVE_TRADING:
        # live connection to Binance - for Offline comment these two lines
        broker = store.getbroker()
        cerebro.setbroker(broker)
        end_datetime = floor_time(datetime.datetime.utcnow(), timeframe)
        timedelta_kwargs = get_timedelta_kwargs(discretization)
        start_datetime = end_datetime - datetime.timedelta(**timedelta_kwargs)
        LiveBars = True
        logger.warning("Attention! - Now it's Online!! ")
    else:
        end_datetime = ceil_time(datetime.datetime.utcnow(), timeframe)
        # end_datetime = datetime.datetime.strptime("2024-02-21 06:00:00", Constants.default_datetime_format)
        # start_datetime = end_datetime - datetime.timedelta(minutes=60 * 12)
        timedelta_kwargs = get_timedelta_kwargs(backtest_period)
        start_datetime = end_datetime - datetime.timedelta(**timedelta_kwargs)
        LiveBars = False
        logger.warning("Attention! - Now it's Offline for testing strategies")

    data = store.getdata(timeframe=bt.TimeFrame.Minutes, compression=compression, dataname=symbol,
                         start_date=start_datetime, LiveBars=LiveBars)  # set True here - if you need to get live bars
    cerebro.adddata(data)  # Adding data
    strategy_kwargs = dict(coin_target=coin_target,
                           timeframe=timeframe,
                           discretization=discretization,
                           not_online=False,
                           )

    print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))
    if not LIVE_TRADING:
        strategy_kwargs['end_datetime'] = end_datetime
        strategy_kwargs['not_online'] = True
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        # cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')

    cerebro.addstrategy(AIStrategy, **strategy_kwargs)
    logger.info(f'{__name__}: Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    try:
        thestrats = cerebro.run()
        logger.info(f'{__name__}: Final Portfolio Value: {cerebro.broker.getvalue():.2f}')
    except (KeyboardInterrupt, SystemExit):
        logger.info(f'{__name__}: SystemExit: Ctrl+{"Break" if os.name == "nt" else "C"} pressed')
    finally:
        if LIVE_TRADING:
            cerebro.plot()
        else:
            thestrat = thestrats[0]
            print('Sharpe ratio:', thestrat.analyzers.sharpe.get_analysis()['sharperatio'])
            sqn_data = thestrat.analyzers.sqn.get_analysis()
            print('SQN:', sqn_data['sqn'])
            print('Trades:', sqn_data['trades'])
            # print('VWR:', thestrat.analyzers.vwr.get_analysis()['vwr'])
            cerebro.plot(start=start_datetime, end=end_datetime)

    # cerebro.plot(start=start_datetime, end=end_datetime)
    # thestrats = cerebro.run()
    # thestrat = thestrats[0]
    # stats = thestrat.analyzers.basicstats.get_analysis()
    # cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    # cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade')

    # thestrat = thestrats[0]
    # print('Sharpe Ratio:', thestrat.analyzers.sharpe.get_analysis())
    # print('Trade analyzer:')
    # print(yaml.dump(thestrat.analyzers.trade.get_analysis()))
    # pprint.pprint(thestrat.analyzers.trade.get_analysis())

    # stats = thestrat.analyzers.trade.get_analysis()
    # pprint.pprint(stats)
    # print(stats_df.to_string())
    # cerebro.addanalyzer(BasicTradeStats, _name='basicstats')
    # cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

    # results = cerebro.run()
    # strat_bs = results[0]
    # stats = strat_bs.analyzers.basicstats.get_analysis()
    # strat_bs.analyzers.basicstats.print()

    # strat_pf = results[0]
    # pyfoliozer = strat_pf.analyzers.getbyname('pyfolio')
    # returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
    #
    # # pyfolio showtime
    # import pyfolio as pf
    # pf.create_full_tear_sheet(
    #     returns,
    #     positions=positions,
    #     transactions=transactions,
    #     # live_start_date=from_date,  # This date is sample specific
    #     round_trips=True)

    # cerebro.plot(start=start_datetime, end=end_datetime)  # Draw a chart

    # b = Bokeh(style='bar', plot_mode='single', scheme=Tradimo())
    # cerebro.plot(b)
