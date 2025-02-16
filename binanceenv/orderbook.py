import copy

import datetime
import numpy as np
from typing import List, Union, Tuple, Optional
from collections import namedtuple

Order = namedtuple('Order', 'OrderType size price order_commission order_cash order_datetime')
Bal = namedtuple('Bal', 'size cost price initial_datetime')

version = 0.021


class TargetCash:
    def __init__(self,
                 symbol: str = 'USDT',
                 initial_cash: float = 100_000.,
                 minimum_trade: float = 5.,
                 maximum_trade: float = 100.,
                 scale_decay: int = 100_0000,
                 use_period: str = 'train'):
        self.symbol = symbol
        self.use_period = use_period
        self.max_cash = initial_cash
        self.minimum_trade = minimum_trade
        self.maximum_trade = maximum_trade
        self.scale_decay = scale_decay
        if self.use_period == 'train':
            self.initial_cash = self.random_starting_cash()
            self.reset_func = self._train_reset
        else:
            self.initial_cash = self.max_cash
            self.reset_func = self._test_reset
        self.cash = initial_cash
        self.scaler = self.simple_scaler

    def simple_scaler(self, value):
        return value / self.scale_decay

    @property
    def scaled_cash(self):
        return self.scaler(self.cash)

    def set_scaler(self, ref):
        self.scaler = ref

    def random_starting_cash(self) -> float:
        return float(np.random.randint(max(int(self.minimum_trade * 5), int(self.max_cash // 8)), int(self.max_cash)))

    def _train_reset(self):
        if np.random.rand() > 0.11:
            self.initial_cash = self.random_starting_cash()
            self.cash = self.initial_cash
        else:
            self.initial_cash = float(self.max_cash)
            self.cash = self.initial_cash

    def _test_reset(self):
        self.cash = self.initial_cash

    def reset(self):
        self.reset_func()


class Balance:
    def __init__(self, target_obj: TargetCash,
                 scaler_method,
                 initial_balance: Bal
                 ):
        self.initial_balance: Bal = initial_balance
        self.size, self.cost, self.price, self.last_datetime = copy.deepcopy(initial_balance)
        self.target = target_obj
        self.scaler = scaler_method
        self.scaled_arr = self.calc_scaled_arr()

    def reset(self, initial_balance: tuple):
        self.initial_balance = initial_balance
        self.size, self.cost, self.price, self.last_datetime = copy.deepcopy(initial_balance)
        self.scaled_arr = self.calc_scaled_arr()

    def calc_scaled_arr(self):
        return np.clip(np.array([self.scaler(self.size),
                                 self.target.scaler(self.cost),
                                 self.target.scaler(self.price)],
                                dtype=np.float32),
                       a_min=0.,
                       a_max=np.inf)

    def __str__(self):
        return f'size={self.size}, cost={self.cost}, price={self.price}'


class Asset:
    def __init__(self,
                 target_obj: TargetCash,
                 commission: float,
                 minimum_trade: float,
                 symbol='BTC',
                 initial_balance: tuple = (0., 0., 0., datetime.datetime.utcnow()),
                 scale_decay: int = 10,
                 ):
        self.symbol = symbol
        self.target = target_obj
        self.initial_balance = Bal(*initial_balance)
        self.scale_decay = scale_decay
        self.scaler = self.simple_scaler
        self.balance = Balance(target_obj=target_obj, scaler_method=self.scaler, initial_balance=self.initial_balance)
        self.commission = commission
        self.minimum_trade = minimum_trade
        self.orders = OrdersBook(asset_obj=self)
        self.trades = self.orders.trades
        self.initial_total_in_cash = self.target.initial_cash + (self.initial_balance.size * self.initial_balance.price)

    def simple_scaler(self, value) -> np.float32:
        return value / self.scale_decay

    def set_scaler(self, ref):
        self.scaler = ref

    def reset(self, initial_balance: tuple = (0., 0., 0., datetime.datetime.utcnow())):
        self.target.reset()
        self.initial_balance = Bal(*initial_balance)
        self.balance.reset(initial_balance)
        self.trades.reset()
        self.orders.reset()
        self.initial_total_in_cash = self.target.initial_cash + (self.initial_balance.size * self.initial_balance.price)

    def __str__(self):
        return f'symbol={self.symbol}, size={self.balance.size}, cost={self.balance.cost}, price={self.balance.price}'


class OrdersBook:
    def __init__(self, asset_obj: Asset):
        self.asset: Asset = asset_obj
        self.target = self.asset.target
        self.symbol = self.asset.symbol
        self.commission = self.asset.commission
        self.minimum_trade = self.asset.minimum_trade
        self.balance = self.asset.balance
        self.book: List[Order,] = []
        self.last_index = 0
        self.__last_order: Optional[Order] = None
        self.trades = TradesBook(asset_obj=asset_obj)

    def buy(self, size, price, order_datetime):
        size_price = price * size
        order_commission = size_price * self.commission
        """ add commission to order cost """
        order_cash = -(size_price + order_commission)
        self.target.cash += order_cash
        self.book.append(Order('buy', size, price, order_commission, order_cash, order_datetime))
        self.__last_order = self.book[-1]
        self.trades.open_trade(self.book[-1])
        self.recalc_balance()

    def sell(self, size, price, order_datetime):
        size_price = price * size
        order_commission = size_price * self.commission
        """ subtract commission from profit """
        order_cash = size_price - order_commission
        self.target.cash += order_cash
        self.book.append(Order('sell', size, price, order_commission, order_cash, order_datetime))
        self.__last_order = self.book[-1]
        self.trades.close_trade(self.book[-1])
        self.recalc_balance()

    @property
    def last_order(self) -> Union[Order, None]:
        return self.__last_order

    def recalc_balance(self):
        """
        Use for long positions _only_
        Returns:
            None
        """
        last_index = self.last_index
        if self.last_index < len(self.book):
            for ix in range(self.last_index, len(self.book)):
                if self.book[ix].OrderType == 'buy':
                    self.balance.size += self.book[ix].size
                    """
                    already have order_commission for buying in .order_cash 
                    """
                    self.balance.cost += abs(self.book[ix].order_cash)
                    """ 
                    balance 'price' contains added commission to 'Sell', 
                    this helps RL to understand best price for sell order 
                    """
                    if self.balance.size > 0:
                        self.balance.price = self.balance.cost / self.balance.size
                    else:
                        self.balance.price = 0
                    self.balance.last_datetime = self.book[ix].order_datetime
                elif self.book[ix].OrderType == 'sell':
                    """  balance.cost minus order_cash for 'Sell' order """
                    self.balance.size -= self.book[ix].size
                    if not self.balance.size:
                        self.balance.price = 0.
                    self.balance.cost = (self.balance.size * self.balance.price)
                    self.balance.last_datetime = self.book[ix].order_datetime
                last_index = ix
            self.last_index = last_index + 1
            self.balance.scaled_arr = self.balance.calc_scaled_arr()

    def show(self):
        print(self.book)

    def reset(self):
        self.__last_order = None
        self.book.clear()
        self.last_index = 0


"""
entry_datetime    - orders.open_order.order_datetime  
exit_datetime    - orders.close_order.order_datetime
size                - orders.close_order.size
entry_price       - orders.open_order.price 
exit_price       - orders.close_order.price 
# opening_commission  - orders.open_order.order_commission 
# closing_commission  - orders.close_order.order_commission  
profit        - trade profit =    ((closing_price*size) - closing_commission) - 
                                        ((opening_price*size) - opening_commission)
"""
TradeOrders = namedtuple('TradeOrders', ['open_order', 'close_order'])


class Trade:
    def __init__(self):
        self.orders: TradeOrders = TradeOrders(None, None)
        self.entry_datetime = None
        self.exit_datetime = None
        self.entry_price = None
        self.exit_price = None
        self.size: float = .0
        self.total_commission: float = .0
        self.profit: float = .0
        self.pnl: float = .0
        self.opened: bool = False
        self.closed: bool = False
        self.status: Optional[str] = None
        self.price_diff = 0.

    def open_trade(self, open_order: Order) -> None:
        self.orders = self.orders._replace(open_order=open_order)
        self.entry_datetime = copy.copy(self.orders.open_order.order_datetime)
        self.entry_price = copy.copy(self.orders.open_order.price)
        self.opened = True
        self.status = 'opened'

    def close_trade(self, close_order: Order) -> None:
        self.orders = self.orders._replace(close_order=close_order)
        self.exit_datetime = copy.copy(self.orders.close_order.order_datetime)
        self.exit_price = copy.copy(self.orders.close_order.price)
        self.closed = True
        self.finalize()

    def finalize(self) -> None:
        # Set the size based on the close_order
        self.size = self.orders.close_order.size

        # Calculate total_commission and profit
        if self.opened:
            # Calculate commission per unit for the open_order
            commission_per_unit = self.orders.open_order.order_commission / self.orders.open_order.size
            # Calculate cash per unit for the open_order
            cash_per_unit = abs(self.orders.open_order.order_cash / self.orders.open_order.size)

            # Total commission is the close_order commission plus the proportional open_order commission
            self.total_commission = self.orders.close_order.order_commission + (commission_per_unit * self.size)
            # Profit is the close_order cash minus the proportional open_order cash
            self.profit = self.orders.close_order.order_cash - (cash_per_unit * self.size)
            self.price_diff = (self.orders.close_order.price / self.orders.open_order.price) - 1
            self.pnl = self.profit / (cash_per_unit * self.size)
            # Set the status to 'closed'
            self.status = 'closed'

    def __str__(self):
        msg = (f'Trade: orders={self.orders},\n'
               f' size={self.size}, t.commission={self.total_commission}, profit={self.profit}, status={self.status}')
        return msg


class TradesBook:
    def __init__(self, asset_obj: Asset):
        self.asset = asset_obj
        self.book: List[Trade] = []
        self.__last_trade: Optional[Trade] = None

    def new_trade(self) -> None:
        self.book.append(Trade())
        self.__last_trade = self.book[-1]

    def open_trade(self, open_order) -> None:
        # TODO: rewrite to: if not self.book or self.book[-1].closed: -> for multibuy
        if not self.book or self.book[-1].status is not None:
            self.new_trade()
        self.book[-1].open_trade(open_order)

    def close_trade(self, close_order) -> None:
        if not self.book:
            """ 
            add open_order data with initial values, 
            if we have initial asset data for this symbol
            """
            size = self.asset.initial_balance.size
            price = self.asset.initial_balance.price
            initial_datetime = self.asset.initial_balance.initial_datetime
            size_price = price * size
            order_commission = size_price * self.asset.commission  # just add commission cos already paid before
            order_cash = - size_price
            self.asset.orders.book.insert(0, Order('Buy', size, price, order_commission, order_cash, initial_datetime))
            self.asset.orders.__last_order = self.asset.orders.book[-1]
            self.asset.trades.open_trade(self.asset.orders.book[0])
            """
            do not recalc balance cos we get data from initial balance
            """
            # self.asset.orders.recalc_balance()
        elif self.book[-1].closed:
            self.new_trade()
        self.book[-1].close_trade(close_order)

    @property
    def last_trade(self) -> Union[Trade, None]:
        return self.__last_trade

    @property
    def trades_qty(self) -> int:
        return len(self.book)

    @property
    def profit(self) -> float:
        return sum(trade.profit for trade in self.book)

    @property
    def win_rate(self) -> float:
        """ win rate calculation """
        num_profitable_trades = sum(1 for trade in self.book if trade.profit > 0)
        total_num_trades = len(self.book)
        if total_num_trades <= 1:
            return -0.5
        return num_profitable_trades / total_num_trades

    def show(self):
        for i, trade in enumerate(self.book):
            print(f'{i}. {trade}')
        print(self.profit)

    def reset(self) -> None:
        self.__last_trade = None
        self.book.clear()


if __name__ == '__main__':
    _target_obj = TargetCash(symbol='USDT', initial_cash=40_000., use_period='test')
    """ checking balance """

    check_balance = _target_obj.initial_cash
    check_commission = 0.

    ix = 0


    def show_order(action, size, price, order_datetime):
        global check_balance
        global check_commission
        global ix
        if action == 'buy':
            action_fn = asset.orders.buy
        else:
            action_fn = asset.orders.sell
        action_fn(size, price, order_datetime)
        print(f'{asset.orders.book[-1].OrderType}: order_cash: {asset.orders.book[-1].order_cash}, '
              f'Balance: {asset.balance}, Cash: {asset.target.cash}')
        print(f'{ix}. {check_balance} + {asset.orders.book[ix].order_cash} = ', end='')
        check_balance += asset.orders.book[ix].order_cash
        check_commission += asset.orders.book[ix].order_commission
        print(f'{check_balance} / {check_commission}')
        ix += 1


    print(f'Initial cash: {_target_obj.initial_cash}')
    asset = Asset(symbol='BTC', commission=.001, minimum_trade=0.00001, target_obj=_target_obj,
                  initial_balance=Bal(0.5, 20020, 40000, datetime.datetime.utcnow()))

    show_order('sell', 0.5, 40000, datetime.datetime.utcnow())
    show_order('buy', 0.5, 30000, datetime.datetime.utcnow())
    show_order('sell', 0.5, 30000, datetime.datetime.utcnow())
    show_order('buy', 0.5, 60000, datetime.datetime.utcnow())
    show_order('sell', 0.5, 60000, datetime.datetime.utcnow())
    show_order('buy', 0.5, 30000, datetime.datetime.utcnow())
    show_order('sell', 0.5, 45000, datetime.datetime.utcnow())

    print(f'{asset.target.cash}')
    asset.orders.show()
    asset.trades.show()
    print(asset.balance.size, asset.balance.price, asset.balance.cost)
    print(asset.balance.scaled_arr)
    print(asset.trades.win_rate)
    #
    # print('Cash:', _target_obj.cash)
    # print('Cash + 10000:', asset.target.cash + 10000)
    #
    # print(asset.balance)
    # print(asset.balance.scaled_arr)
    # asset.orders.buy(1., 10000)
    # print(asset.balance.scaled_arr)
    # asset.orders.buy(1., 10000)
    # print(asset.balance)
    # asset.orders.show()
    # print(asset.balance)
    # print(asset.balance.scaled_arr)
    # asset.orders.buy(1., 40000)
    # print(asset.balance)
    # print(asset.balance.scaled_arr)
    #
    # asset.orders.show()
    # asset.reset((0.5, 67000, 37500))
    # asset.orders.show()
    # print(asset.balance)
    # print(asset.initial_balance)
