import numpy as np
from typing import List
from collections import namedtuple
import numba
from numba import jit

Order = namedtuple('Order', 'OrderType size price order_commission order_cash')
Bal = namedtuple('Bal', 'size cost price')
# BBalance = namedtuple('Balance', 'size cost price')

version = 0.019


class TargetCash:
    def __init__(self, symbol: str = 'USDT',
                 initial_cash: float = 100_000.,
                 minimum_trade: float = 5.,
                 use_period: str = 'train'):
        self.symbol = symbol
        self.use_period = use_period
        self.max_cash = initial_cash
        self.minimum_trade = minimum_trade
        if self.use_period == 'train':
            self.initial_cash = self.random_starting_cash()
            self.reset_func = self._train_reset
        else:
            self.initial_cash = self.max_cash
            self.reset_func = self._test_reset

        self.cash = initial_cash

    def random_starting_cash(self) -> float:
        return float(np.random.randint(max(int(self.minimum_trade * 5), int(self.max_cash//4)), int(self.max_cash)))

    def _train_reset(self):
        self.initial_cash = self.random_starting_cash()
        self.cash = self.initial_cash

    def _test_reset(self):
        self.cash = self.initial_cash

    def reset(self):
        self.reset_func()


class Balance:
    def __init__(self, initial_balance: tuple = (0., 0., 0.)):
        self.size, self.cost, self.price = initial_balance
        self.arr = np.array(list(initial_balance), dtype=np.float32)

    def reset(self, initial_balance: tuple = (0., 0., 0.)):
        self.size, self.cost, self.price = initial_balance
        self.arr = np.array(list(initial_balance), dtype=np.float32)

    def __str__(self):
        return f'size={self.size}, cost={self.cost}, price={self.price}'


class Asset:
    def __init__(self,
                 target_obj: TargetCash,
                 commission: float,
                 minimum_trade: float,
                 symbol='BTC',
                 initial_balance: tuple = (0., 0., 0.),
                 ):
        self.symbol = symbol
        self.target = target_obj
        self.initial_balance = Bal(*initial_balance)
        self.balance = Balance(initial_balance=initial_balance)
        self.minimum_trade = minimum_trade
        self.orders = OrdersBook(symbol=self.symbol,
                                 commission=commission,
                                 minimum_trade=minimum_trade,
                                 target_obj=self.target,
                                 balance_obj=self.balance)

    def reset(self, initial_balance: tuple = (0., 0., 0.)):
        self.target.reset()
        self.initial_balance = Bal(*initial_balance)
        self.balance.reset(initial_balance)
        self.orders.reset()

    def __str__(self):
        return f'symbol={self.symbol}, size={self.balance.size}, cost={self.balance.cost}, price={self.balance.price}'


class OrdersBook:
    def __init__(self, symbol, commission, minimum_trade,
                 target_obj: TargetCash, balance_obj: Balance):
        self.target = target_obj
        self.symbol = symbol
        self.commission = commission
        self.minimum_trade = minimum_trade
        self.book: List[Order,] = []
        self.balance = balance_obj
        self.last_index = 0

    def buy(self, size, price):
        size_price = price * size
        order_commission = size_price * self.commission
        order_cash = -(size_price + order_commission)
        self.target.cash += order_cash
        self.book.append(Order('buy', size, price, order_commission, order_cash))
        self.recalc_balance()

    def sell(self, size, price):
        size_price = price * size
        order_commission = size_price * self.commission
        order_cash = size_price - order_commission
        self.target.cash += order_cash
        self.book.append(Order('sell', size, price, order_commission, order_cash))
        self.recalc_balance()

    def recalc_balance(self):
        last_index = self.last_index
        if self.last_index < len(self.book):
            for ix in range(self.last_index, len(self.book)):
                if self.book[ix].OrderType == 'buy':
                    self.balance.size += self.book[ix].size
                    self.balance.cost += abs(self.book[ix].order_cash)
                    self.balance.price = (self.balance.cost / self.balance.size) if self.balance.size > 0 else 0
                elif self.book[ix].OrderType == 'sell':
                    self.balance.size -= self.book[ix].size
                    self.balance.cost = self.balance.size * self.balance.price
                last_index = ix
            self.last_index = last_index + 1
            self.balance.arr = np.array([self.balance.size, self.balance.cost, self.balance.price], dtype=np.float32)

    def show(self):
        print(self.book)

    def reset(self):
        self.book: List[Order,] = []
        self.last_index = 0


if __name__ == '__main__':
    _target_obj = TargetCash(symbol='USDT', initial_cash=100_000.)
    asset = Asset(symbol='BTC', commission=.002, minimum_trade=0.00001, target_obj=_target_obj)

    asset.orders.buy(1., 30000)
    print('Balance:', asset.balance)
    asset.orders.sell(0.5, 30000)
    print('Balance:', asset.balance)
    asset.orders.buy(1., 40000)
    print('Balance:', asset.balance)

    asset.orders.sell(0.5, 30000)
    asset.orders.sell(1., 40000)
    asset.orders.show()
    print(asset.orders.book[0].OrderType)
    print(asset.balance.size, asset.balance.price)
    print(asset.target.cash)

    print('Cash:', _target_obj.cash)
    print('Cash + 10000:', asset.target.cash + 10000)

    print(asset.balance)
    print(asset.balance.arr)
    asset.orders.buy(1., 10000)
    print(asset.balance.arr)
    asset.orders.buy(1., 10000)
    print(asset.balance)
    asset.orders.show()
    print(asset.balance)
    print(asset.balance.arr)
    asset.orders.buy(1., 40000)
    print(asset.balance)
    print(asset.balance.arr)

    asset.orders.show()
    asset.reset((0.5, 67000, 37500))
    asset.orders.show()
    print(asset.balance)
    print(asset.initial_balance)
