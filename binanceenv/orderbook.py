import numpy as np
from typing import List
from collections import namedtuple
import numba
from numba import jit

Order = namedtuple('Order', 'OrderType size price order_commission order_cash')
Bal = namedtuple('Bal', 'size cost price')
# BBalance = namedtuple('Balance', 'size cost price')

version = 0.009


# # @jit(nopython=True)
# def balance(book):
#     _asset_size: float = 0.
#     _asset_price: float = 0.
#     _asset_cost: float = 0.
#     for order in book:
#         if order.OrderType == 'buy':
#             _asset_size += order.size
#             _asset_cost += (order.size * order.price)
#             _asset_price = _asset_cost / _asset_size
#         elif order.OrderType == 'sell':
#             _asset_size -= order.size
#             _asset_cost = _asset_size * _asset_price
#     return Balance(_asset_size, _asset_cost, _asset_price)

class TargetCash:
    def __init__(self, symbol: str, initial_cash: float):
        self.symbol = symbol
        self.initial_cash = initial_cash
        self.cash = initial_cash

    def reset(self):
        self.cash = self.initial_cash


class Balance:
    def __init__(self, initial_balance: tuple = (0., 0., 0.)):
        self.__initial_balance = initial_balance
        self.size, self.cost, self.price = self.__initial_balance
        self.arr = np.asarray(list(self.__initial_balance), dtype=np.float32)

    def reset(self):
        self.size, self.cost, self.price = self.__initial_balance
        self.arr = np.asarray(list(self.__initial_balance), dtype=np.float32)

    def __str__(self):
        return f'size={self.size}, cost={self.cost}, price={self.price}'


class Asset:
    def __init__(self,
                 target_obj: TargetCash,
                 asset_commission: float,
                 asset_minimum_trade: float,
                 symbol='BTC',
                 initial_balance: tuple = (0., 0., 0.)):
        self.symbol = symbol
        self.target = target_obj
        self.initial_balance = Bal(*initial_balance)
        self.balance = Balance(initial_balance=initial_balance)
        self.orders = OrdersBook(symbol=self.symbol,
                                 asset_commission=asset_commission,
                                 asset_minimum_trade=asset_minimum_trade,
                                 target_obj=self.target,
                                 balance_obj=self.balance)

    def reset(self):
        self.target.reset()
        self.balance.reset()
        self.orders.reset()

    def __str__(self):
        return f'symbol={self.symbol}, size={self.balance.size}, cost={self.balance.cost}, price={self.balance.price}'


class OrdersBook:
    def __init__(self, symbol, asset_commission, asset_minimum_trade,
                 target_obj: TargetCash, balance_obj: Balance):
        self.target = target_obj
        self.symbol = symbol
        self.commission = asset_commission
        self.minimum_trade = asset_minimum_trade
        self.book: List[Order,] = []
        self.balance = balance_obj
        self.last_index = 0

    # @property
    # def cash(self):
    #     return self.target.cash
    #
    # @cash.setter
    # def cash(self, value):
    #     self.target.cash = value
    #
    # @property
    # def initial_cash(self):
    #     return self.target.initial_cash

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
            self.balance.arr = np.asarray([self.balance.size, self.balance.cost, self.balance.price], dtype=np.float32)

    def show(self):
        print(self.book)

    def reset(self):
        self.book: List[Order,] = []
        self.last_index = 0


if __name__ == '__main__':
    _target_obj = TargetCash(symbol='USDT', initial_cash=100_000)
    asset = Asset(symbol='BTC', asset_commission=.002, asset_minimum_trade=0.00001, target_obj=_target_obj)

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
    asset.reset()
    asset.orders.show()
    print(asset.balance)
    print(asset.initial_balance)
