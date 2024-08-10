import numpy as np
from typing import List
from collections import namedtuple
import numba
from numba import jit

Order = namedtuple('Order', 'OrderType size price order_commission order_cash')
# Balance = namedtuple('Balance', 'size cost price')
# BBalance = namedtuple('Balance', 'size cost price')

version = 0.007


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


class AssetBalance:
    def __init__(self, initial_balance: tuple = (0., 0., 0.)):
        self.__initial_balance = initial_balance
        self.book = []
        self.size, self.cost, self.price = self.__initial_balance
        self.arr = np.array(list(self.__initial_balance), dtype=np.float32)
        self.last_index = 0

    def recalc_balance(self):
        last_index = self.last_index
        if self.last_index < len(self.book):
            for ix in range(self.last_index, len(self.book)):
                if self.book[ix].OrderType == 'buy':
                    self.size += self.book[ix].size
                    self.cost += abs(self.book[ix].order_cash)
                    self.price = (self.cost / self.size) if self.size > 0 else 0
                elif self.book[ix].OrderType == 'sell':
                    self.size -= self.book[ix].size
                    self.cost = self.size * self.price
                last_index = ix
            self.last_index = last_index + 1
            self.arr = np.asarray([self.size, self.cost, self.price], dtype=np.float32)

    def reset(self):
        self.size, self.cost, self.price = self.__initial_balance
        self.last_index = 0
        self.book: List[Order] = []

    def __str__(self):
        return f'size={self.size}, cost={self.cost}, price={self.price}'


class TargetCash:
    def __init__(self, symbol: str, initial_cash: float):
        self.symbol = symbol
        self.initial_cash = initial_cash
        self.cash = initial_cash

    def reset(self):
        self.cash = self.initial_cash


class OrderBook:
    def __init__(self, asset_symbol, asset_commission, asset_minimum_trade, target_obj: TargetCash,
                 asset_initial_balance=(0., 0., 0.)):
        self.target = target_obj
        self.symbol = asset_symbol
        self.commission = asset_commission
        self.minimum_trade = asset_minimum_trade
        self.__asset_initial_balance = asset_initial_balance
        self.balance = AssetBalance(initial_balance=self.__asset_initial_balance)
        # self.book = self.balance.book
        self.initial_balance = AssetBalance(initial_balance=self.__asset_initial_balance)

    @property
    def cash(self):
        return self.target.cash

    @cash.setter
    def cash(self, value):
        self.target.cash = value

    @property
    def initial_cash(self):
        return self.target.initial_cash

    def buy(self, size, price):
        size_price = price * size
        order_commission = size_price * self.commission
        order_cash = -(size_price + order_commission)
        self.cash += order_cash
        self.balance.book.append(Order('buy', size, price, order_commission, order_cash))
        self.balance.recalc_balance()

    def sell(self, size, price):
        size_price = price * size
        order_commission = size_price * self.commission
        order_cash = size_price - order_commission
        self.cash += order_cash
        self.balance.book.append(Order('sell', size, price, order_commission, order_cash))
        self.balance.recalc_balance()

    def show(self):
        print(self.balance.book)

    # @property
    # def balance(self):
    #     return self.__balance

    # @property
    # def balance_arr(self):
    #     return self.__balance.arr

    def reset(self):
        self.balance.reset()
        self.target.reset()


if __name__ == '__main__':
    _target_obj = TargetCash(symbol='USDT', initial_cash=100_000)
    asset = OrderBook(asset_symbol='BTC', asset_commission=.002, asset_minimum_trade=0.00001, target_obj=_target_obj)

    asset.buy(1., 30000)
    print('Balance:', asset.balance)
    asset.sell(0.5, 30000)
    print('Balance:', asset.balance)
    asset.buy(1., 40000)
    print('Balance:', asset.balance)

    asset.sell(0.5, 30000)
    asset.sell(1., 40000)
    asset.show()
    print(asset.balance.book[0].OrderType)
    print(asset.balance.size, asset.balance.price)
    print(asset.target.cash)

    print('Cash:', _target_obj.cash)
    print('Cash + 10000:', asset.cash + 10000)

    print(asset.balance)
    print(asset.balance.arr)
    asset.buy(1., 10000)
    print(asset.balance.arr)
    asset.buy(1., 10000)
    print(asset.balance)
    asset.show()
    print(asset.balance)
    print(asset.balance.arr)
    asset.buy(1., 40000)
    print(asset.balance)
    print(asset.balance.arr)

    asset.show()
    asset.reset()
    asset.show()
    print(asset.balance)