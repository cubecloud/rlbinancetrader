import numpy as np
import numba
from numba import jit
from typing import Tuple, Union, Dict

from gymnasium.spaces import Discrete
from gymnasium.spaces import Box

actions_dict: dict = {"Buy": 0, "Sell": 1, "Hold": 2}
actions_reversed_dict: dict = {0: "Buy", 1: "Sell", 2: "Hold"}


class IndicatorsSpace:
    def __init__(self, ind_num):
        self.__observation_space = Box(low=0.0, high=1.0, shape=(ind_num,), dtype=np.float32, seed=42)
        self.name = 'indicators'

    @property
    def observation_space(self):
        return self.__observation_space

    @observation_space.setter
    def observation_space(self, value):
        self.__observation_space = value


class IndicatorsAndAssetsSpace:
    def __init__(self, ind_num, assets_num):
        # low = np.zeros((assets_num + ind_num,))
        # high = np.ones((assets_num + ind_num,))
        # low[:assets_num] = 0.0
        # high[:assets_num] = 1.0
        self.__observation_space = Box(low=0.0, high=1.0, shape=(ind_num + assets_num,), dtype=np.float32, seed=42)
        # self.__observation_space = Box(low=low, high=high, dtype=np.float32, seed=42)
        self.name = 'indicators_assets'

    @property
    def observation_space(self):
        return self.__observation_space

    @observation_space.setter
    def observation_space(self, value):
        self.__observation_space = value


class AssetsCloseIndicatorsSpace:
    def __init__(self, ind_num, assets_num):
        # low = np.zeros((assets_num + ind_num,))
        # high = np.ones((assets_num + ind_num,))
        # low[:assets_num] = 0.0
        # high[:assets_num] = 1.0
        self.__observation_space = Box(low=0.0, high=1.0, shape=(ind_num + assets_num + 1,), dtype=np.float32, seed=42)
        # self.__observation_space = Box(low=low, high=high, dtype=np.float32, seed=42)
        self.name = 'assets_close_indicators'

    @property
    def observation_space(self):
        return self.__observation_space

    @observation_space.setter
    def observation_space(self, value):
        self.__observation_space = value


class IndicatorsAndPNLSpace:
    def __init__(self, ind_num, pnl_num):
        low = np.zeros((pnl_num + ind_num,))
        high = np.ones((pnl_num + ind_num,))
        low[:pnl_num] = -1.0
        high[:pnl_num] = 3.0
        self.__observation_space = Box(low=low, high=high, dtype=np.float32, seed=42)
        self.name = 'indicators_pnl'

    @property
    def observation_space(self):
        return self.__observation_space

    @observation_space.setter
    def observation_space(self, value):
        self.__observation_space = value


class DiscreteActionSpace:
    def __init__(self, n_action):
        self.__action_space = Discrete(n_action, seed=42)  # {0, 1, 2}
        self.name = 'discrete'

    def convert2action(self, action, masked_actions=None):
        amount = 1.
        if masked_actions is None:
            act = np.argmax(action)
        else:
            act = np.ma.masked_array(action, mask=~masked_actions, fill_value=-np.inf).argmax(axis=0)
        return act, amount

    @property
    def action_space(self):
        return self.__action_space

    @action_space.setter
    def action_space(self, value):
        self.__action_space = value


class SellBuyHoldAmount:
    def __init__(self, actions):
        # self.__action_space = Box(low=0, high=1., shape=(assets_qty,), dtype=np.float32)
        self.__action_space: Dict = {"Action": Discrete(actions),
                                     "Amount": Box(low=0, high=1, shape=(1,), dtype=np.float32)}
        self.name = 'sell_buy_hold_amount'

    @staticmethod
    def __get_action_amount(action: int) -> Tuple[int, float]:
        if action < 0:
            amount = abs(action)
            action = actions_dict['Sell']
        elif action > 0:
            amount = action
            action = actions_dict['Buy']
        else:
            amount = 0
            action = actions_dict['Hold']
        return action, amount

    def convert2action(self, action, masked_actions=None):
        return self.__get_action_amount(action)

    @property
    def action_space(self):
        return self.__action_space

    @action_space.setter
    def action_space(self, value):
        self.__action_space = value


class BoxActionSpace:
    def __init__(self, n_action):
        self.__action_space = Box(low=0, high=1, shape=(n_action,), dtype=np.float32)
        self.name = 'box'

    def convert2action(self, action, masked_actions=None):
        if masked_actions is None:
            return np.argmax(action)
        else:
            return np.ma.masked_array(action, mask=~masked_actions, fill_value=-np.inf).argmax(axis=0)

    @property
    def action_space(self):
        return self.__action_space

    @action_space.setter
    def action_space(self, value):
        self.__action_space = value


class BoxExtActionSpace:
    def __init__(self, n_action):
        self.__action_space = Box(low=-1., high=1, shape=(n_action,), dtype=np.float32)
        self.name = 'box1_1'

    def convert2action(self, action, masked_actions=None):
        if masked_actions is None:
            act = np.argmax(action)
            amount = abs(action[act])
        else:
            act = np.ma.masked_array(action, mask=~masked_actions, fill_value=-np.inf).argmax(axis=0)
            amount = abs(action[act])
        return act, amount

    @property
    def action_space(self):
        return self.__action_space

    @action_space.setter
    def action_space(self, value):
        self.__action_space = value


class ActionsBins:
    def __init__(self, box, n_actions):
        self.box_range = (np.max(box) - np.min(box))
        self.n_actions = n_actions
        self.step = self.box_range / n_actions
        self.bins = np.arange(box[0], box[1] + 1e-7, step=self.step, dtype=np.float32)
        self.pairs = [(self.bins[ix - 1], self.bins[ix]) for ix in range(1, len(self.bins))]

    def bins_2actions(self, value) -> Tuple[int, float]:
        amount = 0
        for ix in range(len(self.pairs)):
            if np.min(self.pairs[ix]) <= value <= np.max(self.pairs[ix]):
                amount = abs(value - np.max(self.pairs[ix])) / self.step if value < 0 and (
                        np.min(self.pairs[ix]) < 0 and np.max(self.pairs[ix]) < 0) else abs(value - np.min(
                    self.pairs[ix])) / self.step
                break
        return ix, amount


class BinBoxActionSpace:
    def __init__(self, n_action, low=-1., high=1.):
        self.__action_space = Box(low=low, high=high, shape=(1,), dtype=np.float32)
        self.actions_bins_obj = ActionsBins([low, high], n_action)
        self.name = 'binbox'

    def convert2action(self, action, masked_actions=None):
        return self.actions_bins_obj.bins_2actions(action)

    @property
    def action_space(self):
        return self.__action_space

    @action_space.setter
    def action_space(self, value):
        self.__action_space = value
