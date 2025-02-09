import sys
import random
import numpy as np
import numba
from numba import jit
from typing import Tuple, Union, Dict
from gymnasium import spaces
from dbbinance.fetcher.datautils import minmax_normalization_1_1
from binanceenv.actionspace import (actions_dict,
                                    actions_4_dict,
                                    actions_4_reversed_dict,
                                    actions_4_spot_dict,
                                    actions_4_spot_reversed_dict)

__version__ = 0.016


def get_action_space_obj(action_type='discrete'):
    if action_type == 'discrete':
        action_space_obj = DiscreteActionSpace(n_action=3)
    elif action_type == 'discrete_4':
        action_space_obj = DiscreteActionSpaceSpot(n_action=4)
    elif action_type == 'box':
        action_space_obj = BoxActionSpace(n_action=3)
    elif action_type == 'box_4':
        action_space_obj = BoxActionSpace(n_action=4)
    elif action_type == 'box1_1_3':
        action_space_obj = BoxExtActionSpace(n_action=3)
    elif action_type == 'box1_1_4':
        action_space_obj = BoxExtActionSpace(n_action=4)
    elif action_type == 'binbox':
        action_space_obj = BinBoxActionSpace(n_action=3, low=-1, high=1)
    elif action_type == 'sell_buy_hold_amount':
        action_space_obj = SellBuyHoldAmount()
    else:
        sys.exit(f'Error: Unknown action type {action_type}!')

    return action_space_obj


class IndicatorsSpace:
    def __init__(self, ind_num):
        self.__observation_space = spaces.Box(low=0.0, high=1.0, shape=(ind_num,), dtype=np.float32, seed=42)
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
        self.__observation_space = spaces.Box(low=0.0, high=1.0, shape=(ind_num + assets_num,), dtype=np.float32,
                                              seed=42)
        # self.__observation_space = Box(low=low, high=high, dtype=np.float32, seed=42)
        self.name = 'indicators_assets'

    @property
    def observation_space(self):
        return self.__observation_space

    @observation_space.setter
    def observation_space(self, value):
        self.__observation_space = value


class AssetsCloseIndicatorsSpace:
    def __init__(self, ind_num, assets_num, low=0.0, high=1.0):
        self.__observation_space = spaces.Box(low=low,
                                              high=high,
                                              shape=(ind_num + assets_num + 1,),
                                              dtype=np.float32,
                                              seed=42)
        self.name = 'assets_close_indicators'

    @property
    def observation_space(self):
        return self.__observation_space

    @observation_space.setter
    def observation_space(self, value):
        self.__observation_space = value


class LookbackAssetsCloseIndicatorsSpace:
    def __init__(self, ind_num, assets_data, lookback, low=0.0, high=1.0):
        self.__observation_space = spaces.Box(low=low,
                                              high=high,
                                              shape=((ind_num + assets_data + 1) * lookback,),
                                              dtype=np.float32,
                                              seed=42)
        self.name = 'lookback_assets_close_indicators'

    @property
    def observation_space(self):
        return self.__observation_space

    @observation_space.setter
    def observation_space(self, value):
        self.__observation_space = value


class LookbackAssetsCloseIndicatorsSpaceCNN:
    def __init__(self, ind_num, assets_data, lookback, low=0.0, high=1.0):
        self.__observation_space = spaces.Box(low=low,
                                              high=high,
                                              shape=(lookback, assets_data + ind_num),
                                              dtype=np.float32,
                                              seed=42)
        self.name = 'lookback_assets_close_indicators'

    @property
    def observation_space(self):
        return self.__observation_space

    @observation_space.setter
    def observation_space(self, value):
        self.__observation_space = value


class LookbackAssetsCloseIndicatorsActionSpace:
    def __init__(self, ind_num, assets_data, lookback, actions, low=0.0, high=1.0):
        self.__observation_space = spaces.Box(low=low,
                                              high=high,
                                              shape=(lookback, ind_num + assets_data + 1 + actions),
                                              dtype=np.float32,
                                              seed=42)
        self.name = 'lookback_assets_close_indicators_action'

    @property
    def observation_space(self):
        return self.__observation_space

    @observation_space.setter
    def observation_space(self, value):
        self.__observation_space = value


class LookbackDictOHLCAssetsIndicatorsSpace:
    def __init__(self, ind_num, assets_num, lookback):
        self.__observation_space = spaces.Dict(
            {"assets": spaces.Box(low=0.0, high=1.0, shape=(5 * assets_num,),
                                  dtype=np.float32, seed=42),
             "ohlc": spaces.Box(low=0.0, high=1.0, shape=(4, lookback), dtype=np.float32,
                                seed=42),
             "indicators": spaces.Box(low=0.0, high=1.0, shape=(ind_num, lookback),
                                      dtype=np.float32, seed=42),
             })
        self.name = 'lookback_dict'

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
        self.__observation_space = spaces.Box(low=low, high=high, dtype=np.float32, seed=42)
        self.name = 'indicators_pnl'

    @property
    def observation_space(self):
        return self.__observation_space

    @observation_space.setter
    def observation_space(self, value):
        self.__observation_space = value


class DiscreteActionSpace:
    def __init__(self, n_action):
        self.n_action = n_action
        self.__action_space = spaces.Discrete(n_action, seed=42)  # {0, 1, 2}
        self.name = 'discrete'
        self.actions_keys = np.array(list(actions_4_reversed_dict.keys()), dtype=int)

    def _check_masked(self, action, masked_actions):
        if action in self.actions_keys[masked_actions]:
            return action

        for act in range(self.n_action - 1, -1, -1):
            if masked_actions[act]:
                return act

    def convert2action(self, action: Union[np.ndarray, list], masked_actions=None):
        amount = 1.
        if masked_actions is not None:
            action = self._check_masked(action, masked_actions)
        return action, amount

    @property
    def action_space(self):
        return self.__action_space

    @action_space.setter
    def action_space(self, value):
        self.__action_space = value


class DiscreteActionSpaceSpot(DiscreteActionSpace):
    def __init__(self, n_action):
        super().__init__(n_action)
        self.name = 'discrete_spot'
        self.actions_keys = np.array(list(actions_4_spot_reversed_dict.keys()), dtype=int)

    def _check_masked(self, action, masked_actions):
        if action in self.actions_keys[masked_actions]:
            return action
        # return last possible action if action not correct
        for act in range(self.n_action - 1, -1, -1):
            if masked_actions[act]:
                return act


class BoxActionSpace:
    def __init__(self, n_action):
        self.n_action = n_action
        self.__action_space = spaces.Box(low=0, high=1, shape=(n_action,), dtype=np.float32)
        self.name = 'box'

    def convert2action(self, action, masked_actions=None) -> Tuple[float, float]:
        if masked_actions is None:
            act = np.argmax(action)
        else:
            act = np.ma.masked_array(action, mask=~masked_actions, fill_value=-np.inf).argmax(axis=0)
        amount = action[act]
        return act, amount

    @property
    def action_space(self):
        return self.__action_space

    @action_space.setter
    def action_space(self, value):
        self.__action_space = value


class BoxExtActionSpace:
    def __init__(self, n_action):
        self.__action_space = spaces.Box(low=-1., high=1, shape=(n_action,), dtype=np.float32)
        self.name = 'box1_1'

    @staticmethod
    def scale_amount(value):
        return (value - -1) / 2

    def convert2action(self, action, masked_actions=None):
        if masked_actions is None:
            act = np.argmax(action)
            amount = self.scale_amount(action[act])
        else:
            act = np.ma.masked_array(action, mask=~masked_actions, fill_value=-np.inf).argmax(axis=0)
            amount = self.scale_amount(action[act])
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
        ix = 1
        for ix in range(len(self.pairs)):
            if np.min(self.pairs[ix]) <= value <= np.max(self.pairs[ix]):
                amount = abs(value - np.max(self.pairs[ix])) / self.step if value < 0 and (
                        np.min(self.pairs[ix]) < 0 and np.max(self.pairs[ix]) < 0) else abs(value - np.min(
                    self.pairs[ix])) / self.step
                break
        if ix == 0:
            act = actions_dict['Sell']
        elif ix == 2:
            act = actions_dict['Buy']
        else:
            act = actions_dict['Hold']
        return act, amount


class SellBuyHoldAmount:
    def __init__(self):
        self.__action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        # self.__action_space = Box(low=-1., high=1., shape=(2,), dtype=np.float32)
        self.name = 'two_actions'

    def convert2action(self, action: np.ndarray, masked_actions=None):
        if action[1] < 0:
            act = actions_4_dict['Hold']
            amount = 0.
        else:
            amount = action[1]
            if action[0] < 0:
                act = actions_4_dict['Sell']
            else:
                act = actions_4_dict['Buy']
        if masked_actions is not None:
            if not masked_actions[act]:
                amount = 0.
                act = actions_4_dict['Hold']
        return act, amount

    @property
    def action_space(self):
        return self.__action_space

    @action_space.setter
    def action_space(self, value):
        self.__action_space = value


class BinBoxActionSpace:
    def __init__(self, n_action, low=-1., high=1.):
        self.__action_space = spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32)
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
