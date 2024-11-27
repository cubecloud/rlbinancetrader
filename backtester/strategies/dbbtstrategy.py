import logging
import datetime
import pandas as pd
from typing import List, Any
from collections import deque
from backtesting import Strategy
from backtester.indicators.dbindicatorsparams import *
from backtester.indicators.dbindicators import DbIndicator, IndicatorLoaded
from backtester.indicators.btdbindicators import SellAI, BuyAI

from datawizard.datafetcher import ceil_time
from datawizard.modelcard_v2 import ModelCard
from datawizard.predictionstore import RawPredictionHistory, get_raw_ph_obj
from datawizard.predictionstore import PredictionsTracker

__version__ = 0.020

logger = logging.getLogger()


class DbEyes(Strategy):
    """
    Set indicators to working with PredictionHistoryDatabase
    """
    name = 'DbEyes'
    ind_params = dbeyes_algoparams_1h
    discretization = '1h'
    timeframe = '1h'

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.raw_prediction_history_obj = params.pop('raw_prediction_history_obj', None)
        if self.raw_prediction_history_obj is None:
            self.raw_prediction_history_obj: RawPredictionHistory = get_raw_ph_obj()
        logger.debug(
            f'{self.__class__.__name__}: -> raw prediction history object {self.raw_prediction_history_obj} '
            f'#{self.raw_prediction_history_obj.idnum} initialized')

        self.indicators: dict = {'plus': dict(),
                                 'minus': dict(),
                                 'other': dict()
                                 }
        self.indicators_objs: List[Any,] = []
        self.current_datetime = pd.to_datetime(self.data.index[-1])

        self.directions_lst: tuple = ('plus', 'minus')
        self.total_indicator: dict = {}
        self.price = None
        self.orders_stack = deque()
        self.datetime: datetime.datetime or None = None
        logger.debug(f"{self.__class__.__name__}: Length {self.data.df.shape[0]}")
        self.prediction_trackers_dict: dict = {}
        self.init_dbindicators()

    def init_dbindicators(self):
        for ix, (model_UUID, params) in enumerate(self.ind_params.items()):
            try:
                if params['switch']:
                    _model_card: ModelCard = self.raw_prediction_history_obj.get_card(model_UUID)
                    pt_obj_kwargs: dict = {'symbol': _model_card.symbol, 'market': _model_card.market}
                    pt_obj_key = tuple(sorted(pt_obj_kwargs.items()))
                    if pt_obj_key in self.prediction_trackers_dict.keys():
                        pt_obj = self.prediction_trackers_dict[pt_obj_key]
                    else:
                        pt_obj = PredictionsTracker(symbol=f'{_model_card.symbol}',
                                                    market=f'{_model_card.market}',
                                                    raw_ph_obj=self.raw_prediction_history_obj
                                                    )
                        self.prediction_trackers_dict[pt_obj_key] = pt_obj
                    indicator_obj = DbIndicator(model_uuid=model_UUID,
                                                prediction_tracker_obj=pt_obj)
                    indicator_obj.indicator_id = ix
                    indicator_obj.discretization = self.discretization
                    indicator_obj.timeframe = self.timeframe

                    indicator_obj.current_datetime = pd.to_datetime(self.data.index[-1])
                    indicator_obj.algoparams = params
                    self.indicators_objs.append(indicator_obj)
                    if indicator_obj.direction is None:
                        self.indicators['other'].update({indicator_obj.name: indicator_obj})
                    else:
                        if indicator_obj.direction == 'plus':
                            self.indicators['plus'].update({indicator_obj.name: indicator_obj})
                        else:
                            self.indicators['minus'].update({indicator_obj.name: indicator_obj})
            except:
                if model_UUID in self.directions_lst:
                    self.total_indicator[model_UUID] = params

    def init(self):
        pass
        # self.datetime = self.I(lambda x: x, self.data.index, plot=False, name='datetime')

    def _check_params(self, params):
        for k, v in params.items():
            setattr(self, k, v)
        return params

    def _update_data(self):
        self.current_datetime = pd.to_datetime(self.data.index[-1])
        for indicator_obj in self.indicators_objs:
            indicator_obj.current_datetime = self.current_datetime

    def _set_orders(self, todo: list):
        self.orders_stack.extend(todo)

    def _signals_check(self, direction: str) -> list:
        indicators_lst: list = []
        for ix, (indicator_name, indicator_obj) in enumerate(self.indicators[direction].items()):
            if indicator_obj.indicator is not None:
                if (indicator_obj.prediction > indicator_obj.algoparams.threshold) and (
                        indicator_obj.power > indicator_obj.algoparams.power):
                    indicators_lst.append(indicator_obj)
        return indicators_lst

    def next(self):
        """
        Logic: the decision about what 2do we get on current timeframe, and setup orders for next timeframes.

        Returns:
            None
        """
        """ Process orders from previous timeframe """
        self._update_data()
        # logger.debug(f"{self.__class__.__name__}: {self.current_datetime}")
        self.process_orders()
        todo = self.decision()
        if todo:
            self._set_orders(todo)

    def decision(self) -> list:
        todo_lst: list = []
        self.price = self.data.Close[-1]
        plus_signals = self._signals_check('plus')
        minus_signals = self._signals_check('minus')
        target_time = None
        if self.position.is_long:
            action = ()
            if minus_signals:
                for indicator_obj in minus_signals:
                    if self.current_datetime < indicator_obj.target_time:
                        if target_time is None or indicator_obj.target_time <= target_time:
                            target_time = indicator_obj.target_time
                            action = ('close',
                                      indicator_obj.target_time,
                                      indicator_obj.power_trend,
                                      indicator_obj.algoparams.kwargs,
                                      )
                if action:
                    todo_lst.extend([action])
        elif self.position.size == 0.0:
            if plus_signals:
                action = ()
                for indicator_obj in plus_signals:
                    if self.current_datetime < indicator_obj.target_time:
                        if target_time is None or indicator_obj.target_time <= target_time:
                            target_time = indicator_obj.target_time
                            action = ('buy',
                                      indicator_obj.target_time,
                                      indicator_obj.power_trend,
                                      indicator_obj.algoparams.kwargs,
                                      )
                if action:
                    todo_lst.extend([action])
        return todo_lst

    def process_orders(self):
        # торгуем по крайней цене закрытия
        current_price = self.data.Close[-1]
        counter = 0
        while counter < len(self.orders_stack):
            todo_order = self.orders_stack.popleft()
            counter += 1
            if ceil_time(todo_order[1], ceil_to=self.discretization) == self.current_datetime:
                if todo_order[0] == 'buy':
                    buy_kwargs: dict = {}
                    self.position.entry_price = current_price
                    sl = current_price - (current_price * todo_order[3]['stop_loss'])
                    tp = current_price + (current_price * todo_order[3]['take_profit'])
                    buy_kwargs.update({"sl": sl,
                                       "tp": tp})
                    self.buy(**buy_kwargs)
                elif todo_order[0] == 'close':
                    self.position.close()
            # if order target_datetime > current_datetime -> postpone order
            elif ceil_time(todo_order[1], ceil_to=self.discretization) > self.current_datetime:
                self.orders_stack.append(todo_order)


class DbEyesSim(DbEyes):
    """
    Set indicators to working with PredictionHistoryDatabase
    """
    name = 'DbEyesSim'

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)

    def init_dbindicators(self):
        for ix, (model_UUID, params) in enumerate(self.ind_params.items()):
            try:
                if params['switch']:
                    _model_card: ModelCard = self.raw_prediction_history_obj.get_card(model_UUID)
                    pt_obj_kwargs: dict = {'symbol': _model_card.symbol, 'market': _model_card.market}
                    pt_obj_key = tuple(sorted(pt_obj_kwargs.items()))
                    if pt_obj_key in self.prediction_trackers_dict.keys():
                        pt_obj = self.prediction_trackers_dict[pt_obj_key]
                    else:
                        pt_obj = PredictionsTracker(symbol=f'{_model_card.symbol}',
                                                    market=f'{_model_card.market}',
                                                    raw_ph_obj=self.raw_prediction_history_obj
                                                    )
                        self.prediction_trackers_dict[pt_obj_key] = pt_obj
                    indicator_obj = IndicatorLoaded(model_uuid=model_UUID,
                                                    prediction_tracker_obj=pt_obj)

                    indicator_obj.indicator_id = ix
                    indicator_obj.discretization = self.discretization
                    indicator_obj.timeframe = self.timeframe
                    indicator_obj.preload_indicator(pd.to_datetime(self.data.index[0]),
                                                    pd.to_datetime(self.data.index[-1]))
                    indicator_obj.current_datetime = pd.to_datetime(self.data.index[-1])
                    indicator_obj.algoparams = params
                    self.indicators_objs.append(indicator_obj)
                    if indicator_obj.direction is None:
                        self.indicators['other'].update({indicator_obj.name: indicator_obj})
                    else:
                        if indicator_obj.direction == 'plus':
                            self.indicators['plus'].update({indicator_obj.name: indicator_obj})
                        else:
                            self.indicators['minus'].update({indicator_obj.name: indicator_obj})
            except:
                if model_UUID in self.directions_lst:
                    self.total_indicator[model_UUID] = params

    def init(self):
        for indicators_obj in self.indicators_objs:
            setattr(self, indicators_obj.name,
                    self.I(lambda x: x, indicators_obj.prediction_show, plot=True, name=indicators_obj.name))


class DbEyesSimV1(DbEyesSim):
    """
    Set indicators to working with PredictionHistoryDatabase
    """
    name = 'DbEyesSimV1'
    ind_params = dbeyessimv1_algoparams_1h_t40
    discretization = '1h'

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)

    def decision(self) -> list:
        todo_lst: list = []
        # self.price = self.data.Close[-1]
        plus_signals = self._signals_check('plus')
        minus_signals = self._signals_check('minus')
        target_time = None
        if minus_signals:
            total_minus_signal_prediction = 0
            total_minus_signal_power = 0
            for indicator_obj in minus_signals:
                total_minus_signal_prediction += indicator_obj.prediction
                total_minus_signal_power += indicator_obj.power
                if self.current_datetime < indicator_obj.target_time:
                    if target_time is None or indicator_obj.target_time <= target_time:
                        target_time = indicator_obj.target_time
            if (total_minus_signal_prediction >= self.total_indicator['minus']['threshold']) and (
                    total_minus_signal_power >= self.total_indicator['minus']['power']):
                todo_lst.append(('close',
                                 ceil_time(target_time, ceil_to=self.discretization),
                                 self.total_indicator['minus'],
                                 ))
        if plus_signals:
            total_plus_signal_prediction = 0
            total_plus_signal_power = 0
            for indicator_obj in plus_signals:
                total_plus_signal_prediction += indicator_obj.prediction
                total_plus_signal_power += indicator_obj.power
                if self.current_datetime < indicator_obj.target_time:
                    if target_time is None or indicator_obj.target_time <= target_time:
                        target_time = indicator_obj.target_time
            if (total_plus_signal_prediction >= self.total_indicator['plus']['threshold']) and (
                    total_plus_signal_power >= self.total_indicator['plus']['power']):
                todo_lst.append(('buy',
                                 ceil_time(target_time, ceil_to=self.discretization),
                                 self.total_indicator['plus'],
                                 ))
        return todo_lst

    def process_orders(self):
        # торгуем по крайней цене закрытия
        current_price = self.data.Close[-1]
        counter = 0
        while counter < len(self.orders_stack):
            todo_order = self.orders_stack.popleft()
            counter += 1
            if todo_order[1] == self.current_datetime:
                if self.position.size == 0.0:
                    if todo_order[0] == 'buy':
                        buy_kwargs: dict = {}
                        self.position.entry_price = current_price
                        sl = current_price - (current_price * todo_order[2]['stop_loss'])
                        tp = current_price + (current_price * todo_order[2]['take_profit'])
                        buy_kwargs.update({"sl": sl,
                                           "tp": tp})
                        self.buy(**buy_kwargs)
                elif self.position.is_long:
                    if todo_order[0] == 'close':
                        self.position.close()
            # if order target_datetime > current_datetime -> postpone order
            elif todo_order[1] > self.current_datetime:
                self.orders_stack.append(todo_order)


class DbEyesSimV1T15(DbEyesSimV1):
    """
    Set indicators to working with PredictionHistoryDatabase
    """
    name = 'DbEyesSimV1T15'
    ind_params = dbeyessimv1t15_algoparams_40326
    discretization = '15m'
    timeframe = '15m'


class DbEyesSimV1T30(DbEyesSimV1):
    """
    Set indicators to working with PredictionHistoryDatabase
    """
    name = 'DbEyesSimV1T30'
    ind_params = dbeyessimv1t30_algoparams_59573
    discretization = '30m'
    timeframe = '30m'


