import copy
import logging
import datetime
import pandas as pd
import backtrader as bt
from typing import List, Union, Any, Optional
from collections import deque

from dbbinance.fetcher.datafetcher import ceil_time
from ensembletools.modelstools.modelcard_v2 import ModelCard
from ensembletools.modelstools.predictionstore import RawPredictionHistory
from ensembletools.modelstools.predictiontracker import PredictionTracker
from ensembletools.modelstools.predictionstore import get_raw_ph_obj

from backtester.indicators.dbindicatorsparams import dbeyes_algoparams_1h, dbeyessimv1_algoparams_1h
from backtester.indicators.dbindicatorsparams import *
from backtester.indicators.dbindicators import DbIndicator, IndicatorLoaded

__version__ = 0.022

logger = logging.getLogger()


class MetaAIDBIndicator:
    # ind_params = dbeyessimv1t30_algoparams_59573
    # ind_params = dbeyessimv1t30_algoparams_123966
    # ind_params = dbeyessimt5d15_algoparams_40326
    ind_params = dbeyessimt5d15_algoparams_232552
    directions_lst: list = ['plus', 'minus']
    initialized = False
    count = 0

    def __init__(self):
        """
        Meta AI DataBase Indicator
        """
        MetaAIDBIndicator.count += 1
        self.id_num = int(self.count)
        self.initialized = True
        self.discretization = '30m'
        self.timeframe = '30m'
        self.prediction_trackers_dict: dict = {}
        self.raw_prediction_history_obj: RawPredictionHistory = get_raw_ph_obj()
        logger.debug(
            f'{self.__class__.__name__} #{self.id_num}: -> raw prediction history object '
            f'{self.raw_prediction_history_obj} #{self.raw_prediction_history_obj.idnum} initialized')

        self.indicators: dict = {}
        self.direction = None
        for direction in self.directions_lst:
            self.indicators[direction] = dict()

        self.indicators_objs: List[object,] = []
        self.total_indicator: dict = {}
        self.end_datetime = None
        self.state = 1  # 0 - Live data, 1 - History data, 2 - None
        self.previous_state = int(self.state)
        self.__current_datetime: Union[datetime.datetime, None] = None
        self.__previous_current_datetime = None

        self._signal_plus = .0
        self._signal_plus_target_time: Union[datetime.datetime, None] = None
        self._signal_plus_kwargs: dict = {}
        self._signal_minus = .0
        self._signal_minus_target_time: Union[datetime.datetime, None] = None
        self._signal_minus_kwargs: dict = {}

    def get_result(self,
                   dt: datetime.datetime,
                   direction: str, state: int = 1,
                   timeframe: str = '15m', discretization: str = '15m',
                   end_datetime: Union[datetime.datetime or None] = None):
        if not self.initialized:
            # initialize the object, initialization steps here
            self.__init__()
        # rest of the __call__ method logic goes here
        self.state = state  # 0 - Live data, 1 - History data, 2 - None
        self.direction = direction
        self.timeframe = timeframe
        self.discretization = discretization
        self.end_datetime = end_datetime
        # place important variables before next line cos it activate @setter
        self.current_datetime = dt
        if self.direction in self.directions_lst:
            return self.get_direction_data()

    def init_dbindicators(self):
        self.indicators_objs: List[object,] = []
        for ix, (model_UUID, params) in enumerate(self.ind_params.items()):
            try:
                _ = params['switch']
            except KeyError:
                if model_UUID in self.directions_lst:
                    self.total_indicator[model_UUID] = params
                continue
            if params['switch']:
                _model_card: ModelCard = self.raw_prediction_history_obj.get_card(model_UUID)
                pt_obj_kwargs: dict = {'symbol': _model_card.symbol, 'market': _model_card.market}
                pt_obj_key = tuple(sorted(pt_obj_kwargs.items()))
                if pt_obj_key in self.prediction_trackers_dict.keys():
                    pt_obj = self.prediction_trackers_dict[pt_obj_key]
                else:
                    pt_obj = PredictionTracker(symbol=f'{_model_card.symbol}',
                                                market=f'{_model_card.market}',
                                                raw_ph_obj=self.raw_prediction_history_obj
                                                )
                    self.prediction_trackers_dict[pt_obj_key] = pt_obj
                if not self.state:  # 0 - Live data, 1 - History data, 2 - None
                    indicator_obj = DbIndicator(model_uuid=model_UUID,
                                                prediction_tracker_obj=pt_obj)
                    indicator_obj.discretization = self.discretization
                    indicator_obj.timeframe = self.timeframe
                else:
                    indicator_obj = IndicatorLoaded(model_uuid=model_UUID,
                                                    prediction_tracker_obj=pt_obj)
                    if self.end_datetime is None:
                        _end_datetime = ceil_time(datetime.datetime.utcnow(), ceil_to=self.discretization)
                    else:
                        _end_datetime = self.end_datetime
                    indicator_obj.discretization = self.discretization
                    indicator_obj.timeframe = self.timeframe

                    indicator_obj.preload_indicator(self.current_datetime,
                                                    _end_datetime)
                indicator_obj.indicator_id = ix
                indicator_obj.algoparams = params
                self.indicators_objs.append(indicator_obj)
                if indicator_obj.direction is None:
                    self.indicators['other'].update({indicator_obj.name: indicator_obj})
                else:
                    if indicator_obj.direction == 'plus':
                        self.indicators['plus'].update({indicator_obj.name: indicator_obj})
                    elif indicator_obj.direction == 'minus':
                        self.indicators['minus'].update({indicator_obj.name: indicator_obj})

    @property
    def current_datetime(self):
        return self.__current_datetime

    @current_datetime.setter
    def current_datetime(self, dt: datetime.datetime):
        self.__current_datetime = dt
        self.check_init()

    @property
    def signal_plus(self):
        return self._signal_plus

    @property
    def signal_minus(self):
        return self._signal_minus

    @property
    def signal_plus_target_time(self):
        return self._signal_plus_target_time

    @property
    def signal_minus_target_time(self):
        return self._signal_minus_target_time

    @property
    def signal_plus_kwargs(self):
        return self._signal_plus_kwargs

    @property
    def signal_minus_kwargs(self):
        return self._signal_minus_kwargs

    def check_init(self):
        # the state is changed
        if self.state != self.previous_state:
            logger.info(f'{self.__class__.__name__} #{self.id_num}: Changing state')
            self.init_dbindicators()
            self.previous_state = int(self.state)
        elif not self.indicators_objs:
            self.init_dbindicators()

    def get_direction_data(self):
        # self._signal_reset()
        if self.current_datetime is not None:
            """ Updating current datetime and indicators to get actual predictions """
            for indicator_obj in self.indicators[self.direction].values():
                indicator_obj.current_datetime = self.current_datetime
            return self.direction_decision()

        # self.__previous_current_datetime = copy.deepcopy(self.__current_datetime)

    def _signals_check(self, direction: str) -> list:
        indicators_lst: list = []
        log_signals: list = []
        for _, indicator_obj in self.indicators[direction].items():
            if indicator_obj.indicator is not None:
                log_signals.append(
                    f'(nm:{indicator_obj.model_uuid[-3:]}_{indicator_obj.power_trend:.3f}_{indicator_obj.interval}, '
                    f'pr:{indicator_obj.prediction:.3f}, pw:{indicator_obj.power:.3f})')
                if (indicator_obj.prediction > indicator_obj.algoparams.threshold) and (
                        indicator_obj.power > indicator_obj.algoparams.power):
                    indicators_lst.append(indicator_obj)
            else:
                log_signals.append(
                    f'(nm:{indicator_obj.model_uuid[-3:]}_{indicator_obj.power_trend:.3f}_{indicator_obj.interval}, '
                    f'pr:None, pw:None)')
        logger.debug(
            f'{self.__class__.__name__} #{self.id_num}: {self.current_datetime} ' f'{direction} signals: {log_signals}')
        return indicators_lst

    def direction_decision(self):
        signal = .0
        signal_target_time: Union[datetime.datetime, None] = None
        signal_kwargs: dict = {}
        signals_indicators_objs = self._signals_check(self.direction)
        target_time = None
        if signals_indicators_objs:
            total_signal_prediction: float = .0
            total_signal_power = 0
            for indicator_obj in signals_indicators_objs:
                total_signal_prediction += indicator_obj.prediction
                total_signal_power += indicator_obj.power
                if self.current_datetime < indicator_obj.target_time:
                    if target_time is None or indicator_obj.target_time <= target_time:
                        target_time = indicator_obj.target_time
            if (total_signal_prediction >= self.total_indicator[self.direction]['threshold']) and (
                    total_signal_power >= self.total_indicator[self.direction]['power']):
                signal = float(total_signal_prediction)
                signal_target_time = ceil_time(target_time, ceil_to=self.discretization)
                signal_kwargs = self.total_indicator[self.direction]
        return signal, signal_target_time, signal_kwargs


meta_aidbindicator_obj = MetaAIDBIndicator()


class MarkersBase(bt.Indicator):
    AIEyes = meta_aidbindicator_obj

    def __init__(self):
        self.signal: float = .0
        self.target_time: Union[datetime.datetime, None] = None
        self.kwargs: dict = {}

    def get_result(self, dt, direction, state, timeframe, discretization, end_datetime: Optional) -> tuple:
        return self.AIEyes.get_result(dt, direction, state, timeframe,
                                      discretization, end_datetime)


class BuyAI(MarkersBase):
    direction = 'plus'
    lines = ('buyai',)
    alias = ('BUYAI',)
    params = (('state', 1),  # 0 - Live data, 1 - History data, 2 - None
              ('timeframe', '30m'),
              ('discretization', '30m'),
              ('end_datetime', None),
              )
    plotinfo = dict(plotymargin=0.05, plotyhlines=[0.0, 3.0])

    def __init__(self):
        self.state = self.p.state
        self.timeframe = self.p.timeframe
        self.discretization = self.p.discretization
        self.end_datetime = self.p.end_datetime
        super(MarkersBase, self).__init__()

    def next(self):
        self.state = self.data._state
        self.signal, self.target_time, self.kwargs = self.get_result(bt.num2date(self.data.datetime[0]),
                                                                     self.direction,
                                                                     self.state,
                                                                     self.timeframe,
                                                                     self.discretization,
                                                                     self.end_datetime)
        self.lines.buyai[0] = float(self.signal)


class SellAI(MarkersBase):
    direction = 'minus'
    lines = ('sellai',)
    alias = ('SELLAI',)
    params = (('state', 1),  # 0 - Live data, 1 - History data, 2 - None
              ('timeframe', '30m'),
              ('discretization', '30m'),
              ('end_datetime', None),
              )

    plotinfo = dict(plotymargin=0.05, plotyhlines=[0.0, 3.0])

    def __init__(self):
        self.state = self.p.state
        self.timeframe = self.p.timeframe
        self.discretization = self.p.discretization
        self.end_datetime = self.p.end_datetime
        super(MarkersBase, self).__init__()

    def next(self):
        self.state = self.data._state
        self.signal, self.target_time, self.kwargs = self.get_result(bt.num2date(self.data.datetime[0]),
                                                                     self.direction,
                                                                     self.state,
                                                                     self.timeframe,
                                                                     self.discretization,
                                                                     self.end_datetime)

        self.lines.sellai[0] = float(self.signal)
