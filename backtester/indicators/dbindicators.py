import time
import logging
import datetime
import functools
import traceback
import pandas as pd

from datawizard.datautils import Constants
from dataclasses import dataclass, asdict
from datawizard.modelcard_v2 import ModelCard
from datawizard.predictionstore import RawPredictionHistory, PredictionsTracker, get_raw_ph_obj, get_agg_dict
from datawizard.datautils import check_convert_to_datetime, convert_timeframe_to_freq, get_timedelta_kwargs
from dateutil.relativedelta import *

__version__ = 0.015

logger = logging.getLogger()


def retry(retry_num, delay):
    """
    retry help decorator.
    :param retry_num: the retry num; retry sleep sec
    :return: decorator
    """

    def decorator(func):
        """decorator"""

        # preserve information about the original function, or the func name will be "wrapper" not "func"
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """wrapper"""
            for attempt in range(retry_num):
                try:
                    return func(*args, **kwargs)  # should return the raw function's return value
                except Exception as err:  # pylint: disable=broad-except
                    logger.error(err)
                    logger.error(traceback.format_exc())
                    time.sleep(delay)
                logger.error("Trying attempt %s of %s.", attempt + 1, retry_num)
            logger.error("func %s retry failed", func)
            raise Exception('Exceed max retry num: {} failed'.format(retry_num))

        return wrapper

    return decorator


@dataclass
class ModelAlgoParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


class DbIndicator:

    def __init__(self, model_uuid: str, prediction_tracker_obj: PredictionsTracker = None):
        self.model_uuid: str = model_uuid
        self.indicator_id: int = 0
        self.pt_obj = prediction_tracker_obj
        self.__model_card: ModelCard = self.pt_obj.raw_ph_obj.get_card(self.model_uuid)
        self.__timeframe = self.__model_card.interval
        self.__discretization = self.__model_card.interval
        self.__current_datetime: datetime.datetime or None = None
        self._indicator_data: pd.DataFrame or pd.Series or None = None
        self.__algo_params: ModelAlgoParams or None = None
        self.retry_counter: int = 0

    @property
    def algoparams(self):
        return self.__algo_params

    @algoparams.setter
    def algoparams(self, params: dict):
        self.__algo_params = ModelAlgoParams(**params)

    @property
    def power_trend(self):
        return self.__model_card.power_trend

    @property
    def interval(self):
        return self.__model_card.interval

    @property
    def target_steps(self):
        return self.__model_card.target_steps

    @property
    def market(self):
        return self.__model_card.market

    @property
    def symbol(self):
        return self.__model_card.symbol

    @property
    def model_type(self):
        return self.__model_card.model_type

    @property
    def direction(self):
        if self.__model_card.model_activator_value == 'plus' and self.model_type == 'classification':
            return 'plus'
        elif self.__model_card.model_activator_value == 'minus' and self.model_type == 'classification':
            return 'minus'
        else:
            return None

    @property
    def name(self):
        return f'{self.direction}_{self.power_trend}_{self.interval}_{self.discretization}'

    @property
    def discretization(self):
        return self.__discretization

    @discretization.setter
    def discretization(self, discretization: str):
        self.__discretization = discretization

    @property
    def timeframe(self):
        return self.__timeframe

    @timeframe.setter
    def timeframe(self, timeframe: str):
        self.__timeframe = timeframe

    @property
    def current_datetime(self):
        return self.__current_datetime

    @current_datetime.setter
    def current_datetime(self, dt: datetime.datetime):
        self.__current_datetime = dt
        self._indicator_data = self._indicator(dt)

    @property
    def indicator(self) -> pd.Series:
        return self._indicator_data

    @property
    def prediction(self) -> float:
        return self._indicator_data[1]

    @property
    def power(self) -> float:
        return self._indicator_data['power']

    @property
    def target_time(self) -> datetime.datetime:
        return self._indicator_data['target_time']

    def _indicator(self, dt: datetime.datetime or str) -> pd.DataFrame or None:
        _timedelta_kwargs = get_timedelta_kwargs(self.discretization, current_timeframe='1m')
        _end_datetime = check_convert_to_datetime(dt, utc_aware=False)
        _start_datetime = _end_datetime - relativedelta(**_timedelta_kwargs)
        _df = self.pt_obj.load_model_predicted_data(model_uuid=self.model_uuid,
                                                    start_datetime=_start_datetime,
                                                    end_datetime=_end_datetime,
                                                    timeframe=self.timeframe,
                                                    discretization=self.discretization,
                                                    )
        if _df is not None and _df.shape[0] > 0:
            _df['power'] = _df['power'] / Constants.binsizes[self.discretization]
            return _df.iloc[-1]
        else:
            if self.retry_counter == 2:
                self.retry_counter = 0
                return None
            else:
                self.retry_counter += 1
                #   retry 1 more time with recursion

                logger.warning(f'{self.__class__.__name__}: '
                               f'{self.model_uuid} received None, retry ({self.retry_counter})')
                time.sleep(self.retry_counter)
                return self._indicator(dt)


class IndicatorLoaded(DbIndicator):
    def __init__(self, model_uuid: str, prediction_tracker_obj: PredictionsTracker = None):
        super().__init__(model_uuid, prediction_tracker_obj)
        self.__preloaded_data: pd.DataFrame or None = None

    @property
    def prediction_show(self):
        return self.__preloaded_data[1]

    @property
    def power_show(self):
        return self.__preloaded_data['power']

    @property
    def current_datetime(self):
        return self.__current_datetime

    @current_datetime.setter
    def current_datetime(self, dt: datetime.datetime):
        self.__current_datetime = dt
        try:
            self._indicator_data = self.__preloaded_data.loc[dt]
        except (AttributeError, KeyError):
            logger.warning(
                f'{self.__class__.__name__}: {self.model_uuid} Error: self.__preloaded_data(dt) {dt}, add `None`')
            self._indicator_data = None
            # _timedelta_kwargs = get_timedelta_kwargs(self.discretization, current_timeframe='1m')
            # self._indicator_data = pd.DataFrame(
            #     data=[[self.__current_datetime + relativedelta(**_timedelta_kwargs), 0, .0, .0, ]],
            #     columns=['target_time', 'power', 0, 1, ],
            #     index=[pd.to_datetime(self.__current_datetime)]).iloc[0]

    def preload_indicator(self,
                          _start_datetime: datetime.datetime or str,
                          _end_datetime: datetime.datetime or str
                          ):
        _end_datetime = check_convert_to_datetime(_end_datetime, utc_aware=False)
        _start_datetime = check_convert_to_datetime(_start_datetime, utc_aware=False)
        _df = self.pt_obj.load_model_predicted_data(model_uuid=self.model_uuid,
                                                    start_datetime=_start_datetime,
                                                    end_datetime=_end_datetime,
                                                    timeframe=self.timeframe,
                                                    discretization=self.discretization,
                                                    cached=True)
        logger.debug(
            f'{self.__class__.__name__}: Preload: {self.model_uuid}  {_start_datetime}-{_end_datetime}')

        if _df is not None and _df.shape[0] > 0:
            self.__preloaded_data = _df.copy(deep=True)
            self.__preloaded_data['power'] = self.__preloaded_data['power'] / Constants.binsizes[self.discretization]
        else:
            self.__preloaded_data = None
        pass
