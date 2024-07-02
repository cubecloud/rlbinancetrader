import logging
from typing import List, Dict

import pandas as pd
from ensembletools.modelstools import ModelCard
from ensembletools.indicators import IndicatorLoaded
from ensembletools.modelstools import PredictionTracker
from ensembletools.modelstools import get_raw_ph_obj

__version__ = 0.09
logger = logging.getLogger()


class LoadDbIndicators:
    def __init__(self,
                 start_datetime,
                 end_datetime,
                 symbol_pairs: List[str,] = ('BTCUSDT',),
                 market: str = 'spot',
                 timeframe='15m',
                 discretization='15m',
                 index_type='target_time'
                 ):
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.symbol_pairs = symbol_pairs
        self.market = market
        self.timeframe = timeframe
        self.discretization = discretization
        self.raw_prediction_history_obj = get_raw_ph_obj()
        self.predictiontracker_objs: Dict[str:PredictionTracker, ] = {}
        self.indicators_objs: List[IndicatorLoaded,] = []
        self.symbol_pair_models_uuid: dict = {}
        self.indicators: Dict[str:IndicatorLoaded] = {'plus': dict(), 'minus': dict(), 'other': dict()}
        self.index_type = index_type

    def __init_dbindicators(self, index_type='target_time'):
        for pair in self.symbol_pairs:
            self.predictiontracker_objs.update(
                {pair: PredictionTracker(symbol=pair, market=self.market, raw_ph_obj=self.raw_prediction_history_obj)})
            models_uuid = self.predictiontracker_objs[pair].get_all_models_uuid_list()
            self.symbol_pair_models_uuid.update({pair: models_uuid})

            for ix, model_UUID in enumerate(self.symbol_pair_models_uuid[pair]):
                indicator_obj = self.__init_indicator(model_UUID, index_type)
                indicator_obj.indicator_id = ix
                self.indicators_objs.append(indicator_obj)

                """ for future development """
                # if indicator_obj.direction is None:
                #     self.indicators['other'].update({indicator_obj.name: indicator_obj})
                # else:
                #     if indicator_obj.direction == 'plus':
                #         self.indicators['plus'].update({indicator_obj.name: indicator_obj})
                #     elif indicator_obj.direction == 'minus':
                #         self.indicators['minus'].update({indicator_obj.name: indicator_obj})

    def __update_dbindicators(self, index_type='target_time'):
        for indicator_obj in self.indicators_objs:
            indicator_obj.index_type = index_type
            indicator_obj.discretization = self.discretization
            indicator_obj.timeframe = self.timeframe
            indicator_obj.preload_indicator(self.start_datetime, self.end_datetime)

    def __init_indicator(self, model_UUID, index_type) -> IndicatorLoaded:
        _model_card: ModelCard = self.raw_prediction_history_obj.get_card(model_UUID)
        indicator_obj = IndicatorLoaded(model_uuid=model_UUID,
                                        prediction_tracker_obj=self.predictiontracker_objs[_model_card.symbol])
        indicator_obj.index_type = index_type
        indicator_obj.discretization = self.discretization
        indicator_obj.timeframe = self.timeframe
        indicator_obj.preload_indicator(self.start_datetime, self.end_datetime)
        return indicator_obj

    def set_new_period(self, start_datetime, end_datetime, index_type='target_time'):
        if not self.indicators_objs:
            self.__init_dbindicators(index_type)
            self.index_type = index_type

        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.__update_dbindicators(index_type=index_type)

    def get_data_df(self, index_type='target_time') -> pd.DataFrame:
        if not self.indicators_objs:
            self.__init_dbindicators(index_type)
            self.index_type = index_type

        df_lst: List[pd.DataFrame] = []
        for indicator_obj in self.indicators_objs:
            indicator_obj.index_type = index_type
            indicator_obj.preload_indicator(self.start_datetime, self.end_datetime)
            use_columns = [1]
            indicator_obj.columns = use_columns
            _df = indicator_obj.prediction_show
            _df.columns = [indicator_obj.name]
            df_lst.append(_df)
        _df = pd.concat(df_lst, axis=1)
        return _df
