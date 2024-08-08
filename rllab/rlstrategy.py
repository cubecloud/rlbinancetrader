from backtesting import Strategy
from backtester.newbacktestcore import NewBacktest as Back
from backtester.strategies import *

bt = Back(data_df,
          strategy,
          cash=start_cache,
          commission=.002,
          trade_on_close=False,
          )

# Получаем статистику бэктестинга
if strategies_kwargs_lst:
    strat_kwargs = dict(**strategies_kwargs_lst[ix])
    if obj_name is not None and obj_name.startswith('Db'):
        strat_kwargs.update({'raw_prediction_history_obj': self.raw_ph_obj})
    _stats = bt.run(**strat_kwargs)
else:
    _stats = bt.run(events_data=events_data, cards_data=cards_data)
# if strategies_kwargs_lst:
#     bt._strategy._check_params(**strategies_kwargs_lst[ix])
# _stats = bt.run()
# Печатаем статистику и записываем
stats_path_filename = f'{path_filename}_{strategy.name}.txt'
_original_stdout = sys.stdout
_stats_file = open(stats_path_filename, 'w')
sys.stdout = _stats_file
print(strategy.name)
print(_stats)
_stats_file.close()
sys.stdout = _original_stdout
logger.info(f"{self.__class__.__name__}: Saved stats: {stats_path_filename}")
_path_filename = f'{path_filename}_{strategy.name}{suffix}.html'
bt.plot(plot_volume=True, relative_equity=True, open_browser=False, filename=_path_filename)
logger.info(f"{self.__class__.__name__}: Saved backtest: {_path_filename}")
result.update({ix: _stats})
