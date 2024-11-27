import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd

import multiprocessing as mp
from collections import OrderedDict
from numpy.random import default_rng
from functools import lru_cache, partial

from itertools import repeat, product, compress
from concurrent.futures import ProcessPoolExecutor, as_completed
from backtester.optimizeoptuna import prepare_strategy_params

import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

try:
    from tqdm.auto import tqdm as _tqdm

    _tqdm = partial(_tqdm, leave=False)
except ImportError:
    def _tqdm(seq, **_):
        return seq

# import backtrader as bt
# from backtrader_plotting import Bokeh
# from backtrader_plotting.schemes import Tradimo
# from backtesting.lib import SignalStrategy
from backtesting import Backtest, Strategy
# from backtester.core import Backtester
# from datawizard.datautils import slugify
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

__version__ = 0.051

logger = logging.getLogger()


def convert_best_params(optuna_best_params: dict, template: dict):
    prepared_best_params = OrderedDict()
    for ix, (k, v) in enumerate(template.items()):
        if k.startswith('model_') or k in ('minus', 'plus'):
            model_algoparams: dict = {}
            for key, value in optuna_best_params.items():
                if key.endswith(f'_{ix:02}'):
                    model_algoparams.update({key[:-3]: value})
            if model_algoparams:
                prepared_best_params.update({k: model_algoparams})
            # if model_algoparams:
            #     try:
            #         if model_algoparams['switch'] == 1:
            #             prepared_best_params.update({k: model_algoparams})
            #     except KeyError:
            #         prepared_best_params.update({k: model_algoparams})
    return prepared_best_params


def get_reference_points(maximize: Union[str, List],
                         references_range_dict: Union[dict, None] = None) -> Union[np.ndarray, None]:
    if references_range_dict is None:
        references_range_dict: dict = {'Return (Ann.) [%]': (130, 202, 15),
                                       'Win Rate [%]': (56, 90, 15),
                                       '# Trades': (23, 64, 1),
                                       }

    if isinstance(maximize, str):
        ref_range = references_range_dict.get(maximize, None)
        if ref_range is None:
            return None
        else:
            return np.expand_dims(np.arange(ref_range), 1)
    elif isinstance(maximize, list):
        references_arr = []
        for ix, max_key in enumerate(maximize):
            ref_range = references_range_dict.get(max_key, None)
            if ref_range is None:
                return None
            else:
                if max_key == '# Trades':
                    references_arr.append(np.arange(*ref_range))
                else:
                    references_arr.append(np.linspace(*ref_range))
        references_arr = np.asarray([*product(*references_arr)])
        return references_arr


# noinspection PyArgumentList
def init_optuna_study(direction: Union[str, list] = 'maximize',
                      study_name: str = 'testing-search',  # Unique identifier of the study.
                      storage_name: str = 'postgresql+psycopg2://sunday:my_test_pass@localhost/sunday',
                      sampler=None,
                      reference_points=None,
                      elite_population_selection_strategy=None):
    is_new_study = False
    samplers_dict = dict(cmaes=optuna.samplers.CmaEsSampler(n_startup_trials=1000,
                                                            restart_strategy='ipop',
                                                            independent_sampler=None,
                                                            seed=42,
                                                            ),
                         qmc=optuna.samplers.QMCSampler(scramble=False,
                                                        independent_sampler=None,
                                                        seed=42,
                                                        ),
                         nsgaii=optuna.samplers.NSGAIISampler(population_size=120,
                                                              mutation_prob=0.1,
                                                              crossover_prob=0.1,
                                                              seed=42, ),

                         tpe=optuna.samplers.TPESampler(n_startup_trials=900,
                                                        n_ei_candidates=300,
                                                        seed=42),
                         nsgaiii=optuna.samplers.NSGAIIISampler(
                             reference_points=reference_points,
                             elite_population_selection_strategy=elite_population_selection_strategy,
                             population_size=120,
                             mutation_prob=0.1,
                             crossover_prob=0.1,
                             seed=42, )
                         )

    if sampler is None:
        sampler_arg = samplers_dict.get('tpe')
    else:
        sampler_arg = samplers_dict.get(sampler, None)
        if sampler_arg is None:
            raise ValueError('`sampler`, must have one of the value = `tpe`, `nsgaii`, `nsgaiii`, `qmc` or `cmaes`')
        if isinstance(direction, list):
            if sampler == 'cmaes':
                logger.info(
                    f"{__name__}: Sampler = {sampler} can't used in multiobjective study setting `nsgaiii` sampler")
                sampler_arg = samplers_dict.get('nsgaiii', None)
    try:
        study = optuna.load_study(study_name=study_name,
                                  storage=storage_name,
                                  sampler=sampler_arg,
                                  )
    except:
        study_kwargs = dict(study_name=study_name,
                            storage=storage_name,
                            sampler=sampler_arg,
                            load_if_exists=True,
                            )
        if isinstance(direction, list):
            study_kwargs['directions'] = direction
        else:
            study_kwargs['direction'] = direction
        study = optuna.create_study(**study_kwargs)

        is_new_study = True

    logger.info(f"{__name__}: Sampler is {study.sampler.__class__.__name__}")
    return study, is_new_study


def get_top_results(study, maximize, win_rate: int = 60, top_qty=250) -> pd.DataFrame:
    best_trials_df = pd.DataFrame()
    if isinstance(maximize, str):
        win_rate_col = 'Win Rate [%]'
        trial_df = study.trials_dataframe(attrs=('number', 'value', 'params', 'user_attrs',), )
        if win_rate_col != maximize:
            col_name = 'values_0'
        else:
            col_name = 'Return (Ann.) [%]'
        unique_values = trial_df[(trial_df[col_name] > -np.inf) & (trial_df[col_name] < np.inf)][col_name].unique()
        val_percent = np.quantile(unique_values, 0.85)
        trial_df = trial_df[(trial_df[col_name].isin(unique_values)) & (trial_df[col_name] > val_percent)]

        # Win rate > 56%
        trial_df = trial_df[(trial_df[win_rate_col] > win_rate)]
        trial_df = trial_df.sort_values([col_name, "user_attrs_SQN"], ascending=False)
        subset = [col_name, win_rate_col]
        best_trials_df = trial_df.drop_duplicates(subset=subset).head(top_qty)
        if best_trials_df.empty:
            best_trials_df = trial_df.drop_duplicates(subset=[subset[0]]).head(top_qty)
    elif isinstance(maximize, list):
        win_rate_col = 'Win Rate [%]'
        trade_col = '# Trades'
        trial_df = study.trials_dataframe(attrs=('number', 'value', 'params', 'user_attrs',), )

        col_range = [0, 1, 2]
        if win_rate_col in maximize:
            win_rate_ix = maximize.index(win_rate_col)
            trial_df = trial_df[(trial_df[f'values_{win_rate_ix}'] > win_rate)]
        else:
            win_rate_ix = 2
            # Win rate > 56%
            trial_df = trial_df[(trial_df[win_rate_col] > win_rate)]

        col_range.remove(win_rate_ix)
        if trade_col in maximize:
            trade_ix = maximize.index(trade_col)
            col_range.remove(trade_ix)

        unique_values = []
        val_percent = []

        for i in col_range:
            col_name = f"values_{i}"
            unique_values.append(
                trial_df[(trial_df[col_name] > -np.inf) & (trial_df[col_name] < np.inf)][col_name].unique())
            val_percent.append(np.quantile(unique_values[i], 0.85))

        for i in col_range:
            trial_df = trial_df[
                (trial_df[f"values_{i}"].isin(unique_values[i])) & (trial_df[f"values_{i}"] > val_percent[i])]
        trial_df = trial_df.sort_values(["values_0", "user_attrs_SQN"], ascending=False)
        if win_rate_ix not in col_range and win_rate_ix < 2:
            col_range.append(win_rate_ix)
        subset = [f"values_{i}" for i in col_range]
        best_trials_df = trial_df.drop_duplicates(subset=subset).head(top_qty)
        if best_trials_df.empty:
            best_trials_df = trial_df.drop_duplicates(subset=[subset[0]]).head(top_qty)
    del trial_df
    del unique_values
    del val_percent
    return best_trials_df


class NewBacktest(Backtest):
    def __init__(self,
                 data: pd.DataFrame,
                 strategy: Type[Strategy],
                 *args,
                 cash: float = 10_000,
                 commission: float = .0,
                 margin: float = 1.,
                 trade_on_close=False,
                 hedging=False,
                 exclusive_orders=False,
                 ):
        data.columns = [item.lower().capitalize() for item in data.columns]
        super().__init__(
            data,
            strategy,
            cash=cash,
            *args,
            commission=commission,
            margin=margin,
            trade_on_close=trade_on_close,
            hedging=hedging,
            exclusive_orders=exclusive_orders
        )
        self.keys_to_include = None

    def optimize_optuna(self, *,
                        maximize: Union[str, Callable[[pd.Series], float]] = 'SQN',
                        max_trials: int = 100,
                        constraint: Callable[[dict], bool] = None,
                        keys_to_include=None,
                        study_name: str = 'testing-search',  # Unique identifier of the study.
                        storage_name: str = 'postgresql+psycopg2://sunday:my_test_pass@localhost/sunday',
                        sampler='nsgaii',
                        # base_path: str = "/home/cubecloud/Data/sunday_tests/strategies",
                        **kwargs) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Optimize strategy parameters using Optuna.
        Returns result `pd.Series` of the best run.
        `maximize` is a string key from the
        `backtesting.backtesting.Backtest.run`-returned results series,
        or a function that accepts this series object and returns a number;
        the higher the better. By default, the method maximizes
        Van Tharp's System Quality Number (SQN).
    
        `max_trials` is the maximal number of trials to perform.
    
        Additional keyword arguments can be passed to Optuna's `study.optimize()` method.
    
        Returns the best trial's result as a `pd.Series` object.
        """
        study_kwargs: dict = {'direction': 'maximize',
                              'study_name': study_name,
                              'storage_name': storage_name,
                              'sampler': sampler,
                              'reference_points': None
                              }
        references_range_dict: Dict[str, tuple] = {'Return (Ann.) [%]': (130, 202, 15),
                                                   'Win Rate [%]': (56, 90, 15),
                                                   '# Trades': (23, 64, 1),
                                                   }
        maximize_key = str(maximize)
        if isinstance(maximize, (str, Callable)):
            maximize_key: str = f'{maximize}'
        elif isinstance(maximize, (list, tuple)):
            maximize_key: list = [f'{k}' for k in maximize]
            study_kwargs.update({'direction': ['maximize' for _ in maximize]})
        else:
            assert isinstance(maximize, (
                str, list, tuple, Callable)), f'TypeError: maximize is not `str`, `list`, `tuple` or `Callable`'

        if constraint is None:
            def constraint(_):
                return True
        elif not callable(constraint):
            raise TypeError("`constraint` must be a function that accepts a dict "
                            "of strategy parameters and returns a bool whether "
                            "the combination of parameters is admissible or not")

        # cpu_count = os.cpu_count() or 1
        # n_jobs = cpu_count - 2 if cpu_count > 6 else cpu_count
        n_jobs = 1

        def objective(trial):
            params = prepare_strategy_params(trial=trial, kwargs=kwargs, constraint=constraint)
            if params:
                try:
                    _stats_values = self.run(**params)
                except ValueError:
                    return np.nan
                for _key in keys_to_include:
                    if isinstance(maximize_key, list) and _key not in maximize_key:
                        trial.set_user_attr(key=_key, value=_stats_values[_key])
                    elif isinstance(maximize_key, str) and _key != maximize_key:
                        trial.set_user_attr(key=_key, value=_stats_values[_key])
                if isinstance(maximize_key, list):
                    return [_stats_values[v] for v in maximize_key]
                else:
                    return _stats_values[maximize_key]
            else:
                if isinstance(maximize_key, list):
                    return [np.nan for _ in maximize_key]
                else:
                    return np.nan

        def prepare_references_range(maximize_: list or str, references_dict: dict) -> dict:
            checked_references_range_dict: dict = {}
            if isinstance(maximize_, str):
                if maximize_ in references_dict.keys():
                    checked_references_range_dict[maximize_] = references_dict[maximize_]
            elif isinstance(maximize_, list):
                for k, v in references_dict.items():
                    if k in maximize_key and k in references_dict.keys():
                        checked_references_range_dict[k] = references_dict[k]
            return checked_references_range_dict

        if sampler == 'nsgaiii':
            study_kwargs.update(
                {'reference_points': get_reference_points(maximize,
                                                          prepare_references_range(maximize_key,
                                                                                   references_range_dict))})

        study, is_new_study = init_optuna_study(**study_kwargs)

        if is_new_study:
            logger.info(f"{self.__class__.__name__}: New study ")
        else:
            logger.info(f"{self.__class__.__name__}: Loaded study")
            if sampler == 'nsgaiii':
                logger.info(f"{self.__class__.__name__}: Get the actual reference points")
                best_trials_df = get_top_results(study, maximize)
                if isinstance(maximize_key, str):
                    ref_range = references_range_dict.get(maximize_key, None)
                    if ref_range is not None:
                        if ref_range[2] == 1:
                            new_max = best_trials_df["values_0"].max()
                        else:
                            new_max = best_trials_df["values_0"].max() + ref_range[2]
                        new_range: tuple = (best_trials_df["values_0"].min(), new_max, ref_range[2])
                    else:
                        if maximize_key != '# Trades':
                            step = 15
                        else:
                            step = 1
                        new_range: tuple = (best_trials_df["values_0"].min(), best_trials_df["values_0"].max(), step)
                    references_range_dict.update({maximize_key: tuple(new_range)})
                elif isinstance(maximize_key, list):
                    for ix, key in enumerate(maximize_key):
                        ref_range = references_range_dict.get(key, None)
                        if ref_range is not None:
                            if ref_range[2] == 1:
                                new_max = best_trials_df[f"values_{ix}"].max()
                            else:
                                new_max = best_trials_df[f"values_{ix}"].max() + ref_range[2]
                            new_range: tuple = (best_trials_df[f"values_{ix}"].min(), new_max, ref_range[2])
                        else:
                            if key != '# Trades':
                                step = 15
                            else:
                                step = 1
                            new_range: tuple = (best_trials_df[f"values_{ix}"].min(),
                                                best_trials_df[f"values_{ix}"].max(),
                                                step)
                        references_range_dict.update({key: tuple(new_range)})
                del best_trials_df
                logger.info(f"{self.__class__.__name__}: New reference points ranges{references_range_dict}")
                study_kwargs.update(
                    {'reference_points': get_reference_points(maximize,
                                                              prepare_references_range(maximize_key,
                                                                                       references_range_dict))})

                logger.info(f"{self.__class__.__name__}: Reload study with new reference points")
                study, _ = init_optuna_study(**study_kwargs)
                logger.info(f"{self.__class__.__name__}: Loaded study")

        study.optimize(objective,
                       n_jobs=n_jobs,
                       callbacks=[MaxTrialsCallback(n_trials=max_trials, states=(TrialState.COMPLETE,))])

        best_trials_df = get_top_results(study, maximize)
        best_trial_number = best_trials_df['number'].iloc[0]
        logger.info(f"{self.__class__.__name__}: Best trial number: {best_trial_number}")
        best_trial_params = study.trials[best_trial_number].params
        kwargs_value = kwargs.get('ind_params', None)
        assert kwargs_value is not None, f"Error: kwargs error - {kwargs_value}"
        best_value = convert_best_params(best_trial_params, kwargs_value)
        best_trial_params: dict = {'ind_params': best_value}
        _stats = self.run(**best_trial_params)
        del best_trials_df

        logger.info(f"{self.__class__.__name__}: Best params: {best_trial_params}")
        logger.info(f"{self.__class__.__name__}: Stats: {[stat for stat in _stats[maximize]]}")

        trial_df = study.trials_dataframe(attrs=('number', 'value', 'params', 'user_attrs'), )
        trial_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        trial_df.dropna(inplace=True)

        # Return the best result as a pd.Series object
        new_cols = []
        for col_name in trial_df.columns:
            if 'user_attrs' in col_name:
                col_name = col_name[11:]
            elif 'params' in col_name:
                col_name = col_name[7:]
            elif 'value' in col_name:
                try:
                    ix = int(col_name[-1:])
                    col_name = maximize_key[ix]
                except:
                    col_name = f'{maximize_key}'
            new_cols.append(col_name)
        trial_df.columns = new_cols
        return _stats, trial_df

    @staticmethod
    def _mp_task(backtest_uuid, args):
        batch_index = args[0]
        keys_to_include = args[1]
        bt, param_batches, maximize_func = NewBacktest._mp_backtests[backtest_uuid]
        return batch_index, [
            stats[keys_to_include].values if stats['# Trades'] else [np.nan for _ in keys_to_include]
            for stats in (bt.run(**params)
                          for params in param_batches[batch_index])]

        # @staticmethod
        # def _mp_task_optuna(backtest_uuid, args):
        #     batch_index = args[0]
        #     keys_to_include = args[1]
        #     bt, param_batches, maximize_func = NewBacktest._mp_backtests[backtest_uuid]
        #     return batch_index, [stats[keys_to_include] if stats['# Trades'] else [np.nan for _ in keys_to_include]
        #                          for stats in (bt.run(**params)
        #                                        for (_, params) in param_batches[batch_index])]

    _mp_backtests: Dict[float, Tuple['NewBacktest', List, Callable]] = {}

    def optimize(self, *,
                 maximize: Union[str, Callable[[pd.Series], float]] = 'SQN',
                 method: str = 'grid',
                 max_tries: Union[int, float] = None,
                 constraint: Callable[[dict], bool] = None,
                 return_heatmap: bool = False,
                 return_optimization: bool = False,
                 random_state: int = None,
                 keys_to_include=None,
                 study_name=None,
                 storage_name=None,
                 sampler='nsgaii',
                 **kwargs) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame], Tuple[pd.Series, pd.DataFrame, dict]]:
        """
        Optimize strategy parameters to an optimal combination.
        Returns result `pd.Series` of the best run.

        `maximize` is a string key from the
        `backtesting.backtesting.Backtest.run`-returned results series,
        or a function that accepts this series object and returns a number;
        the higher the better. By default, the method maximizes
        Van Tharp's [System Quality Number](https://google.com/search?q=System+Quality+Number).

        `method` is the optimization method. Currently two methods are supported:

        * `"grid"` which does an exhaustive (or randomized) search over the
          cartesian product of parameter combinations, and
        * `"skopt"` which finds close-to-optimal strategy parameters using
          [model-based optimization], making at most `max_tries` evaluations.

        [model-based optimization]: \
            https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html

        `max_tries` is the maximal number of strategy runs to perform.
        If `method="grid"`, this results in randomized grid search.
        If `max_tries` is a floating value between (0, 1], this sets the
        number of runs to approximately that fraction of full grid space.
        Alternatively, if integer, it denotes the absolute maximum number
        of evaluations. If unspecified (default), grid search is exhaustive,
        whereas for `method="skopt"`, `max_tries` is set to 200.

        `constraint` is a function that accepts a dict-like object of
        parameters (with values) and returns `True` when the combination
        is admissible to test with. By default, any parameters combination
        is considered admissible.

        If `return_heatmap` is `True`, besides returning the result
        series, an additional `pd.Series` is returned with a multiindex
        of all admissible parameter combinations, which can be further
        inspected or projected onto 2D to plot a heatmap
        (see `backtesting.lib.plot_heatmaps()`).

        If `return_optimization` is True and `method = 'skopt'`,
        in addition to result series (and maybe heatmap), return raw
        [`scipy.optimize.OptimizeResult`][OptimizeResult] for further
        inspection, e.g. with [scikit-optimize]\
        [plotting tools].

        [OptimizeResult]: \
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
        [scikit-optimize]: https://scikit-optimize.github.io
        [plotting tools]: https://scikit-optimize.github.io/stable/modules/plots.html

        If you want reproducible optimization results, set `random_state`
        to a fixed integer random seed.

        Additional keyword arguments represent strategy arguments with
        list-like collections of possible values. For example, the following
        code finds and returns the "best" of the 7 admissible (of the
        9 possible) parameter combinations:

            backtest.optimize(sma1=[5, 10, 15], sma2=[10, 20, 40],
                              constraint=lambda p: p.sma1 < p.sma2)

        .. TODO::
            Improve multiprocessing/parallel execution on Windows with start method 'spawn'.
        """
        if keys_to_include is None:
            self.keys_to_include = ['Equity Final [$]', 'Return (Ann.) [%]',
                                    'SQN', 'Calmar Ratio']
        else:
            self.keys_to_include = keys_to_include

        if not kwargs:
            raise ValueError('Need some strategy parameters to optimize')

        # Add the keys to include from 'stats'

        maximize_key = None
        if isinstance(maximize, str):
            maximize_key = str(maximize)
            stats = self._results if self._results is not None else self.run()
            if maximize_key not in stats:
                raise ValueError('`maximize`, if str, must match a key in pd.Series '
                                 'result of backtest.run()')

            self.keys_to_include = [k for k in self.keys_to_include if k in stats]

            if maximize_key in self.keys_to_include:
                if self.keys_to_include.index(maximize_key) != 0:
                    self.keys_to_include.remove(maximize_key)
                    self.keys_to_include.insert(0, f'{maximize_key}')
            else:
                self.keys_to_include.insert(0, f'{maximize_key}')
            if method != 'optuna':
                def maximize(stats: pd.Series, _key=maximize):
                    return stats[_key]

        elif isinstance(maximize, list) and method == 'optuna':
            stats = self._results if self._results is not None else self.run()
            for maximize_key in reversed(maximize):
                maximize_key = str(maximize_key)
                if maximize_key not in stats:
                    raise ValueError('`maximize`, if str, must match a key in pd.Series '
                                     'result of backtest.run()')
                if maximize_key in self.keys_to_include:
                    if self.keys_to_include.index(maximize_key) != 0:
                        self.keys_to_include.remove(maximize_key)
                        self.keys_to_include.insert(0, f'{maximize_key}')
                else:
                    self.keys_to_include.insert(0, f'{maximize_key}')
            if method != 'optuna':
                def maximize(stats: pd.Series, _key=maximize):
                    return stats[_key]

        elif not callable(maximize):
            raise TypeError('`maximize` must be str (a field of backtest.run() result '
                            'Series) or a function that accepts result Series '
                            'and returns a number; the higher the better')

        have_constraint = bool(constraint)
        if constraint is None:
            def constraint(_):
                return True
        elif not callable(constraint):
            raise TypeError("`constraint` must be a function that accepts a dict "
                            "of strategy parameters and returns a bool whether "
                            "the combination of parameters is admissible or not")

        if return_optimization and method != 'skopt':
            raise ValueError("return_optimization=True only valid if method='skopt'")

        def _tuple(x):
            return x if isinstance(x, Sequence) and not isinstance(x, str) else (x,)

        for k, v in kwargs.items():
            if len(_tuple(v)) == 0:
                raise ValueError(f"Optimization variable '{k}' is passed no "
                                 f"optimization values: {k}={v}")

        class AttrDict(dict):
            def __getattr__(self, item):
                return self[item]

        def _grid_size():
            size = np.prod([len(_tuple(v)) for v in kwargs.values()])
            if size < 10_000 and have_constraint:
                size = sum(1 for p in product(*(zip(repeat(k), _tuple(v))
                                                for k, v in kwargs.items()))
                           if constraint(AttrDict(p)))
            return size

        def _optimize_grid() -> Union[tuple[pd.Series, pd.DataFrame], pd.Series]:
            rand = default_rng(random_state).random
            gr_size = _grid_size()
            grid_frac = (1 if max_tries is None else
                         max_tries if 0 < max_tries <= 1 else
                         max_tries / gr_size)

            param_combos = [dict(params)  # back to dict so it pickles
                            for params in (AttrDict(params)
                                           for params in product(*(zip(repeat(k), _tuple(v))
                                                                   for k, v in kwargs.items())))
                            if constraint(params)  # type: ignore
                            and rand() <= grid_frac]

            if not param_combos:
                raise ValueError('No admissible parameter combinations to test')

            if len(param_combos) > 300:
                warnings.warn(f'Searching for best of {len(param_combos)} configurations.',
                              stacklevel=2)

            heatmap = pd.DataFrame(np.nan,
                                   columns=self.keys_to_include,
                                   index=pd.MultiIndex.from_tuples(
                                       [p.values() for p in param_combos],
                                       names=next(iter(param_combos)).keys()))

            def _batch(seq):
                n = np.clip(int(len(seq) // (os.cpu_count() or 1)), 1, 300)
                for i in range(0, len(seq), n):
                    yield seq[i:i + n]

            # Save necessary objects into "global" state; pass into concurrent executor
            # (and thus pickle) nothing but two numbers; receive nothing but numbers.
            # With start method "fork", children processes will inherit parent address space
            # in a copy-on-write manner, achieving better performance/RAM benefit.
            backtest_uuid = np.random.random()
            param_batches = list(_batch(param_combos))
            NewBacktest._mp_backtests[backtest_uuid] = (self, param_batches, maximize)  # type: ignore
            try:
                # If multiprocessing start method is 'fork' (i.e. on POSIX), use
                # a pool of processes to compute results in parallel.
                # Otherwise, (i.e. on Windows), sequential computation will be "faster".
                if mp.get_start_method(allow_none=False) == 'fork':
                    with ProcessPoolExecutor() as executor:
                        futures = [executor.submit(NewBacktest._mp_task, backtest_uuid, (i, self.keys_to_include))
                                   for i in range(len(param_batches))]
                        for future in _tqdm(as_completed(futures), total=len(futures),
                                            desc='NewBacktest.optimize'):
                            batch_index, stats_lst = future.result()
                            for _stats_values, params in zip(stats_lst, param_batches[batch_index]):
                                # logger.debug(f"{self.__class__.__name__}: stats: {_stats_values}, "
                                #              f"type {type(_stats_values)}")
                                heatmap.loc[tuple(params.values())] = _stats_values

                else:
                    if os.name == 'posix':
                        warnings.warn("For multiprocessing support in `NewBacktest.optimize()` "
                                      "set multiprocessing start method to 'fork'.")
                    for batch_index in _tqdm(range(len(param_batches))):
                        _, stats_lst = NewBacktest._mp_task(backtest_uuid, (batch_index, self.keys_to_include))
                        for _stats_values, params in zip(stats_lst, param_batches[batch_index]):
                            heatmap.loc[tuple(params.values())] = _stats_values
            finally:
                del NewBacktest._mp_backtests[backtest_uuid]

            best_params = heatmap.idxmax()[0]

            if pd.isnull(best_params):
                # No trade was made in any of the runs. Just make a random
                # run so we get some, if empty, results
                stats = self.run(**param_combos[0])
            else:
                stats = self.run(**dict(zip(heatmap.index.names, best_params)))
            if return_heatmap:
                return stats, heatmap
            return stats

        def _optimize_skopt() -> Union[pd.Series, Tuple[pd.Series, pd.Series], Tuple[pd.Series, pd.Series, dict]]:
            try:
                from skopt import forest_minimize
                from skopt.space import Integer, Real, Categorical
                from skopt.utils import use_named_args
                from skopt.callbacks import DeltaXStopper
                from skopt.learning import ExtraTreesRegressor
            except ImportError:
                raise ImportError("Need package 'scikit-optimize' for method='skopt'. "
                                  "pip install scikit-optimize")

            nonlocal max_tries
            max_tries = (200 if max_tries is None else
                         max(1, int(max_tries * _grid_size())) if 0 < max_tries <= 1 else
                         max_tries)

            dimensions = []
            for key, values in kwargs.items():
                values = np.asarray(values)
                if values.dtype.kind in 'mM':  # timedelta, datetime64
                    # these dtypes are unsupported in skopt, so convert to raw int
                    # TODO: save dtype and convert back later
                    values = values.astype(int)

                if values.dtype.kind in 'iumM':
                    dimensions.append(Integer(low=values.min(), high=values.max(), name=key))
                elif values.dtype.kind == 'f':
                    dimensions.append(Real(low=values.min(), high=values.max(), name=key))
                else:
                    dimensions.append(Categorical(values.tolist(), name=key, transform='onehot'))

            # Avoid recomputing re-evaluations:
            # "The objective has been evaluated at this point before."
            # https://github.com/scikit-optimize/scikit-optimize/issues/302
            memoized_run = lru_cache()(lambda tup: self.run(**dict(tup)))

            # np.inf/np.nan breaks sklearn, np.finfo(float).max breaks skopt.plots.plot_objective
            INVALID = 1e300
            progress = iter(_tqdm(repeat(None), total=max_tries, desc='Backtest.optimize'))

            @use_named_args(dimensions=dimensions)
            def objective_function(**params):
                next(progress)
                # Check constraints
                # TODO: Adjust after https://github.com/scikit-optimize/scikit-optimize/pull/971
                if not constraint(AttrDict(params)):
                    return INVALID
                res = memoized_run(tuple(params.items()))
                value = -maximize(res)
                if np.isnan(value):
                    return INVALID
                return value

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', 'The objective has been evaluated at this point before.')

                res = forest_minimize(
                    func=objective_function,
                    dimensions=dimensions,
                    n_calls=max_tries,
                    base_estimator=ExtraTreesRegressor(n_estimators=20, min_samples_leaf=2),
                    acq_func='LCB',
                    kappa=3,
                    n_initial_points=min(max_tries, 20 + 3 * len(kwargs)),
                    initial_point_generator='lhs',  # 'sobel' requires n_initial_points ~ 2**N
                    callback=DeltaXStopper(9e-7),
                    random_state=random_state)

            stats = self.run(**dict(zip(kwargs.keys(), res.x)))
            output = [stats]

            if return_heatmap:
                heatmap = pd.Series(dict(zip(map(tuple, res.x_iters), -res.func_vals)),
                                    name=maximize_key)
                heatmap.index.names = kwargs.keys()
                heatmap = heatmap[heatmap != -INVALID]
                heatmap.sort_index(inplace=True)
                output.append(heatmap)

            if return_optimization:
                valid = res.func_vals != INVALID
                res.x_iters = list(compress(res.x_iters, valid))
                res.func_vals = res.func_vals[valid]
                output.append(res)

            return stats if len(output) == 1 else tuple(output)

        if method == 'grid':
            output = _optimize_grid()
        elif method == 'skopt':
            output = _optimize_skopt()
        elif method == 'optuna':
            output = self.optimize_optuna(maximize=maximize,
                                          max_trials=max_tries,
                                          constraint=constraint,
                                          keys_to_include=self.keys_to_include,
                                          study_name=study_name,  # Unique identifier of the study.
                                          storage_name=storage_name,
                                          sampler=sampler,
                                          **kwargs
                                          )
        else:
            raise ValueError(f"Method should be 'grid' or 'skopt', not {method!r}")
        return output
