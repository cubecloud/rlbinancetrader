"""Прогон движка v7 с oracle-хуком, с cooldown и без него.

Переиспользует загрузку данных и параметры из ``strategy_rl.v7runner`` (П3:
никакой своей версии движка). Класс хука параметризуется, чтобы измерить цену
cooldown-правила НЕ трогая ``strategy_rl/v7hook.py``: подкласс
``AgentHookNoCooldownBT`` после агентского закрытия ставит
``_await_signal_close = True`` — тогда родитель НЕ взводит cooldown
(``regimeb_bt_strategy.py:276-277``).

Семантика строго движковая: решение на баре i → исполнение по ``open(i+1)`` с
комиссией; никакого ``max(High)`` внутри сделки; компаунд — полным движком.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_STRATEGY_RL = Path(__file__).resolve().parents[2] / "strategy_rl"
if str(_STRATEGY_RL) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_RL))
# sunday-корень нужен ДО импорта v7hook (он тянет tooling.audit...)
_SUNDAY = Path("/home/cubecloud/Python/projects/sunday")
if str(_SUNDAY) not in sys.path:
    sys.path.insert(0, str(_SUNDAY))

from backtesting import Backtest  # noqa: E402
from v7hook import AgentHookBT  # noqa: E402
from v7runner import P7, load_bt_df, load_calib  # noqa: E402

ExitPolicy = Callable[[object, int], bool]


class AgentHookNoCooldownBT(AgentHookBT):
    """Как ``AgentHookBT``, но агентский выход НЕ взводит cooldown.

    Отличие в одной строке: после ``position.close()`` выставляется
    ``_await_signal_close = True``, поэтому родительский ``next()`` считает
    выход «сигнальным» и cooldown не ставит. Служит для измерения чистой цены
    cooldown-правила (Reserve_B без cooldown минус Reserve_B с cooldown).
    """

    def next(self):  # noqa: D401
        """Шаг стратегии: ранний выход без взвода cooldown (disarm-режим)."""
        flag_before = self._await_signal_close
        # вызвать деда (RegimeDipBuyerBT.next), минуя AgentHookBT.next
        super(AgentHookBT, self).next()
        parent_closed = self._await_signal_close and not flag_before
        if (self.exit_policy is not None and self.position
                and not parent_closed and self._pos_tag == "dip"):
            i = len(self.data) - 1
            if self.exit_policy(self, i):
                self.position.close()
                self._await_signal_close = True   # DISARM cooldown


def _trades_frame(stats, idx: pd.Index) -> pd.DataFrame:
    """Трейдбук движка -> DataFrame с позиционными барами (как в v7runner)."""
    tr = stats._trades.copy().sort_values("EntryBar")
    return pd.DataFrame({
        "entry_bar": tr["EntryBar"].astype(int).to_numpy(),
        "exit_bar": tr["ExitBar"].astype(int).to_numpy(),
        "entry_price": tr["EntryPrice"].astype(float).to_numpy(),
        "exit_price": tr["ExitPrice"].astype(float).to_numpy(),
        "return_pct": tr["ReturnPct"].astype(float).to_numpy(),
    })


def run_engine(state_path: str, exit_policy: ExitPolicy | None = None,
               disarm_cooldown: bool = False, commission: float = 0.001):
    """Полный прогон движка. -> (stats, trades_df).

    ``exit_policy=None`` -> чистая v7. ``disarm_cooldown=True`` -> агентский
    выход не взводит cooldown (класс ``AgentHookNoCooldownBT``).
    """
    df, _ = load_bt_df(state_path)
    hook_cls = AgentHookNoCooldownBT if disarm_cooldown else AgentHookBT
    hook_cls.calib_table = load_calib()
    hook_cls.exit_policy = staticmethod(exit_policy) if exit_policy else None
    hook_cls.expert_log = None
    try:
        bt = Backtest(df, hook_cls, cash=10_000_000, commission=commission,
                      trade_on_close=False, exclusive_orders=True)
        stats = bt.run(**P7)
    finally:
        hook_cls.exit_policy = None
    return stats, _trades_frame(stats, df.index)


def load_ohlc(state_path: str) -> pd.DataFrame:
    """OHLC движка (позиционный бар-индекс), для frozen-расчёта резерва."""
    df, _ = load_bt_df(state_path)
    return df[["Open", "High", "Low", "Close"]].reset_index(drop=True)
