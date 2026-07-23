"""Изолятор политики (коррекция находки №1 критики 0.3).

Проблема: exit_policy(strategy, i) отдаёт политике весь объект стратегии,
внутри которого лежит будущее (например _next_open[i] — цена открытия
следующего бара). Oracle-политикам этапа 0.5 это нужно НАМЕРЕННО;
обучаемой политике — запрещено.

Решение: обучаемая политика получает ТОЛЬКО каузальный вектор признаков
бара i, собранный из белого списка колонок STATE + контекст сделки,
вычисленный из прошлого. Доступа к объекту стратегии у неё нет.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Поля объекта стратегии, содержащие будущее относительно бара решения.
# Изолированная политика не может их видеть ПО ПОСТРОЕНИЮ (ей передаётся
# только готовый вектор), список — для документации и тестов.
FORBIDDEN_STRATEGY_ATTRS = ("_next_open",)


class IsolatedExitPolicy:
    """Адаптер: model_fn(obs_vector)->bool поверх хука exit_policy(self, i).

    obs = [строка STATE по белому списку колонок] + [контекст сделки:
    в позиции, минут в сделке/1440, нереализованный P&L, откат от пика].
    Контекст считается из данных, видимых движку на баре i (Close[i],
    цена входа открытой сделки) — только прошлое.
    """

    def __init__(self, model_fn, state_df: pd.DataFrame, feature_cols: list[str]):
        missing = [c for c in feature_cols if c not in state_df.columns]
        assert not missing, f"нет колонок в STATE: {missing}"
        self.model_fn = model_fn
        self.feature_cols = list(feature_cols)
        self._time_index = state_df.index
        self._feat = np.nan_to_num(
            state_df[self.feature_cols].astype(np.float32).to_numpy(),
            nan=0.0, posinf=0.0, neginf=0.0)
        self._peak_unreal = 0.0

    def obs_dim(self) -> int:
        return self._feat.shape[1] + 4

    def __call__(self, strategy, i: int) -> bool:
        # ВАЖНО: время бара берём из данных ДВИЖКА и джойним по времени,
        # а не по позиции — защита от рассинхронизации индексов
        t = strategy.data.index[-1]
        row_pos = self._time_index.get_loc(t)
        trade = strategy.trades[-1] if strategy.trades else None
        if trade is None:
            return False
        close_i = float(strategy.data.Close[-1])
        gross = close_i / float(trade.entry_price) - 1.0
        unreal = (1.0 + gross) * (1 - 0.001) ** 2 - 1.0
        if getattr(trade, "entry_bar", None) == i:  # свежая сделка — сброс пика
            self._peak_unreal = 0.0
        self._peak_unreal = max(self._peak_unreal, unreal)
        bars_in = i - int(trade.entry_bar)
        ctx = np.array([1.0, bars_in / 1440.0, unreal, unreal - self._peak_unreal],
                       dtype=np.float32)
        obs = np.concatenate([self._feat[row_pos], ctx])
        return bool(self.model_fn(obs))
