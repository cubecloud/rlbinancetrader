"""Синтетический ряд для тестов — фикстура без внешних данных (PLAN 1.5.3).

Быстрый набор тестов обязан идти без файлов с диска, поэтому здесь
генератор детерминированного ряда из N баров со всеми обязательными
колонками state-таблицы.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spotrl.data.state_dataset import StateDataset, from_frame


def make_synthetic_frame(n_bars: int = 5_000, seed: int = 0,
                         start_price: float = 100.0) -> pd.DataFrame:
    """Детерминированный минутный ряд с колонками каузального state.

    Args:
        n_bars: число баров.
        seed: сид генератора; при одном сиде ряд побитово одинаков.
        start_price: стартовая цена.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.001, size=n_bars)
    close = start_price * np.exp(np.cumsum(steps))
    open_ = np.concatenate([[start_price], close[:-1]])
    spread = np.abs(rng.normal(0.0, 0.0005, size=n_bars)) * close
    frame = pd.DataFrame(
        {"open": open_,
         "high": np.maximum(open_, close) + spread,
         "low": np.minimum(open_, close) - spread,
         "close": close,
         "volume": rng.uniform(1.0, 10.0, size=n_bars),
         "q_buy": rng.uniform(0.0, 1.0, size=n_bars),
         "q_sell": rng.uniform(0.0, 1.0, size=n_bars),
         "regime_code": rng.integers(0, 3, size=n_bars),
         "leg_dn": rng.random(n_bars) < 0.2},
        index=pd.date_range("2024-03-01", periods=n_bars, freq="min"))
    return frame


def make_synthetic_state(n_bars: int = 5_000, seed: int = 0) -> StateDataset:
    """StateDataset на синтетическом ряде (обёртка над make_synthetic_frame)."""
    return from_frame(make_synthetic_frame(n_bars=n_bars, seed=seed),
                      source=f"<synthetic:{n_bars}:{seed}>")
