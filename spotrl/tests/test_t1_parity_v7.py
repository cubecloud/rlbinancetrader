"""Т1 — парность с v7: состав сделок и Return против эталона sunday.

Тест медленный (полный прогон движка, минуты) и требует данных sunday,
поэтому пропускается, если файлов нет. Гейт Э0.1: 162/162 сделки совпадают,
|ΔReturn| <= 0.1 п.п.
"""
from __future__ import annotations

import os

import pytest

from spotrl.tests.conftest import REF_STATE, REF_TRADES

_STATE = os.path.expanduser(REF_STATE)
_TRADES = os.path.expanduser(REF_TRADES)
_HAVE_DATA = os.path.exists(_STATE) and os.path.exists(_TRADES)


@pytest.mark.slow
@pytest.mark.skipif(not _HAVE_DATA,
                    reason="нет эталонных данных sunday (state parquet / judge_trades_v7.csv)")
def test_parity_v7_2024_2026():
    """162 сделки бар-в-бар и совпадение return_pct с эталоном."""
    from spotrl.bridge.v7parity import check_parity, run_v7

    _, trades, _ = run_v7(_STATE)
    result = check_parity(trades, _TRADES)
    assert result.n_trades == 162, f"сделок {result.n_trades}, ожидалось 162"
    assert result.passed, f"расхождения: {result.problems}"
    assert result.max_return_diff_pp <= 0.1
