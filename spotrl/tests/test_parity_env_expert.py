"""Регресс-обёртка главного гейта паритета env↔expert (медленный, на артефактах).

Помечен ``@pytest.mark.slow`` — в быстром наборе (addopts `-m "not slow"`) НЕ
запускается: прогон идёт по полному ряду (1.24M/1.68M баров, десятки секунд).
Пропускается, если артефактов v7 нет. Фиксирует главный результат части 1:
пять эквити-НЕзависимых разделителей — 0 расхождений на обеих эпохах, cb=0 на
2024 (после починки cooldown-зазора). Полная логика — в
`spotrl/tests/parity_env_expert.py`.
"""
from __future__ import annotations

import os

import pytest

from spotrl.tests.parity_env_expert import run_parity

_D = os.path.expanduser("~/Data")
_EPOCHS = {
    "2021": ("sunday_tests/state_v0/state_v3_causal_2021-01_2024-03.parquet",
             "rlbinancetrader/v7signals_2021.parquet",
             "rlbinancetrader/expert_v7_2021-01_2024-03.parquet",
             "rlbinancetrader/trades_v7_labeled_2021.csv"),
    "2024": ("sunday_tests/state_v0/state_v3_causal_2024-03_2026-07.parquet",
             "rlbinancetrader/v7signals_2024.parquet",
             "rlbinancetrader/expert_v7_2024-03_2026-07.parquet",
             "rlbinancetrader/trades_v7_labeled_2024.csv"),
}


def _paths(tag: str):
    """Абсолютные пути артефактов эпохи или None, если чего-то нет."""
    parts = [os.path.join(_D, p) for p in _EPOCHS[tag]]
    return parts if all(os.path.exists(p) for p in parts) else None


@pytest.mark.slow
@pytest.mark.parametrize("tag", ["2021", "2024"])
def test_five_separators_have_zero_mismatch(tag):
    """Пять эквити-НЕзависимых разделителей — 0 расхождений env↔expert."""
    paths = _paths(tag)
    if paths is None:
        pytest.skip(f"нет артефактов v7 для эпохи {tag}")
    counts = run_parity(*paths)
    assert counts.compared > 1_000_000
    assert counts.in_position == 0
    assert counts.pos_tag == 0
    assert counts.cooldown == 0
    assert counts.entered_on_up == 0


@pytest.mark.slow
def test_cb_full_parity_on_2024():
    """cb совпадает бит-в-бит на 2024 (cooldown-зазор устранён)."""
    paths = _paths("2024")
    if paths is None:
        pytest.skip("нет артефактов v7 для эпохи 2024")
    assert run_parity(*paths).cb == 0
