"""Регресс-обёртка главного гейта паритета env↔expert (медленный, на артефактах).

Помечен ``@pytest.mark.slow`` — в быстром наборе (addopts `-m "not slow"`) НЕ
запускается: прогон идёт по полному ряду (1.24M/1.68M баров, десятки секунд).
Пропускается, если артефактов v7 нет. Фиксирует ПОЛНЫЙ паритет 6/6: пять
эквити-НЕзависимых разделителей и circuit breaker — 0 расхождений на ОБЕИХ
эпохах. cb доведён до нуля мировым леджером эквити (конвенция v7: cash=10M,
sizing 0.9999, целые лоты, односторонняя комиссия), считаемым из собственной
книги среды; по-сделочный ассерт леджера (size/цены) против trades CSV
подтверждает, что паритет получен из совпадения эквити-траектории, а не из
компенсирующих ошибок. Полная логика — в `spotrl/tests/parity_env_expert.py`.
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
def test_full_parity_six_of_six(tag):
    """Полный паритет 6/6 env↔expert: пять разделителей + cb = 0 на обеих эпохах.

    Плюс по-сделочный ассерт мирового леджера против trades CSV: целые лоты
    (size) и цены (adjusted-вход, сырой выход) совпадают на всех сделках, кроме
    терминального forced-end (исключён из ценовой сверки по конвенции клампа
    exit_bar в v7). Это доказывает, что cb-паритет — из совпадения эквити, а не
    из компенсирующих ошибок.
    """
    paths = _paths(tag)
    if paths is None:
        pytest.skip(f"нет артефактов v7 для эпохи {tag}")
    counts = run_parity(*paths)
    assert counts.compared > 1_000_000
    assert counts.in_position == 0
    assert counts.pos_tag == 0
    assert counts.cooldown == 0
    assert counts.entered_on_up == 0
    assert counts.cb == 0
    assert counts.trades_checked > 100
    assert counts.ledger_size == 0
    assert counts.ledger_price == 0
