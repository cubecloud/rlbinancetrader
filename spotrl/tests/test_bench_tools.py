"""Дымовой тест инструментов замера: они обязаны запускаться и не врать.

Проверяется не скорость (она зависит от машины), а работоспособность:
гейт эквивалентности ловит подмену прогона, замер возвращает положительное
число шагов в секунду.
"""
from __future__ import annotations

import numpy as np

from spotrl.bench.equivalence import (collect_runs, compare, compare_digest,
                                      digest, rollout, scenarios)
from spotrl.bench.speed import make_actions, measure_raw_env


def test_equivalence_gate_is_deterministic():
    """Два одинаковых прогона совпадают побитово: 0 расхождений."""
    first = collect_runs(n_steps=300, verbose=False)
    second = collect_runs(n_steps=300, verbose=False)
    assert compare(first, second, verbose=False) == 0


def test_equivalence_gate_catches_a_difference():
    """Гейт не декоративный: подмена одного элемента даёт FAIL."""
    first = collect_runs(n_steps=300, verbose=False)
    spoiled = {k: v.copy() for k, v in first.items()}
    spoiled["base_obs"][10, 0] += np.float32(1.0)
    assert compare(first, spoiled, verbose=False) == 1


def test_scenarios_cover_trade_branches():
    """Сценарий 'full' действительно открывает и закрывает сделки."""
    config, p_flip = scenarios()["full"]
    run = rollout(config, p_flip, n_steps=3_000)
    assert len(run["trades_num"]) > 0
    assert set(run["reasons"]) - {""}


def test_speed_measurement_returns_positive_rate():
    """Замер скорости возвращает осмысленное число шагов в секунду."""
    assert measure_raw_env(n_steps=2_000, repeats=1) > 0.0
    assert make_actions(50).shape == (50, 3)


def test_digest_matches_bitwise_comparison():
    """Хэш-вид гейта ведёт себя так же, как побитовое сравнение."""
    first = collect_runs(n_steps=300, verbose=False)
    second = collect_runs(n_steps=300, verbose=False)
    assert compare_digest(digest(first), digest(second), verbose=False) == 0
    spoiled = {k: v.copy() for k, v in first.items()}
    spoiled["base_rew"][5] += 1e-12
    assert compare_digest(digest(first), digest(spoiled), verbose=False) == 1
