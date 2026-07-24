"""Т2 — чистота наблюдения: повторный вызов даёт побитово тот же вектор.

Ловит дефект старого кода: мутацию `peak_unreal` внутри вычисления
наблюдения (`policyio.py:57-59`, `env.py:89`).
"""
from __future__ import annotations

import numpy as np

from spotrl.spec.actions import FLIP, STAY


def test_obs_is_pure_out_of_position(env):
    """Два вызова наблюдения подряд вне позиции — побитово равны."""
    env.reset(seed=1)
    first = env._obs()
    second = env._obs()
    assert np.array_equal(first, second)


def test_obs_is_pure_in_position(env):
    """То же в позиции: наблюдение не двигает пик незакрытой доходности."""
    env.reset(seed=1)
    env.step(np.array([FLIP, 0, 0]))
    for _ in range(20):
        env.step(np.array([STAY, 0, 0]))
    peak_before = env.book.open_trade.peak_unreal
    first = env._obs()
    second = env._obs()
    assert np.array_equal(first, second)
    assert env.book.open_trade.peak_unreal == peak_before


def test_reserved_slots_are_constant(env):
    """Зарезервированные слоты всегда 0.0 (PLAN 4.5.2, приём 1)."""
    env.reset(seed=2)
    obs = env._obs()
    n_reserved = len(env.config.obs_spec.reserved)
    assert np.all(obs[-n_reserved:] == 0.0)
