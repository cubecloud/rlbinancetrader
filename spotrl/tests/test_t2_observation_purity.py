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


def test_equity_drawdown_signed_and_path_parity(env):
    """Просадка эквити в obs — знаковая (≤0, форма дока) и совпадает у двух путей."""
    env.reset(seed=3)
    idx = env.config.obs_spec.index_of("w_equity_drawdown")
    actions = [np.array([FLIP, 0, 0])] + [np.array([STAY, 0, 0])] * 50
    for a in actions:
        obs, *_ = env.step(a)
        assert obs[idx] <= 1e-7                              # форма дока: ≤0
        expected = env._equity / env._peak_equity - 1.0
        assert abs(obs[idx] - expected) < 1e-6               # obs = знаковая просадка
        assert np.array_equal(obs, env.observe())            # паритет step vs observe
