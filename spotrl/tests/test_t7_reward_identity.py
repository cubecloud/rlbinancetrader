"""Т7 — тождественность наград: сумма поминутных = log(1 + Return/100).

Тождество верно ровно при gamma = 1.0; при gamma = 0.99 оно обязано
нарушаться — вторая проверка следит за тем, что тест живой, а не
проходит по совпадению (PLAN 5.1).
"""
from __future__ import annotations

import numpy as np
import pytest

from spotrl.config import EnvConfig
from spotrl.spec.actions import FLIP, STAY


def _one_trade_rewards(env, hold_bars: int = 120):
    """Открыть позицию, подержать, закрыть; вернуть (награды, сделка)."""
    env.reset(seed=7)
    rewards = [env.step(np.array([FLIP, 0, 0]))[1]]
    for _ in range(hold_bars):
        if not env.book.in_position:
            break
        rewards.append(env.step(np.array([STAY, 0, 0]))[1])
    if env.book.in_position:
        rewards.append(env.step(np.array([FLIP, 0, 0]))[1])
    assert env.book.closed, "сделка не закрылась — тест бессмысленен"
    return np.asarray(rewards), env.book.closed[-1]


def test_sum_of_minute_rewards_equals_trade_log_return(env):
    """Сумма поминутных наград = log(1 + Return/100) с точностью 1e-6."""
    rewards, trade = _one_trade_rewards(env)
    expected = float(np.log1p(trade.return_pct / 100.0))
    assert abs(rewards.sum() - expected) < 1e-6


def test_discounted_sum_breaks_identity(env):
    """При gamma = 0.99 тождество обязано нарушиться (проверка живости)."""
    rewards, trade = _one_trade_rewards(env)
    gamma = 0.99
    discounted = float(sum(r * gamma ** i for i, r in enumerate(rewards)))
    expected = float(np.log1p(trade.return_pct / 100.0))
    assert abs(discounted - expected) > 1e-6


def test_gamma_must_be_one():
    """Конфигурация со значением gamma != 1.0 отвергается на границе."""
    with pytest.raises(ValueError):
        EnvConfig(gamma=0.99)
