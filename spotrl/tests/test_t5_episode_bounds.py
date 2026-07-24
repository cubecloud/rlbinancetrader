"""Т5 — границы эпизода: обрыв по длине даёт truncated, а не terminated.

Если вернуть terminated, SB3 не бутстрапит хвост ценности и обучение
получает систематически заниженную оценку (PLAN 5.2).
"""
from __future__ import annotations

import numpy as np

from spotrl.spec.actions import STAY


def test_length_cut_is_truncated(env):
    """На конце эпизода truncated=True при terminated=False."""
    env.reset(seed=5)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < env.config.episode_len + 5:
        _, _, terminated, truncated, _ = env.step(np.array([STAY, 0, 0]))
        steps += 1
    assert truncated is True
    assert terminated is False


def test_open_trade_is_closed_at_end(env, state):
    """Ни одна сделка не остаётся без исхода: на последнем баре — выход."""
    env.reset(seed=6)
    env._start = len(state) - 60          # подводим эпизод к концу данных
    env._t = env._start
    env.step(np.array([1, 0, 0]))         # FLIP: открыть позицию
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(np.array([STAY, 0, 0]))
    assert not env.book.in_position
    assert env.book.closed[-1].exit_reason in ("sl", "tp", "end")
