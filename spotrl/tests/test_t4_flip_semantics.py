"""Т4 — семантика FLIP: вне позиции всегда открывает, в позиции всегда закрывает.

Ни одно действие не должно быть молча проигнорировано: недопустимых
действий в v0.5 нет по построению пространства (масок нет).
"""
from __future__ import annotations

import numpy as np

from spotrl.spec.actions import FLIP, STAY


def test_flip_opens_when_flat(env):
    """FLIP вне позиции открывает позицию по open следующего бара."""
    env.reset(seed=3)
    assert not env.book.in_position
    env.step(np.array([FLIP, 0, 0]))
    assert env.book.in_position


def test_flip_closes_when_in_position(env):
    """FLIP в позиции закрывает её и записывает сделку в книгу."""
    env.reset(seed=3)
    env.step(np.array([FLIP, 0, 0]))
    env.step(np.array([FLIP, 0, 0]))
    assert not env.book.in_position
    assert len(env.book.closed) == 1
    assert env.book.closed[0].exit_reason == "agent"


def test_stay_never_changes_position(env):
    """STAY не меняет состояние позиции ни в одном из состояний."""
    env.reset(seed=3)
    for _ in range(50):
        env.step(np.array([STAY, 0, 0]))
    assert not env.book.in_position
    env.step(np.array([FLIP, 0, 0]))
    for _ in range(50):
        env.step(np.array([STAY, 0, 0]))
    assert env.book.in_position or env.book.closed[-1].exit_reason in ("sl", "tp", "end")


def test_entry_bar_params_are_taken_from_heads(env):
    """На баре входа SL/TP берутся из голов согласно спецификации."""
    env.reset(seed=4)
    env.step(np.array([FLIP, 1, 1]))
    trade = env.book.open_trade
    assert trade is not None
    assert trade.sl_frac == env.config.action_spec.sl_buckets[1]
    assert trade.tp_frac == env.config.action_spec.tp_buckets[1]
