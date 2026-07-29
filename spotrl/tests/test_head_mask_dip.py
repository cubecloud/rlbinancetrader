"""Тесты маски головы 0 под ПЕРВУЮ СВОБОДУ (dip-in-position).

Проверяет СЕМАНТИКУ свободы, которую smoke-тест каркаса НЕ покрывает (на
dummy-env obs константны → все бары замаскированы, крэша нет, но семантика
не проверена). Здесь — руками собранный батч flat/transition/dip.
"""
from __future__ import annotations

import numpy as np

from spotrl.algo.head_masking import (position_head_active_dip,
                                       first_freedom_head_mask)
from spotrl.spec.actions import HEAD_POSITION, HEAD_SL, HEAD_TP

# минимальный obs с двумя нужными столбцами; mu=0, sd=1 → obs_std == raw.
_COLS = {"a_in_position": 0, "a_pos_tag_dip": 1}
_MU = np.zeros(2, np.float32)
_SD = np.ones(2, np.float32)


def _batch():
    """Три бара: flat, transition-in-position, dip-in-position."""
    return np.array([
        [0.0, 0.0],   # flat: не в позиции
        [1.0, 0.0],   # transition-in-position: в позиции, но НЕ dip
        [1.0, 1.0],   # dip-in-position: под контролем агента
    ], np.float32)


def test_dip_predicate_selects_only_dip_in_position():
    """Предикат dip: True только для dip-in-position бара."""
    active = position_head_active_dip(_batch(), _MU, _SD, _COLS)
    assert active.tolist() == [False, False, True]


def test_first_freedom_mask_head0_only_on_dip():
    """Маска головы 0 = 1 только на dip; SL/TP всегда 0 при params_own=False."""
    m = first_freedom_head_mask(_batch(), _MU, _SD, _COLS, params_own=False)
    assert m[:, HEAD_POSITION].tolist() == [0.0, 0.0, 1.0]
    # SL/TP головы при params_own=False всегда 0 (леса заморожены).
    assert m[:, HEAD_SL].tolist() == [0.0, 0.0, 0.0]
    assert m[:, HEAD_TP].tolist() == [0.0, 0.0, 0.0]


def test_predicate_survives_standardization():
    """Предикат чинит масштаб: стандартизованный obs де-стандартизуется через
    mu/sd (бинарные столбцы). Проверяем на нетривиальных mu/sd."""
    mu = np.array([0.5083, 0.4249], np.float32)   # как в манифесте reg_cd
    sd = np.array([0.4999, 0.4943], np.float32)
    raw = _batch()
    std = (raw - mu) / sd
    active = position_head_active_dip(std, mu, sd, _COLS)
    assert active.tolist() == [False, False, True]
