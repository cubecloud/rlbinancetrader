"""Быстрые тесты честного гейта: чистота маски исключения + её семантика.

Тяжёлый прогон (saved clone + кросс-эпоховые клоны) — через CLI
`spotrl.bc.honest_gate`, в быстрый набор НЕ входит.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spotrl.bc.honest_gate import (build_exclusion_mask, build_unambiguous_flip,
                                    build_transition_position_exclusion,
                                    build_mustcopy, mechanical_tp_mask,
                                    _assert_mask_purity)


def _mini_epoch() -> pd.DataFrame:
    """Мини-датасет: транзишн-hold, dip-hold(пусто), FLIP-выходы разных причин."""
    n = 10
    df = pd.DataFrame({
        "a_in_position": np.zeros(n, np.float32),
        "bc_action": np.zeros(n, np.int64),
        "m_exit_flag": np.zeros(n, np.float32),
        "a_pos_tag_transition": np.zeros(n, np.float32),
        "a_pos_tag_dip": np.zeros(n, np.float32),
        "m_regime_code": np.zeros(n, np.float32),
        "exit_reason": [""] * n,
        "is_flip_exit": np.zeros(n, bool),
        "is_flip_entry": np.zeros(n, bool),
        "m_buy_margin": np.zeros(n, np.float32),
    })
    # бар 0: transition-hold (in-pos, transition, bull, exit_sig, STAY)
    df.loc[0, ["a_in_position", "a_pos_tag_transition", "m_regime_code",
               "m_exit_flag"]] = [1, 1, 2, 1]
    # бар 1: dip in-pos STAY БЕЗ сигнала → НЕ латч (не в исключении)
    df.loc[1, ["a_in_position", "a_pos_tag_dip", "m_regime_code"]] = [1, 1, 2]
    # бар 2: legflip выход (однозначный FLIP)
    df.loc[2, ["is_flip_exit", "exit_reason"]] = [True, "legflip"]
    # бар 3: signal выход (НЕ однозначный, поверхность свободы)
    df.loc[3, ["is_flip_exit", "exit_reason"]] = [True, "signal"]
    # бар 4: вход с положительным margin (однозначный)
    df.loc[4, ["is_flip_entry", "m_buy_margin"]] = [True, 0.3]
    # бар 5: вход с отрицательным margin (не однозначный)
    df.loc[5, ["is_flip_entry", "m_buy_margin"]] = [True, -0.2]
    return df


def test_mask_purity_assert_passes():
    """assert чистоты: маска берёт только df, без ссылок на логиты клона."""
    _assert_mask_purity()  # не должен бросить


def test_exclusion_is_only_hold_latches():
    """В исключении — только транзишн/oracle hold STAY-бары, вход пуст."""
    df = _mini_epoch()
    ex = build_exclusion_mask(df)
    assert ex["transition_hold"].sum() == 1 and ex["transition_hold"][0]
    assert ex["oracle_hold"].sum() == 0          # dip без сигнала не держится
    assert ex["entry_exclude"].sum() == 0        # ВХОД пуст
    # не-латч dip-in-pos STAY (бар 1) НЕ в исключении
    assert not ex["exclude"][1]
    # FLIP-бары (2,3,4,5) не STAY → не в исключении
    for b in (2, 3, 4, 5):
        assert not ex["exclude"][b]


def test_unambiguous_flip_excludes_signal_exits():
    """Однозначный FLIP = legflip/tp/b2b выходы + вход с margin>0 (не signal, не margin<0)."""
    df = _mini_epoch()
    u = build_unambiguous_flip(df)
    assert u["exit_unambig"][2] and not u["exit_unambig"][3]     # legflip да, signal нет
    assert u["entry_unambig"][4] and not u["entry_unambig"][5]   # margin>0 да, <0 нет
    assert u["exit_signal"][3]


def test_mask_does_not_depend_on_any_score():
    """Маска идентична при любой подмешанной d (она d не видит вовсе)."""
    df = _mini_epoch()
    m1 = build_exclusion_mask(df)["exclude"]
    # добавим фиктивную колонку логитов — маска обязана не измениться
    df2 = df.copy()
    df2["d_clone"] = np.random.default_rng(0).normal(size=len(df))
    m2 = build_exclusion_mask(df2)["exclude"]
    assert np.array_equal(m1, m2)


def _mini_exit_epoch() -> pd.DataFrame:
    """Мини-фрейм выходов: legflip, механич.tp (M), signal-tp (must-copy)."""
    n = 6
    df = pd.DataFrame({
        "a_in_position": np.ones(n, np.float32),
        "bc_action": np.zeros(n, np.int64),
        "a_pos_tag_transition": np.zeros(n, np.float32),
        "exit_reason": [""] * n,
        "is_flip_exit": np.zeros(n, bool),
        "is_flip_entry": np.zeros(n, bool),
        "m_exit_flag": np.zeros(n, np.float32),
        "m_buy_margin": np.zeros(n, np.float32),
    })
    df.loc[0, ["is_flip_exit", "exit_reason", "bc_action"]] = [True, "legflip", 1]
    df.loc[1, ["is_flip_exit", "exit_reason", "bc_action"]] = [True, "tp", 1]  # M
    df.loc[2, ["is_flip_exit", "exit_reason", "m_exit_flag", "bc_action"]] = \
        [True, "tp", 1, 1]                                       # signal-tp
    df.loc[3, ["is_flip_exit", "exit_reason", "bc_action"]] = [True, "b2b", 1]
    # бар 4: transition-in-position STAY (свобода RL)
    df.loc[4, "a_pos_tag_transition"] = 1
    return df


def test_transition_position_exclusion_is_pure_and_stay_only():
    """(c): transition-in-position STAY-исключение — только STAY-бары, чистое."""
    df = _mini_exit_epoch()
    ex = build_transition_position_exclusion(df)
    assert ex[4] and ex.sum() == 1                       # только бар 4
    # джерримендеринг-инвариант: исключение НЕ задевает ни одного FLIP-выхода.
    fe = df["is_flip_exit"].to_numpy().astype(bool)
    assert int((ex & fe).sum()) == 0


def test_mechanical_tp_and_mustcopy_partition():
    """M = tp & m_exit_flag==0; must-copy = legflip|b2b|signal-tp (не M)."""
    df = _mini_exit_epoch()
    M = mechanical_tp_mask(df)
    mc = build_mustcopy(df)
    assert M[1] and M.sum() == 1                          # только механич. tp
    assert mc[0] and mc[2] and mc[3] and not mc[1]        # legflip,signal-tp,b2b
    assert int((mc & M).sum()) == 0                       # разбиение чисто


def test_new_pure_functions_pass_purity_assert():
    """Все чистые функции гейта (incl. must-copy) проходят assert чистоты."""
    _assert_mask_purity()
