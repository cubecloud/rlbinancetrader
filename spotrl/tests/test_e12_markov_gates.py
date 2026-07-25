"""Юнит-тесты логики гейтов немарковости Э1.2 на синтетической разметке.

Реальные числа на BC-разметке v7 считает драйвер (см. отчёт handoff); он
требует sunday-окружения и перезапуска v7. Здесь проверяется САМА логика
разделения на маленьких кадрах: корректный кадр проходит, вырожденный и
немарковский — нет.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spotrl.tests.e12_markov_gates import entry_gate, exit_gate


def _expert(bars, in_pos, tag, mech, exit_sig, cooldown, cb):
    """Собрать минимальный экспертный лог для гейтов."""
    n = len(bars)
    return pd.DataFrame({
        "bar": bars, "in_pos_before": in_pos, "pos_tag": tag,
        "mech_exit": mech, "agent_exit": [False] * n, "entry_placed": [False] * n,
        "exit_sig": exit_sig, "cooldown_active": cooldown, "cb_active": cb,
        "time": pd.date_range("2024-01-01", periods=n, freq="D")})


def test_exit_gate_separated_by_entered_on_up():
    """entered_on_up разделяет EXIT/HOLD среди leg_dn dip-баров -> unseparated=0."""
    # trade A: entry_bar=3 -> leg_dn[2]=False -> eou=1, leg_dn-бары 3,4 -> EXIT
    # trade B: entry_bar=8 -> leg_dn[7]=True  -> eou=0, leg_dn-бары 8,9 -> HOLD
    leg = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1], dtype=bool)
    bars = np.arange(1, 10)
    in_pos = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1], dtype=bool)  # bars 3,4,5 и 8,9
    tag = np.array(["", "", "dip", "dip", "dip", "", "", "dip", "dip"])
    mech = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=bool)  # EXIT на 3,4
    exit_sig = np.zeros(9, dtype=bool)
    expert = _expert(bars, in_pos, tag, mech, exit_sig,
                     np.zeros(9, bool), np.zeros(9, bool))
    trades = pd.DataFrame({"entry_bar": [3, 8], "exit_bar": [5, 9]})
    res = exit_gate(expert, trades, leg)
    assert res.unseparated == 0
    assert res.population > 0


def test_exit_gate_cosepararates_signal_exits():
    """Сигнальный выход `_exit[i]` — второй разделитель наравне с entered_on_up.

    Среди eou=0 dip-баров (legflip неактивен) EXIT vs HOLD объясняется ровно
    признаком exit_sig; со-разделение даёт 0 неразделённых.
    """
    leg = np.array([1, 1, 1, 1, 1], dtype=bool)      # eou=0 (leg[0]=True)
    bars = np.arange(1, 5)
    expert = _expert(bars, np.array([1, 1, 1, 1], bool),
                     np.array(["dip"] * 4),
                     np.array([1, 0, 1, 0], bool),      # EXIT на сигнальных барах
                     np.array([1, 0, 1, 0], bool),      # exit_sig=True там же
                     np.zeros(4, bool), np.zeros(4, bool))
    trades = pd.DataFrame({"entry_bar": [1], "exit_bar": [4]})
    res = exit_gate(expert, trades, leg)
    assert res.unseparated == 0


def test_entry_gate_separated_by_rule_state():
    """ENTER/WAIT разделены {cooldown, cb, halt_day} -> unseparated=0, не вырожден."""
    n = 8
    bars = np.arange(1, n + 1)
    entry_signal = np.ones(n + 1, dtype=bool)          # сигнал на всех барах
    # flat всюду; ENTER там где нет cooldown/cb, WAIT где есть
    cooldown = np.array([0, 0, 1, 1, 0, 0, 1, 1], bool)
    placed = (~cooldown)
    expert = _expert(bars, np.zeros(n, bool), np.array([""] * n),
                     np.zeros(n, bool), np.zeros(n, bool), cooldown,
                     np.zeros(n, bool))
    expert["entry_placed"] = placed
    res = entry_gate(expert, entry_signal)
    assert res.unseparated == 0
    assert res.population == n


def test_entry_gate_detects_unseparated():
    """Если метка не объясняется разделителями — гейт видит неразделённые бары."""
    n = 4
    bars = np.arange(1, n + 1)
    entry_signal = np.ones(n + 1, dtype=bool)
    # одинаковые разделители (всё False), но разные метки -> неразделённо
    placed = np.array([True, False, True, False])
    expert = _expert(bars, np.zeros(n, bool), np.array([""] * n),
                     np.zeros(n, bool), np.zeros(n, bool),
                     np.zeros(n, bool), np.zeros(n, bool))
    expert["entry_placed"] = placed
    res = entry_gate(expert, entry_signal)
    assert res.unseparated == n
    assert not res.passed
