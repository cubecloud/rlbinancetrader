"""Тесты харнесса oracle-резерва (этап Э0.2).

Быстрый набор — чистые функции на синтетике (без движка): anti-future,
open(i+1)-семантика, partition-identity, дельта ≥ 0, хвосты, cooldown-счётчик.
Медленный — движковый smoke: чистая v7 = 162 сделки на эталоне 2024-26.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from spotrl.analysis import oracle_reserve as orc
from spotrl.tests.conftest import REF_STATE, REF_TRADES

COMM = 0.001


def _make_trades_and_open():
    """Синтетика: 3 dip-сделки (SL с пред-SL пиком, signal+, tp) + 1 transition.

    open[5]=130 — пик внутри SL-сделки (fill bar 5, decision bar 4).
    open[15]=1000 — БУДУЩЕЕ за пределами всех сделок (для anti-future).
    """
    n = 30
    open_arr = np.full(n, 90.0)         # фон ниже SL-цены (не даёт ложных выходов)
    # SL-сделка: entry 2, exit 10, entry_price 100, v7 ret -0.07
    open_arr[5] = 130.0                 # пред-SL пик
    # signal-сделка: entry 12, exit 18, entry_price 100, v7 ret +0.02
    open_arr[14] = 110.0                # пик > exit v7
    # tp-сделка: entry 20, exit 25, entry_price 100, v7 ret +0.12
    open_arr[22] = 105.0                # ниже TP-выхода -> oracle НЕ рубит
    open_arr[11] = 1000.0               # бар в зазоре между сделками (вне всех)
    high_arr = open_arr + 1.0           # High >= Open (реалистично)
    trades = pd.DataFrame({
        "entry_bar": [2, 12, 20, 26],
        "exit_bar": [10, 18, 25, 29],
        "entry_price": [100.0, 100.0, 100.0, 100.0],
        "return_pct": [-0.07, 0.02, 0.12, 0.05],
        "pos_tag": ["dip", "dip", "dip", "transition"],
        "exit_reason": ["sl", "signal", "tp", "b2b"],
    })
    return trades, open_arr, high_arr


def test_delta_nonnegative_and_dip_only():
    """Frozen-дельта неотрицательна и считается только по dip-сделкам."""
    trades, op, hi = _make_trades_and_open()
    res = orc.frozen_trade_reserve(trades, op, hi, COMM)
    assert len(res) == 3, "transition-сделка не должна попасть в резерв"
    assert (res["delta"] >= 0).all(), "дельта обязана быть ≥ 0"


def test_open_next_bar_semantics():
    """Цена выхода = open[decision_bar+1]*(1-comm), не close/high, не open[d]."""
    trades, op, hi = _make_trades_and_open()
    res = orc.frozen_trade_reserve(trades, op, hi, COMM)
    sl = res[res["exit_reason"] == "sl"].iloc[0]
    assert sl["best_fill_bar"] == 5
    assert sl["decision_bar"] == 4
    expected = 130.0 * (1 - COMM) / 100.0 - 1.0
    assert abs(sl["ret_oracle"] - expected) < 1e-12
    # High НЕ участвует в цене выхода (только в справочном потолке)
    assert sl["ret_oracle"] < 140.0 * (1 - COMM) / 100.0 - 1.0


def test_anti_future_beyond_trade():
    """Бары за пределами сделки (open[15]=1000) не влияют на резерв."""
    trades, op, hi = _make_trades_and_open()
    res_a = orc.frozen_trade_reserve(trades, op, hi, COMM)
    op2 = op.copy()
    op2[11] = 5.0                        # зазор между сделками 1 и 2 (вне всех)
    op2[19] = 5.0                        # зазор между сделками 2 и 3 (вне всех)
    res_b = orc.frozen_trade_reserve(trades, op2, hi, COMM)
    assert np.allclose(res_a["delta"].to_numpy(), res_b["delta"].to_numpy())


def test_eligible_window_boundaries():
    """Кандидаты только в [entry+2 .. exit-1]; вход/бар механики исключены."""
    trades, op, hi = _make_trades_and_open()
    # пик ровно на entry_bar+1 (=3) и на exit_bar (=10) — оба ВНЕ окна
    op = op.copy()
    op[:] = 90.0                         # фон ниже SL-цены (ранний выход не берётся)
    op[3] = 500.0                        # entry+1 (позиция ещё не открыта)
    op[10] = 500.0                       # бар механического исполнения
    res = orc.frozen_trade_reserve(trades, op, hi, COMM)
    sl = res[res["exit_reason"] == "sl"].iloc[0]
    assert sl["best_fill_bar"] == -1, "выход в недопустимый бар не должен браться"
    assert sl["delta"] == 0.0


def test_short_trade_no_eligible_bars():
    """Сделка без eligible-баров исполнения даёт нулевую дельту."""
    trades = pd.DataFrame({
        "entry_bar": [5], "exit_bar": [6], "entry_price": [100.0],
        "return_pct": [-0.07], "pos_tag": ["dip"], "exit_reason": ["sl"]})
    op = np.full(20, 200.0)
    res = orc.frozen_trade_reserve(trades, op, None, COMM)
    assert res.iloc[0]["delta"] == 0.0


def test_partition_identity():
    """Сумма бакетных дельт равна полному frozen-резерву."""
    trades, op, hi = _make_trades_and_open()
    res = orc.frozen_trade_reserve(trades, op, hi, COMM)
    dec = orc.decompose(res, n_dip=3, n_dip_sl=1)
    assert orc.partition_check(res, dec)
    assert abs(sum(dec.by_reason_pp.values()) - dec.reserve_total_pp) < 1e-9


def test_component_b_denominators():
    """Компонента (б) считается по обоим знаменателям (n_dip, n_dip_sl)."""
    trades, op, hi = _make_trades_and_open()
    res = orc.frozen_trade_reserve(trades, op, hi, COMM)
    dec = orc.decompose(res, n_dip=3, n_dip_sl=1)
    # компонента (б) — только SL-бакет
    assert dec.component_b_sum_pp == pytest.approx(dec.by_reason_pp["sl"])
    # два знаменателя отличаются ровно во столько раз, во сколько n_dip/n_dip_sl
    assert dec.component_b_per_dip == pytest.approx(dec.component_b_sum_pp / 3)
    assert dec.component_b_per_sl == pytest.approx(dec.component_b_sum_pp / 1)


def test_tp_preserved_and_tails():
    """Доля сохранённых TP = 1.0; метрики хвостов в допуске."""
    trades, op, hi = _make_trades_and_open()
    res = orc.frozen_trade_reserve(trades, op, hi, COMM)
    tm = orc.tail_metrics(res)
    assert tm["tp_preserved_frac"] == 1.0, "tp-сделку oracle рубить не должен"
    assert 0.0 <= tm["tail_capture_ratio"] <= 1.0 + 1e-9
    # leave-top-1 убирает крупнейшую дельту -> доля оставшегося < 1
    assert tm["leave_top_k"]["leave_top_1_frac_kept"] < 1.0


def test_sl_bucket_fragility():
    """Leave-one-out по SL-бакету: вердикт на одной сделке ловится."""
    # один SL-бакет с дельтой 0.3687 (=36.87 п.п.); n_dip=3 -> линия 0.5*3=1.5 п.п.
    trades, op, hi = _make_trades_and_open()
    res = orc.frozen_trade_reserve(trades, op, hi, COMM)
    frag = orc.sl_bucket_fragility(res, n_dip=3, threshold_pct=0.5)
    assert frag["n_sl"] == 1
    assert frag["verdict_base"] == "ALIVE"
    # единственная SL-сделка: удаление её -> сумма 0 < линии -> мертва после 1
    assert frag["dead_after_dropping_one"] is True
    assert frag["drops_to_dead"] == 1
    # при заведомо большом n_dip линия недостижима -> мертва сразу (drops=0)
    frag2 = orc.sl_bucket_fragility(res, n_dip=1000, threshold_pct=0.5)
    assert frag2["verdict_base"] == "DEAD"


def test_new_cooldowns_counts_non_sl_exits():
    """Новые cooldown вводятся ранними выходами не-SL сделок."""
    trades, op, hi = _make_trades_and_open()
    res = orc.frozen_trade_reserve(trades, op, hi, COMM)
    # ранний выход есть на SL и signal сделках; НОВЫЙ cooldown — только signal
    assert orc.new_cooldowns_from_oracle(res) == 1


def test_build_exit_policy_decision_bars():
    """exit_policy срабатывает ровно на предрасчитанных decision_bars."""
    trades, op, hi = _make_trades_and_open()
    res = orc.frozen_trade_reserve(trades, op, hi, COMM)
    pol = orc.build_exit_policy(res)
    assert 4 in pol.decision_bars                 # SL-сделка, decision bar 4

    class _Stub:
        pass
    assert pol(_Stub(), 4) is True
    assert pol(_Stub(), 999) is False


# --- медленный движковый smoke ------------------------------------------------
_STATE = os.path.expanduser(REF_STATE)
_TRADES = os.path.expanduser(REF_TRADES)
_HAVE = os.path.exists(_STATE) and os.path.exists(_TRADES)


@pytest.mark.slow
@pytest.mark.skipif(not _HAVE, reason="нет эталонных данных sunday")
def test_engine_pure_v7_parity():
    """Чистая v7 через харнесс = 162 сделки на эталоне 2024-26."""
    from spotrl.analysis.engine_run import run_engine
    stats, trades = run_engine(_STATE, exit_policy=None)
    assert len(trades) == 162
    assert abs(float(stats["Return [%]"]) - 267.46) < 0.5


@pytest.mark.slow
@pytest.mark.skipif(not _HAVE, reason="нет эталонных данных sunday")
def test_engine_disarm_cooldown_and_ohlc():
    """Ветка disarm-cooldown исполняется; load_ohlc отдаёт OHLC."""
    from spotrl.analysis.engine_run import load_ohlc, run_engine
    ohlc = load_ohlc(_STATE)
    assert {"Open", "High", "Low", "Close"}.issubset(ohlc.columns)
    # «всегда выходить» — задевает агентскую ветку выхода в disarm-режиме
    _, trades = run_engine(_STATE, exit_policy=lambda s, i: True,
                           disarm_cooldown=True)
    assert len(trades) > 0
