"""Быстрые тесты BC-обучения и гейта копирования (без полного датасета).

Тяжёлые прогоны (обучение на ~2.9М строк, гейт на 100 сидов) запускаются через
CLI модулей `spotrl.bc.{train_clone,eval_gate,exit_collision}` и НЕ входят в
быстрый набор — здесь проверяется логика на синтетике за доли секунды.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spotrl.bc.exit_collision import run_epoch
from spotrl.bc.generalization import _auc, _recall_fpr_at_train_thr
from spotrl.bc.train_clone import (EXTRA_COLS, apply_scaler, calibrate_shift,
                                   derive_extra, fit_scaler)


def _synthetic_epoch(tp_collision: bool) -> pd.DataFrame:
    """Мини-датасет in-position: FLIP-выходы + STAY, с/без tp-коллизии."""
    rng = np.random.default_rng(0)
    n = 200
    cols = {c: rng.normal(size=n).astype(np.float32) for c in (
        "m_exit_flag", "m_leg_dn", "a_unreal_pnl")}
    df = pd.DataFrame(cols)
    df["a_in_position"] = 1.0
    df["bc_action"] = 0
    df["is_flip_exit"] = False
    df["pos_tag"] = "dip"
    df["exit_reason"] = ""
    # два tp-выхода dip с unreal у порога 0.12
    df.loc[0, ["bc_action", "is_flip_exit", "a_unreal_pnl", "exit_reason"]] = [1, True, 0.121, "tp"]
    df.loc[1, ["bc_action", "is_flip_exit", "a_unreal_pnl", "exit_reason"]] = [1, True, 0.120, "tp"]
    df["m_exit_flag"] = 0.0
    df["m_leg_dn"] = 0.0
    # STAY-бары: все ниже порога, кроме (опц.) одного коллизионного выше 0.12
    df.loc[2:, "a_unreal_pnl"] = 0.05
    if tp_collision:
        df.loc[5, "a_unreal_pnl"] = 0.15   # STAY dip выше tp-порога → коллизия
    return df


def test_exit_collision_separable_k_zero():
    """Отделимый выход → k_tp=0, вердикт разрешает A=0."""
    rep = run_epoch(_synthetic_epoch(tp_collision=False), "syn")
    assert rep.k_tp == 0
    assert "разрешён" in rep.verdict


def test_exit_collision_band_detected():
    """STAY-бар выше tp-порога → k_tp>=1 обнаружен."""
    rep = run_epoch(_synthetic_epoch(tp_collision=True), "syn")
    assert rep.k_tp >= 1


def test_scaler_roundtrip_and_clip():
    """Стандартизация центрирует/масштабирует и клипует в пределах CLIP_STD."""
    X = np.random.default_rng(1).normal(10, 5, size=(1000, 4)).astype(np.float32)
    mu, sd = fit_scaler(X)
    Xn = apply_scaler(X, mu, sd)
    assert abs(Xn.mean()) < 0.1 and abs(Xn.std() - 1.0) < 0.1
    assert Xn.max() <= 10.0 + 1e-6 and Xn.min() >= -10.0 - 1e-6


def test_scaler_constant_feature_neutral():
    """Константный признак → sd=1 (нейтраль, без деления на 0)."""
    X = np.ones((10, 2), dtype=np.float32)
    X[:, 1] = np.arange(10)
    mu, sd = fit_scaler(X)
    assert sd[0] == 1.0
    assert np.isfinite(apply_scaler(X, mu, sd)).all()


def test_calibrate_shift_feasible_when_separated():
    """Разведённые популяции (зазор>20 nat) → существует s* (feasible)."""
    d = {"e": {"stay": np.full(10000, -30.0), "flip_all": np.full(400, 30.0)}}
    cal = calibrate_shift(d)
    assert cal.feasible and cal.s_lo <= cal.s_star <= cal.s_hi
    assert cal.gap_nat >= cal.gap_needed


def test_calibrate_shift_infeasible_when_overlap():
    """Пересекающиеся хвосты → скалярный сдвиг не существует (not feasible)."""
    rng = np.random.default_rng(2)
    d = {"e": {"stay": rng.normal(0, 5, 10000), "flip_all": rng.normal(2, 5, 400)}}
    cal = calibrate_shift(d)
    assert not cal.feasible
    # s* (FPR-anchored) ВСЕГДА определён, даже при infeasible интервале.
    assert np.isfinite(cal.s_star)


def test_derive_extra_cooldown_active_threshold_at_zero():
    """`a_cooldown_active` = (cd>0): 0 ровно при cd==0, 1 при любом cd>0.

    Фикс «остаток паузы»: истинные входы cd==0 → флаг 0; дозревающая пауза
    (кратна 1/423) → флаг 1. Порог у нуля точен, без float-шума.
    """
    cd = np.array([0.0, 1.0 / 423.0, 0.0165, 0.5, 1.0, 0.0], dtype=np.float32)
    df = pd.DataFrame({"a_cooldown_remain": cd})
    extra = derive_extra(df)
    assert extra.shape == (len(cd), len(EXTRA_COLS))
    np.testing.assert_array_equal(
        extra[:, 0], np.array([0, 1, 1, 1, 1, 0], dtype=np.float32))
    # флаг идеально делит cd==0 (истинные входы) от cd>0 (пауза активна).
    assert (extra[cd == 0.0, 0] == 0.0).all()
    assert (extra[cd > 0.0, 0] == 1.0).all()


def test_derive_extra_survives_standardization():
    """Редкий булев флаг после стандартизации даёт хорошо разделённый z.

    Активная пауза редка (~0.6%): flag=1 после (x-mu)/sd уходит в большой +z,
    flag=0 — в малый -z. Разделяющий сигнал НЕ тонет (в отличие от сырого cd).
    """
    cd = np.zeros(10000, dtype=np.float32)
    cd[:60] = 0.005                      # ~0.6% активной паузы
    df = pd.DataFrame({"a_cooldown_remain": cd})
    flag = derive_extra(df)
    mu, sd = fit_scaler(flag)
    z = apply_scaler(flag, mu, sd)
    z_active = z[flag[:, 0] > 0.5].mean()
    z_idle = z[flag[:, 0] < 0.5].mean()
    assert z_active - z_idle > 10.0     # разнос классов флага велик


def test_generalization_auc_helper_edges():
    """_auc: NaN при одном классе, идеальный порядок → 1.0."""
    assert np.isnan(_auc(np.array([1.0, 2.0]), np.array([0, 0])))
    assert np.isnan(_auc(np.array([1.0, 2.0]), np.array([1, 1])))
    assert _auc(np.array([0.1, 0.2, 0.9, 1.0]),
                np.array([0, 0, 1, 1])) == 1.0


def test_generalization_recall_fpr_helper():
    """_recall_fpr_at_train_thr: порог по train, recall/FPR на test; NaN без pos."""
    # 2 train-позитива, target=0.995 → k=1 → порог = верхний позитив (1.0).
    tr_s = np.array([0.0, 0.1, 0.9, 1.0]); tr_y = np.array([0, 0, 1, 1])
    te_s = np.array([0.05, 1.5]); te_y = np.array([0, 1])
    rec, fpr = _recall_fpr_at_train_thr(tr_s, tr_y, te_s, te_y, target=0.995)
    assert rec == 1.0 and fpr == 0.0
    # нет позитивов в train → NaN.
    r2, f2 = _recall_fpr_at_train_thr(tr_s, np.zeros(4, int), te_s, te_y)
    assert np.isnan(r2) and np.isnan(f2)
