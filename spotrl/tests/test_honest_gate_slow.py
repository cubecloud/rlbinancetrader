"""Slow тест переопределённого гейта на синтетике: тяжёлые пути end-to-end.

Строит две крошечные эпохи со всеми нужными колонками (must-copy выходы, dip-holds,
transition-holds, входы), обучает+сохраняет клон, прогоняет run_insample /
run_outofepoch / separability_guard / _verdict / main. Проверяет, что код
исполняется и метрики в допустимых диапазонах (НЕ что клон проходит — это на
реальных данных). Под slow.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from spotrl.spec.observation import ObservationSpec


def _write_epoch(path, epoch, seed):
    """Синтетическая эпоха: must-copy выходы (отделимый ключ m_leg_dn), dip/
    transition holds, входы с margin. Отделимость обеспечивает высокий L2 AUC."""
    rng = np.random.default_rng(seed)
    names = list(ObservationSpec.v2().names)
    n = 2400
    X = rng.normal(size=(n, len(names))).astype(np.float32)
    df = pd.DataFrame(X, columns=names)
    df["bar"] = np.arange(n)
    # обнулить ключевые колонки, задать по ролям.
    for c in ("a_in_position", "a_pos_tag_dip", "a_pos_tag_transition",
              "m_exit_flag", "m_leg_dn", "m_regime_code", "m_entry_signal",
              "m_buy_margin", "a_cooldown_remain"):
        df[c] = 0.0
    y = np.zeros(n, np.int64)
    is_entry = np.zeros(n, bool); is_exit = np.zeros(n, bool)
    exit_reason = np.array([""] * n, dtype=object)

    # in-position сегменты каждые 60 баров по 25 (dip); часть — transition.
    seg_starts = list(range(50, n - 60, 60))
    for k, s in enumerate(seg_starts):
        e = s + 25
        df.loc[s:e, "a_in_position"] = 1.0
        if k % 4 == 0:
            df.loc[s:e, "a_pos_tag_transition"] = 1.0
            df.loc[s:e, "m_regime_code"] = 2.0
            df.loc[s + 20, "m_exit_flag"] = 1.0        # transition-hold сигнал
        else:
            df.loc[s:e, "a_pos_tag_dip"] = 1.0
            # must-copy выход на баре e: legflip (ключ m_leg_dn=1).
            is_exit[e] = True; y[e] = 1; exit_reason[e] = "legflip"
            df.loc[e, "m_leg_dn"] = 1.0
        # вход на баре s (margin>0, ключ m_entry_signal=1).
        is_entry[s] = True; y[s] = 1
        df.loc[s, "m_entry_signal"] = 1.0; df.loc[s, "m_buy_margin"] = 0.5

    # один механический tp (M) и один signal-tp (must-copy) для покрытия ветвей.
    df.loc[70, ["a_in_position", "a_pos_tag_dip"]] = [1.0, 1.0]
    is_exit[75] = True; y[75] = 1; exit_reason[75] = "tp"; df.loc[75, "m_exit_flag"] = 0.0
    is_exit[135] = True; y[135] = 1; exit_reason[135] = "tp"; df.loc[135, "m_exit_flag"] = 1.0

    df["bc_action"] = y
    df["is_flip_entry"] = is_entry
    df["is_flip_exit"] = is_exit
    df["exit_reason"] = exit_reason
    df["pos_tag"] = "dip"
    df["bc_weight"] = 1.0
    df["epoch"] = epoch
    df.to_parquet(path / f"bc_clone_v7_{epoch}.parquet")


@pytest.fixture()
def tiny_data(tmp_path):
    """Каталог с двумя крошечными эпохами bc_clone (со всеми колонками гейта)."""
    _write_epoch(tmp_path, "2021", 1)
    _write_epoch(tmp_path, "2024", 2)
    return tmp_path


@pytest.mark.slow
def test_redefined_gate_end_to_end(tiny_data, monkeypatch):
    """run_insample + run_outofepoch + separability + _verdict исполняются."""
    import spotrl.bc.train_clone as T
    import spotrl.bc.honest_gate as H

    monkeypatch.setattr(T, "N_EPOCHS", 4)
    monkeypatch.setattr(T, "BATCH", 1024)
    monkeypatch.setattr(T, "HIDDEN", (32, 32))
    orig_train = T.train
    monkeypatch.setattr(T, "train", lambda *a, **k: orig_train(*a, n_epochs=4))

    model_path = str(tiny_data / "clone")
    monkeypatch.setattr(sys, "argv",
                        ["train", "--data", str(tiny_data), "--out", model_path])
    T.main()

    insample, meta, excl, unamb = H.run_insample(str(tiny_data), model_path)
    for e in H.EPOCHS:
        pe = insample["per_epoch"][e]
        assert pe["n_mustcopy"] > 0
        assert 0.0 <= pe["recall_mustcopy_insample"] <= 1.0
        assert pe["n_transition_excl"] > 0
    sep = insample["separability"]
    assert 0.0 <= sep["min_auc"] <= 1.0

    oof = H.run_outofepoch(str(tiny_data), seed=0, n_epochs=4)
    assert len(oof["auc"]) == 2
    v = H._verdict(insample, oof)
    assert set(v) >= {"a_recall_mustcopy_insample", "b_margin_mustcopy",
                      "c_transition_apriori", "d_separability", "e_auc",
                      "clone_passes_redefined_gate"}


@pytest.mark.slow
def test_gate_cli_main(tiny_data, monkeypatch):
    """CLI honest_gate.main() исполняется на синтетике и пишет JSON."""
    import spotrl.bc.train_clone as T
    import spotrl.bc.honest_gate as H

    monkeypatch.setattr(T, "N_EPOCHS", 3)
    monkeypatch.setattr(T, "BATCH", 1024)
    monkeypatch.setattr(T, "HIDDEN", (32, 32))
    orig_train = T.train
    monkeypatch.setattr(T, "train", lambda *a, **k: orig_train(*a, n_epochs=3))

    model_path = str(tiny_data / "clone")
    monkeypatch.setattr(sys, "argv",
                        ["train", "--data", str(tiny_data), "--out", model_path])
    T.main()

    out = str(tiny_data / "gate.json")
    monkeypatch.setattr(sys, "argv",
                        ["gate", "--data", str(tiny_data), "--model", model_path,
                         "--n-epochs", "3", "--out", out])
    H.main()
    assert (tiny_data / "gate.json").exists()
