"""Slow интеграционные тесты BC-пайплайна на синтетических parquet.

Прогоняют реальные пути обучения (logsumexp + hard-negative mining, сборка
sb3-PPO), teacher-forced гейта и closed-loop диагностики на крошечных данных —
проверяют, что код исполняется end-to-end и метрики согласованы. Под slow-маркером
(в быстрый набор не входят). Полный прогон на реальных ~2.9М строках — через CLI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spotrl.spec.observation import ObservationSpec


def _write_epoch(path, epoch, seed):
    """Синтетический bc_clone parquet: отделимый сигнал в m_entry_signal."""
    rng = np.random.default_rng(seed)
    names = list(ObservationSpec.v2().names)
    n = 4000
    X = rng.normal(size=(n, len(names))).astype(np.float32)
    df = pd.DataFrame(X, columns=names)
    df["bar"] = np.arange(n)
    y = np.zeros(n, dtype=np.int64)
    flip_bars = rng.choice(n, size=20, replace=False)
    y[flip_bars] = 1
    # отделимый ключ: FLIP-бары имеют m_entry_signal=1, иначе 0.
    df["m_entry_signal"] = 0.0
    df.loc[flip_bars, "m_entry_signal"] = 1.0
    df["bc_action"] = y
    df["is_flip_entry"] = False
    df["is_flip_exit"] = False
    df.loc[flip_bars[:10], "is_flip_entry"] = True
    df.loc[flip_bars[10:], "is_flip_exit"] = True
    df["pos_tag"] = "dip"
    df["exit_reason"] = ""
    df["bc_weight"] = 1.0
    df["epoch"] = epoch
    df.to_parquet(path / f"bc_clone_v7_{epoch}.parquet")


@pytest.fixture()
def tiny_data(tmp_path):
    """Каталог с двумя крошечными эпохами bc_clone."""
    _write_epoch(tmp_path, "2021", 1)
    _write_epoch(tmp_path, "2024", 2)
    return tmp_path


@pytest.mark.slow
def test_bc_pipeline_end_to_end(tiny_data, monkeypatch):
    """Обучение → калибровка → teacher-forced → closed-loop исполняются end-to-end."""
    import spotrl.bc.train_clone as T
    from spotrl.bc import closed_loop, eval_gate

    monkeypatch.setattr(T, "N_EPOCHS", 4)
    monkeypatch.setattr(T, "HN_SIZE", 128)
    monkeypatch.setattr(T, "BATCH", 1024)
    monkeypatch.setattr(T, "HIDDEN", (32, 32))

    X_raw, y, w, meta = T.load_pooled(str(tiny_data))
    mu, sd = T.fit_scaler(X_raw)
    Xn = T.apply_scaler(X_raw, mu, sd)
    _, policy = T.build_policy(Xn.shape[1], 0)
    tr = T.train(policy, Xn, y, w, 0, n_epochs=4)
    assert np.isfinite(tr["final_loss"])

    logits = T.head0_logits(policy, Xn)
    assert logits.shape == (Xn.shape[0], 2)
    d = logits[:, 1] - logits[:, 0]
    d_by = {e: {"stay": d[m["idx"]][m["is_stay"]],
                "flip_all": d[m["idx"]][m["is_entry"] | m["is_exit"]]}
            for e, m in meta["per_epoch"].items()}
    cal = T.calibrate_shift(d_by)
    assert np.isfinite(cal.gap_nat)

    # teacher-forced гейт исполняется, метрики в допустимых диапазонах.
    s = cal.s_star if cal.feasible else 0.0
    tf = eval_gate.teacher_forced(policy, str(tiny_data), s, mu, sd, seeds=5)
    for r in tf:
        assert 0.0 <= r.fpr_mean <= 1.0
        assert 0.0 <= r.recall_entry_mean <= 1.0

    # closed-loop диагностика исполняется, доля чистых в [0,1].
    cl = closed_loop.run_closed_loop(str(tiny_data), policy, s, mu, sd, seeds=5)
    for c in cl.values():
        assert 0.0 <= c["clean_frac"] <= 1.0
        assert abs(c["clean_frac_analytic"] - c["clean_frac"]) <= 1.0


@pytest.mark.slow
def test_cli_entrypoints(tiny_data, monkeypatch):
    """CLI main() обучения, гейта и коллизионного теста исполняются на синтетике."""
    import sys
    import spotrl.bc.train_clone as T
    from spotrl.bc import eval_gate, exit_collision

    monkeypatch.setattr(T, "N_EPOCHS", 3)
    monkeypatch.setattr(T, "HN_SIZE", 128)
    monkeypatch.setattr(T, "BATCH", 1024)
    monkeypatch.setattr(T, "HIDDEN", (32, 32))
    # train() читает N_EPOCHS через default аргумента, связанный при def → передаём явно.
    orig_train = T.train
    monkeypatch.setattr(T, "train", lambda *a, **k: orig_train(*a, n_epochs=3))

    model_path = str(tiny_data / "policy")
    monkeypatch.setattr(sys, "argv",
                        ["train", "--data", str(tiny_data), "--out", model_path])
    T.main()

    gate_out = str(tiny_data / "gate.json")
    monkeypatch.setattr(sys, "argv",
                        ["gate", "--data", str(tiny_data), "--model", model_path,
                         "--seeds", "3", "--closed-loop", "--out", gate_out])
    eval_gate.main()

    coll_out = str(tiny_data / "coll.json")
    monkeypatch.setattr(sys, "argv",
                        ["coll", "--data", str(tiny_data), "--out", coll_out])
    exit_collision.main()


@pytest.mark.slow
def test_cross_epoch_generalization_runs(tiny_data):
    """Кросс-эпоховая проверка обобщения исполняется и возвращает обе стороны."""
    import spotrl.bc.train_clone as T
    from spotrl.bc import generalization
    from spotrl.bc.generalization import run_cross_epoch
    res = run_cross_epoch(str(tiny_data), n_epochs=3)
    assert {r.train_epoch for r in res} == {"2021", "2024"}
    for r in res:
        assert np.isfinite(r.gap_in) and np.isfinite(r.gap_out)
        assert isinstance(r.generalizes, bool)
        assert set(r.describe()) >= {"gap_in", "gap_out", "generalizes"}


@pytest.mark.slow
def test_cross_epoch_cli(tiny_data, monkeypatch):
    """CLI main() кросс-эпоховой проверки исполняется на синтетике."""
    import sys
    from spotrl.bc import generalization
    orig = generalization.run_cross_epoch
    monkeypatch.setattr(generalization, "run_cross_epoch",
                        lambda data, **k: orig(data, n_epochs=3))
    out = str(tiny_data / "gen.json")
    monkeypatch.setattr(sys, "argv", ["gen", "--data", str(tiny_data), "--out", out])
    generalization.main()
