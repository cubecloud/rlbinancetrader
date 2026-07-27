"""Slow тест value-warmup end-to-end на синтетике (без реальной среды).

Проверяет полный путь run_warmup: сбор reward замокан синтетическим (env не
нужен), критик обучается ТОЛЬКО на trade-close return-to-go, EV считается,
логиты головы 0 остаются бит-в-бит, модель+манифест сохраняются. Под slow.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from spotrl.spec.observation import ObservationSpec


def _write_epoch(path, epoch, seed):
    """Крошечный bc_clone parquet с in-position сегментами (для trade-close)."""
    rng = np.random.default_rng(seed)
    names = list(ObservationSpec.v2().names)
    n = 2000
    X = rng.normal(size=(n, len(names))).astype(np.float32)
    df = pd.DataFrame(X, columns=names)
    df["bar"] = np.arange(n)
    y = np.zeros(n, dtype=np.int64)
    flip = rng.choice(n, size=20, replace=False)
    y[flip] = 1
    df["m_entry_signal"] = 0.0
    df.loc[flip, "m_entry_signal"] = 1.0
    # in-position сегменты: каждые 100 баров 30 баров «в позиции».
    inpos = np.zeros(n, np.float32)
    for s in range(0, n, 100):
        inpos[s:s + 30] = 1.0
    df["a_in_position"] = inpos
    df["bc_action"] = y
    df["is_flip_entry"] = False
    df["is_flip_exit"] = False
    df.loc[flip[:10], "is_flip_entry"] = True
    df.loc[flip[10:], "is_flip_exit"] = True
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


class _FakeEnv:
    """Мини-заглушка среды: шагает _t от 0, усечение на последнем баре."""

    def __init__(self, n):
        """n = число решающих баров (= len(df))."""
        self._n = n
        self._n_bars = n + 1
        self._t = 0
        self._start = 0
        self._exit_is_signal = False

    def reset(self, seed=None):
        """Сброс на бар 0."""
        self._t = 0
        self._start = 0
        return None, {}

    def step(self, action):
        """Шаг: детерминированный reward, усечение при достижении конца."""
        r = 0.001 * (1 + int(action[0]))
        self._t += 1
        trunc = self._t >= self._n
        return None, r, False, trunc, {}


@pytest.mark.slow
def test_collect_epoch_rewards_fake_env(tiny_data, monkeypatch):
    """collect_epoch_rewards / collect_rewards / trade_close на заглушке среды."""
    import pandas as pd
    from pathlib import Path
    import spotrl.bc.build_clone_dataset as B
    import spotrl.bc.value_warmup as V

    df = pd.read_parquet(tiny_data / "bc_clone_v7_2024.parquet")
    monkeypatch.setattr(B, "_make_env", lambda *a, **k: _FakeEnv(len(df)))

    cache = tiny_data / "rw.npy"
    r1 = V.collect_epoch_rewards("2024", "s", "sig", df, cache)
    assert len(r1) == len(df) and cache.exists()
    r2 = V.collect_epoch_rewards("2024", "s", "sig", df, cache)  # из кеша
    assert np.array_equal(r1, r2)

    # trade-close на реальном a_in_position синтетики + episode-window диагностика.
    ip = df["a_in_position"].to_numpy(np.float32)
    g = V.trade_close_return_to_go(r1, ip)
    assert g.shape == r1.shape and np.all(g[ip < 0.5] == 0.0)
    gw = V.episode_window_return_to_go(r1, 100)
    assert gw.shape == r1.shape

    # collect_rewards через заглушку (обе эпохи), пути state игнорируются.
    R, inpos, meta = V.collect_rewards(str(tiny_data))
    assert len(R) == len(inpos)


@pytest.mark.slow
def test_run_warmup_end_to_end(tiny_data, monkeypatch):
    """run_warmup: критик обучается, логиты замёрзли, модель сохранена."""
    import spotrl.bc.train_clone as T
    import spotrl.bc.value_warmup as V

    monkeypatch.setattr(T, "N_EPOCHS", 3)
    monkeypatch.setattr(T, "BATCH", 512)
    monkeypatch.setattr(T, "HIDDEN", (32, 32))
    monkeypatch.setattr(V, "N_EPOCHS", 5)
    monkeypatch.setattr(V, "BATCH", 512)

    # обучить и сохранить крошечный клон (даёт модель+манифест со скейлером).
    model_path = str(tiny_data / "clone")
    monkeypatch.setattr(sys, "argv",
                        ["train", "--data", str(tiny_data), "--out", model_path])
    orig_train = T.train
    monkeypatch.setattr(T, "train", lambda *a, **k: orig_train(*a, n_epochs=3))
    T.main()

    # замокать сбор reward (env не нужен): синтетический поток, выровнен с meta.
    def fake_collect_rewards(data_dir, base=None):
        """Синтетический поток reward + a_in_position (среда не нужна)."""
        _, y, _, meta = V.load_pooled(data_dir)
        rng = np.random.default_rng(0)
        R = rng.normal(scale=0.01, size=len(y)).astype(np.float32)
        in_pos = np.zeros(len(y), np.float32)
        import pandas as pd
        from pathlib import Path
        for e, m in meta["per_epoch"].items():
            df = pd.read_parquet(Path(data_dir) / f"bc_clone_v7_{e}.parquet")
            in_pos[m["idx"]] = df["a_in_position"].to_numpy(np.float32)
        return R, in_pos, meta

    monkeypatch.setattr(V, "collect_rewards", fake_collect_rewards)

    out_path = str(tiny_data / "clone_vw")
    r = V.run_warmup(str(tiny_data), model_path, out_path, every=2)

    assert r["logits_bitexact"] is True                 # голова заморожена
    assert r["max_logit_delta"] == 0.0
    assert np.isfinite(r["ev_after"]) and np.isfinite(r["ev_before"])
    assert r["target"] == "trade_close_return_to_go_gamma1"
    assert (tiny_data / "clone_vw.zip").exists()
    assert (tiny_data / "clone_vw.manifest.json").exists()


@pytest.mark.slow
def test_cli_value_warmup(tiny_data, monkeypatch):
    """CLI main() value_warmup исполняется на синтетике (сбор reward замокан)."""
    import spotrl.bc.train_clone as T
    import spotrl.bc.value_warmup as V

    monkeypatch.setattr(T, "N_EPOCHS", 3)
    monkeypatch.setattr(T, "BATCH", 512)
    monkeypatch.setattr(T, "HIDDEN", (32, 32))
    monkeypatch.setattr(V, "N_EPOCHS", 4)

    model_path = str(tiny_data / "clone")
    orig_train = T.train
    monkeypatch.setattr(T, "train", lambda *a, **k: orig_train(*a, n_epochs=3))
    monkeypatch.setattr(sys, "argv",
                        ["train", "--data", str(tiny_data), "--out", model_path])
    T.main()

    def fake_collect_rewards(data_dir, base=None):
        """Синтетический поток reward + a_in_position (среда не нужна)."""
        _, y, _, meta = V.load_pooled(data_dir)
        R = np.random.default_rng(0).normal(scale=0.01, size=len(y)).astype(np.float32)
        import pandas as pd
        from pathlib import Path
        in_pos = np.zeros(len(y), np.float32)
        for e, m in meta["per_epoch"].items():
            df = pd.read_parquet(Path(data_dir) / f"bc_clone_v7_{e}.parquet")
            in_pos[m["idx"]] = df["a_in_position"].to_numpy(np.float32)
        return R, in_pos, meta

    monkeypatch.setattr(V, "collect_rewards", fake_collect_rewards)
    out = str(tiny_data / "vw.json")
    monkeypatch.setattr(sys, "argv",
                        ["vw", "--data", str(tiny_data), "--model-in", model_path,
                         "--model-out", str(tiny_data / "clone_vw"),
                         "--val-trade-every", "2", "--out", out])
    V.main()
