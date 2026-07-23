"""Data-слой ветки (П1): источник истины — PG, кэш — ускоритель.

Материализация окна = запуск sunday state-билдера (их код, их slice-тест
каузальности — П3/П7) subprocess'ом; результат регистрируется в манифесте.
Загрузка = чтение parquet по манифесту и раскладка на (ohlcv_df, features_df)
в интерфейсе, который ожидает среда (П9: как в BinanceEnvCash).

Пример:
    from strategy_rl.datalayer import StateCache
    sc = StateCache()
    key = sc.materialize("2026-06-01", "2026-07-01")   # минуты CPU, идемпотентно
    ohlcv_df, features_df = sc.load(key)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

from strategy_rl.sundaybridge import SUNDAY_ROOT, load_pg_env, sunday_git_head

BUILDER = os.path.join(SUNDAY_ROOT, "tooling", "audit", "state_v0_builder.py")
BUILDER_OUT_DIR = os.path.expanduser("~/Data/sunday_tests/state_v0")
CACHE_DIR = os.path.expanduser("~/Data/rlbinancetrader/state_cache")

# группы колонок state_v3 (см. sunday_state_datasets.md)
OHLCV_COLS = ["open", "high", "low", "close", "volume",
              "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote"]
TRADEBOOK_PREFIX = "tb_"   # в parquet нули; в среде считаются из трейдбука АГЕНТА
EXPECTED_NCOLS = 105


def _sha256(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class StateCache:
    """Кэш материализованных окон state с манифестами."""

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    # ---------- ключи и манифесты ----------

    @staticmethod
    def key(start: str, end: str) -> str:
        s = pd.Timestamp(start).strftime("%Y%m%d%H%M")
        e = pd.Timestamp(end).strftime("%Y%m%d%H%M")
        return f"rlv6_{s}_{e}"

    def _manifest_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.manifest.json")

    def manifest(self, key: str) -> dict | None:
        p = self._manifest_path(key)
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)

    # ---------- материализация (builder subprocess) ----------

    def materialize(self, start: str, end: str, force: bool = False,
                    timeout_s: int = 3600) -> str:
        """Строит state-окно sunday-билдером; идемпотентно по манифесту.

        Возвращает key. Билдер сам падает при провале slice-теста
        каузальности — тогда манифест не пишется.
        """
        key = self.key(start, end)
        man = self.manifest(key)
        if man and not force and os.path.exists(man["parquet"]) \
                and _sha256(man["parquet"]) == man["sha256"]:
            return key

        load_pg_env()
        env = dict(os.environ)
        env["PYTHONPATH"] = SUNDAY_ROOT
        proc = subprocess.run(
            [sys.executable, BUILDER, str(start), str(end), key],
            cwd=SUNDAY_ROOT, env=env, capture_output=True, text=True,
            timeout=timeout_s)
        if proc.returncode != 0:
            raise RuntimeError(
                f"builder failed (rc={proc.returncode}):\n"
                f"stdout tail: {proc.stdout[-2000:]}\n"
                f"stderr tail: {proc.stderr[-2000:]}")
        if "causality slice-test: PASS" not in proc.stdout:
            raise RuntimeError("builder finished without slice-test PASS — "
                               f"не доверяем выходу.\nstdout: {proc.stdout[-2000:]}")

        parquet = os.path.join(BUILDER_OUT_DIR, f"state_v3_{key}.parquet")
        if not os.path.exists(parquet):
            raise RuntimeError(f"builder output not found: {parquet}")

        man = {
            "key": key,
            "window": {"start": str(pd.Timestamp(start)),
                       "end": str(pd.Timestamp(end))},
            "parquet": parquet,
            "sha256": _sha256(parquet),
            "builder": BUILDER,
            "sunday_commit": sunday_git_head(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "slice_test": "PASS",
            "builder_stdout_tail": proc.stdout[-800:],
        }
        with open(self._manifest_path(key), "w") as f:
            json.dump(man, f, ensure_ascii=False, indent=1)
        return key

    # ---------- загрузка ----------

    def load(self, key: str, verify_hash: bool = False):
        """-> (ohlcv_df, features_df); features — всё, кроме OHLCV и tb_*.

        tb_* сознательно НЕ отдаются (П4/1.3.1: контекст сделки среда
        вычисляет из трейдбука агента; предвычисленные tb_ — только для
        IL-датасета, отдельным путём).
        """
        man = self.manifest(key)
        if man is None:
            raise KeyError(f"no manifest for {key}; call materialize() first")
        if verify_hash and _sha256(man["parquet"]) != man["sha256"]:
            raise RuntimeError(f"parquet hash mismatch for {key} — кэш испорчен")
        df = pd.read_parquet(man["parquet"])
        if len(df.columns) != EXPECTED_NCOLS:
            raise RuntimeError(
                f"schema drift: {len(df.columns)} cols != {EXPECTED_NCOLS}; "
                "сверьте версию билдера и обновите EXPECTED_NCOLS осознанно")
        ohlcv_df = df[OHLCV_COLS].copy()
        feat_cols = [c for c in df.columns
                     if c not in OHLCV_COLS and not c.startswith(TRADEBOOK_PREFIX)]
        features_df = df[feat_cols].copy()
        return ohlcv_df, features_df

    def registered(self) -> list[dict]:
        out = []
        for f in sorted(os.listdir(self.cache_dir)):
            if f.endswith(".manifest.json"):
                with open(os.path.join(self.cache_dir, f)) as fh:
                    out.append(json.load(fh))
        return out
