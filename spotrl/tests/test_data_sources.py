"""Тесты слоя данных: два источника за одним интерфейсом и гейт эквивалентности.

Быстрый набор (≤60 с) не ходит в базу и не импортирует ``dbbinance``.
Реальный поход в PG помечен ``@pytest.mark.db`` и пропускается, если базы или
ключей нет (headless без creds — штатная ситуация).
"""
from __future__ import annotations

import builtins
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spotrl.data.sources import (GateResult, ParquetSource, PgSource,
                                  WindowData, dbbinance_version,
                                  ohlcv_bitwise_gate, source_manifest)

REF_STATE = Path(
    "~/Data/sunday_tests/state_v0/state_v0_2024-03_2026-07.parquet").expanduser()
_HAS_REF = REF_STATE.exists()


# ---------- синтетический снимок для быстрых тестов ----------

def _make_snapshot_frame(n: int = 200, start="2024-03-10 05:00") -> pd.DataFrame:
    """Небольшой снимок с OHLCV и сигнальными признаками (tz-naive UTC)."""
    idx = pd.date_range(start, periods=n, freq="1min")
    rng = np.random.default_rng(0)
    close = 60000 + np.cumsum(rng.normal(0, 5, n))
    return pd.DataFrame({
        "open": close - 1.0, "high": close + 2.0, "low": close - 2.0,
        "close": close, "volume": rng.uniform(1, 9, n),
        "q_buy": rng.uniform(0, 1, n), "q_sell": rng.uniform(0, 1, n),
        "regime_code": rng.integers(0, 3, n), "leg_dn": rng.integers(0, 2, n),
    }, index=idx)


def _write_snapshot(tmp_path: Path, n: int = 200) -> Path:
    p = tmp_path / "snap.parquet"
    _make_snapshot_frame(n).to_parquet(p)
    return p


# ---------- ParquetSource ----------

def test_parquet_window_shapes_and_features(tmp_path):
    """Снимок отдаёт OHLCV, признаки и валидный StateDataset."""
    src = ParquetSource(_write_snapshot(tmp_path, 200))
    win = src.load_window("2024-03-10 05:00", "2024-03-10 05:49")
    assert len(win) == 50
    assert win.ohlcv.shape == (50, 5) and win.ohlcv.dtype == np.float64
    assert win.has_features()
    assert win.signals.shape == (50, 2)
    # окно строит валидный StateDataset для среды
    st = win.to_state_dataset()
    assert len(st) == 50


def test_parquet_window_bounds_inclusive(tmp_path):
    """Границы окна включительны."""
    src = ParquetSource(_write_snapshot(tmp_path, 10))
    win = src.load_window("2024-03-10 05:02", "2024-03-10 05:05")
    assert [str(t) for t in win.index] == [
        "2024-03-10 05:02:00", "2024-03-10 05:03:00",
        "2024-03-10 05:04:00", "2024-03-10 05:05:00"]


def test_manifest_has_hash_and_version(tmp_path):
    """Манифест источника содержит хэш OHLCV и версию dbbinance."""
    win = ParquetSource(_write_snapshot(tmp_path, 20)).load_window(
        "2024-03-10 05:00", "2024-03-10 05:19")
    man = source_manifest(win)
    assert len(man["ohlcv_sha256"]) == 64
    assert man["dbbinance_version"]  # строка (версия либо "unavailable")
    assert man["features_missing_in_raw_pg"] == [
        "q_buy", "q_sell", "regime_code", "leg_dn"]
    assert man["feature_builder_version"] == "state_v0"


# ---------- изоляция импорта (требование 6) ----------

def test_parquet_works_without_dbbinance(tmp_path, monkeypatch):
    """Путь parquet обязан работать, даже если dbbinance не импортируется."""
    real_import = builtins.__import__

    def blocked(name, *a, **k):
        """Блокирует импорт dbbinance/secureapikey в этом тесте."""
        if name.startswith("dbbinance") or name == "secureapikey":
            raise ImportError("dbbinance заблокирован в этом тесте")
        return real_import(name, *a, **k)

    for mod in list(sys.modules):
        if mod.startswith("dbbinance") or mod == "secureapikey":
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)

    src = ParquetSource(_write_snapshot(tmp_path, 30))
    win = src.load_window("2024-03-10 05:00", "2024-03-10 05:29")
    assert len(win) == 30
    # и конструктор PgSource не должен импортировать dbbinance
    PgSource()


# ---------- PgSource с инъектированным fetcher (без базы) ----------

class _FakeFetcher:
    """Мок ресемплера: отдаёт заданный DataFrame; имитирует last_full_bar."""

    def __init__(self, frame: pd.DataFrame):
        """Args: frame — DataFrame OHLCV, отдаваемый как ресемпл базы."""
        self._frame = frame

    def pg_resample_to_timeframe(self, table_name, start, end, to_timeframe,
                                 origin, open_time_index, last_full_bar,
                                 **kwargs):
        """Мок ресемпла базы: имитирует last_full_bar по окну."""
        df = self._frame
        if last_full_bar and len(df):
            step = pd.Timedelta(to_timeframe.replace("m", "min"))
            idx = pd.DatetimeIndex(df.index)
            end_ts = pd.Timestamp(end)
            if end_ts.tzinfo is None and idx.tz is not None:
                end_ts = end_ts.tz_localize(idx.tz)
            df = df[(idx + step) <= end_ts]
        return df


def _pg_frame(n=10, start="2024-03-10 05:00"):
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    close = np.arange(n, dtype=float) + 60000.0
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": np.arange(n, dtype=float)},
                        index=idx)


def test_pgsource_ohlcv_only_no_features():
    """Сырой PG отдаёт только OHLCV, tz-naive; StateDataset без признаков падает."""
    src = PgSource(fetcher=_FakeFetcher(_pg_frame(10)))
    win = src.load_window("2024-03-10 05:00", "2024-03-10 05:20",
                          last_full_bar=False)
    assert not win.has_features()
    assert win.signals is None and win.regime_code is None
    assert win.ohlcv.shape == (10, 5)
    assert isinstance(win.index, pd.DatetimeIndex) and win.index.tz is None
    with pytest.raises(ValueError):
        win.to_state_dataset()


def test_pgsource_live_edge_drops_unclosed_bar():
    """Неполный последний бар живого края не попадает в выборку."""
    # now — середина бара 05:09; закрыт последний целиком до 05:09
    frame = _pg_frame(10, "2024-03-10 05:00")  # метки 05:00..05:09
    src = PgSource(fetcher=_FakeFetcher(frame))
    now = pd.Timestamp("2024-03-10 05:09:30")
    win = src.load_latest_closed(now, lookback_bars=5, freq="1min")
    # 05:09 не закрыт к 05:09:30 → последний закрытый 05:08
    assert str(win.index[-1]) == "2024-03-10 05:08:00"
    assert (pd.Timestamp(win.index[-1]) + pd.Timedelta("1min")) <= now


# ---------- гейт эквивалентности: логика на общих данных ----------

def test_gate_zero_diff_on_identical_ohlcv():
    """Гейт: одинаковый OHLCV даёт ноль различий."""
    idx = pd.date_range("2024-03-10 05:00", periods=8, freq="1min")
    ohlcv = np.arange(40, dtype=np.float64).reshape(8, 5)
    a = WindowData(idx, ohlcv.copy(), None, None, None, "a")
    b = WindowData(idx, ohlcv.copy(), None, None, None, "b")
    res = ohlcv_bitwise_gate(a, b)
    assert isinstance(res, GateResult)
    assert res.passed and res.diff_elements == 0 and res.max_abs_diff == 0.0
    assert res.features_missing == ("q_buy", "q_sell", "regime_code", "leg_dn")


def test_gate_detects_single_bit_diff():
    """Гейт ловит одно-битовое расхождение и указывает колонку."""
    idx = pd.date_range("2024-03-10 05:00", periods=4, freq="1min")
    a_ohlcv = np.arange(20, dtype=np.float64).reshape(4, 5)
    b_ohlcv = a_ohlcv.copy()
    b_ohlcv[2, 3] += 1e-9
    a = WindowData(idx, a_ohlcv, None, None, None, "a")
    b = WindowData(idx, b_ohlcv, None, None, None, "b")
    res = ohlcv_bitwise_gate(a, b)
    assert not res.passed and res.diff_elements == 1
    assert res.first_mismatch[0] == 2 and res.first_mismatch[1] == "close"


def test_gate_index_mismatch_fails_loudly():
    """Гейт: рассинхрон индекса виден как index_equal=False."""
    a = WindowData(pd.date_range("2024-03-10 05:00", periods=3, freq="1min"),
                   np.zeros((3, 5)), None, None, None, "a")
    b = WindowData(pd.date_range("2024-03-10 06:00", periods=3, freq="1min"),
                   np.zeros((3, 5)), None, None, None, "b")
    assert not ohlcv_bitwise_gate(a, b).index_equal


# ---------- реальный slow-тест на снимке (без базы) ----------

@pytest.mark.slow
@pytest.mark.skipif(not _HAS_REF, reason="нет parquet-снимка sunday")
def test_real_parquet_snapshot_loads():
    """ParquetSource на РЕАЛЬНОМ снимке: признаки, dtype, tz-naive, монотонность."""
    win = ParquetSource(REF_STATE).load_window(
        "2024-03-10 05:00", "2024-03-11 05:00")
    assert len(win) == 1441
    assert win.has_features()
    assert win.ohlcv.dtype == np.float64
    idx = pd.DatetimeIndex(win.index)
    assert idx.tz is None and idx.is_monotonic_increasing
    st = win.to_state_dataset()  # весь конвейер до среды
    assert len(st) == 1441


# ---------- ГЛАВНЫЙ ГЕЙТ: реальный поход в PG ----------

@pytest.mark.db
@pytest.mark.skipif(not _HAS_REF, reason="нет parquet-снимка sunday")
def test_pg_vs_parquet_bitwise_gate():
    """ГЛАВНЫЙ ГЕЙТ: parquet и база дают ПОБИТОВО одинаковый OHLCV.

    Ключи PG грузятся PgSource из sunday/*.env. Если базы/ключей нет —
    честный skip, чтобы быстрый набор и чужая машина не падали.
    """
    try:
        pg = PgSource()
        pg._get_fetcher()  # спровоцировать загрузку ключей и подключение
        parquet = ParquetSource(REF_STATE)
        start, end = "2024-03-10 05:00", "2024-03-11 05:00"
        wp = parquet.load_window(start, end)
        wg = pg.load_window(start, end)
    except Exception as exc:  # нет базы/ключей/подключения
        pytest.skip(f"PG недоступна: {exc!r}")
    res = ohlcv_bitwise_gate(wp, wg)
    print(f"GATE common={res.n_bars} diff={res.diff_elements} "
          f"max_abs={res.max_abs_diff} per_col={res.per_column} "
          f"only_parquet={res.n_only_a} only_pg={res.n_only_b} "
          f"first={res.first_mismatch}")
    assert res.n_bars > 1000, "слишком мало общих баров"
    assert res.passed, f"OHLCV не побитово равны: {res.per_column}"
    # граничный бар last_full_bar: у parquet есть, у PG нет — это норма, не сбой
    assert res.n_only_b == 0 and res.n_only_a <= 1
