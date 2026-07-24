"""Тесты слоя данных: два источника за одним обобщённым интерфейсом и гейт.

Обобщённый :class:`WindowData` держит именованную матрицу колонок; PgSource
тянет из базы сырьё + поток ордеров (расширенный kline), ParquetSource — то, что
есть в снимке (включая производные признаки v7).

Быстрый набор (≤60 с) не ходит в базу и не импортирует ``dbbinance``.
Реальный поход в PG помечен ``@pytest.mark.db`` и пропускается без базы/ключей.
"""
from __future__ import annotations

import builtins
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spotrl.data.sources import (OHLCV_COLUMNS, PG_EXTENDED_COLUMNS,
                                  SUNDAY_STATE_COLUMNS, GateResult,
                                  ParquetSource, PgSource, WindowData,
                                  bitwise_gate, source_manifest)

REF_STATE = Path(
    "~/Data/sunday_tests/state_v0/state_v0_2024-03_2026-07.parquet").expanduser()
_HAS_REF = REF_STATE.exists()


# ---------- синтетический снимок для быстрых тестов ----------

def _make_snapshot_frame(n: int = 200, start="2024-03-10 05:00") -> pd.DataFrame:
    """Снимок: OHLCV, поток ордеров И производные признаки v7 (tz-naive UTC)."""
    idx = pd.date_range(start, periods=n, freq="1min")
    rng = np.random.default_rng(0)
    close = 60000 + np.cumsum(rng.normal(0, 5, n))
    return pd.DataFrame({
        "open": close - 1.0, "high": close + 2.0, "low": close - 2.0,
        "close": close, "volume": rng.uniform(1, 9, n),
        "quote_asset_volume": rng.uniform(1e6, 3e6, n),
        "trades": rng.integers(500, 5000, n).astype(float),
        "taker_buy_base": rng.uniform(5, 30, n),
        "taker_buy_quote": rng.uniform(3e5, 1.5e6, n),
        "q_buy": rng.uniform(0, 1, n), "q_sell": rng.uniform(0, 1, n),
        "regime_code": rng.integers(0, 3, n), "leg_dn": rng.integers(0, 2, n),
    }, index=idx)


def _write_snapshot(tmp_path: Path, n: int = 200) -> Path:
    p = tmp_path / "snap.parquet"
    _make_snapshot_frame(n).to_parquet(p)
    return p


# ---------- ParquetSource ----------

def test_parquet_default_columns_and_features(tmp_path):
    """Снимок по умолчанию отдаёт state-набор v7 и строит StateDataset."""
    src = ParquetSource(_write_snapshot(tmp_path, 200))
    win = src.load_window("2024-03-10 05:00", "2024-03-10 05:49")
    assert len(win) == 50
    assert win.columns == SUNDAY_STATE_COLUMNS
    assert win.ohlcv.shape == (50, 5) and win.data.dtype == np.float64
    assert win.has_features()
    st = win.to_state_dataset()
    assert len(st) == 50


def test_parquet_column_choice_is_parameter(tmp_path):
    """Набор колонок — параметр: снимок может отдать сырьё+поток ордеров."""
    src = ParquetSource(_write_snapshot(tmp_path, 30),
                        columns=PG_EXTENDED_COLUMNS)
    win = src.load_window("2024-03-10 05:00", "2024-03-10 05:29")
    assert win.columns == PG_EXTENDED_COLUMNS
    assert not win.has_features()
    with pytest.raises(ValueError):
        win.to_state_dataset()
    assert win.column("trades").shape == (30,)
    assert win.has("taker_buy_quote")


def test_parquet_window_bounds_inclusive(tmp_path):
    """Границы окна включительны."""
    src = ParquetSource(_write_snapshot(tmp_path, 10))
    win = src.load_window("2024-03-10 05:02", "2024-03-10 05:05")
    assert [str(t) for t in win.index] == [
        "2024-03-10 05:02:00", "2024-03-10 05:03:00",
        "2024-03-10 05:04:00", "2024-03-10 05:05:00"]


def test_manifest_records_columns_and_versions(tmp_path):
    """Манифест: список выбранных колонок, источник, версии, хэш."""
    win = ParquetSource(_write_snapshot(tmp_path, 20)).load_window(
        "2024-03-10 05:00", "2024-03-10 05:19")
    man = source_manifest(win)
    assert len(man["data_sha256"]) == 64
    assert man["columns"] == list(SUNDAY_STATE_COLUMNS)
    assert man["feature_builder_version"] == "state_v0"
    assert man["dbbinance_version"]  # строка (версия либо "unavailable")


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
    assert len(src.load_window("2024-03-10 05:00", "2024-03-10 05:29")) == 30
    PgSource()  # конструктор не импортирует dbbinance


# ---------- PgSource с инъектированным fetcher (без базы) ----------

class _FakeFetcher:
    """Мок ресемплера: отдаёт заданный DataFrame; имитирует last_full_bar."""

    def __init__(self, frame: pd.DataFrame):
        """Args: frame — DataFrame, отдаваемый как ресемпл базы."""
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
    """Синтетический ресемпл базы: extended-колонки, tz-aware UTC индекс."""
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    close = np.arange(n, dtype=float) + 60000.0
    return pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "volume": np.arange(n, dtype=float),
        "quote_asset_volume": np.arange(n, dtype=float) * 1000.0,
        "trades": np.arange(n, dtype=float) + 100,
        "taker_buy_base": np.arange(n, dtype=float) + 0.5,
        "taker_buy_quote": np.arange(n, dtype=float) * 500.0,
    }, index=idx)


def test_pgsource_extended_columns_no_v7():
    """Сырой PG отдаёт extended-набор без производных v7; tz-naive."""
    src = PgSource(fetcher=_FakeFetcher(_pg_frame(10)))
    win = src.load_window("2024-03-10 05:00", "2024-03-10 05:20",
                          last_full_bar=False)
    assert win.columns == PG_EXTENDED_COLUMNS
    assert not win.has_features()
    assert win.ohlcv.shape == (10, 5)
    assert win.column("trades")[0] == 100.0
    assert isinstance(win.index, pd.DatetimeIndex) and win.index.tz is None
    with pytest.raises(ValueError):
        win.to_state_dataset()


def test_pgsource_explicit_columns():
    """Явный набор колонок ограничивает выдачу PgSource."""
    src = PgSource(columns=("open", "close", "trades"),
                   fetcher=_FakeFetcher(_pg_frame(6)))
    win = src.load_window("2024-03-10 05:00", "2024-03-10 05:10",
                          last_full_bar=False)
    assert win.columns == ("open", "close", "trades")


def test_pgsource_live_edge_drops_unclosed_bar():
    """Неполный последний бар живого края не попадает в выборку."""
    src = PgSource(fetcher=_FakeFetcher(_pg_frame(10, "2024-03-10 05:00")))
    now = pd.Timestamp("2024-03-10 05:09:30")
    win = src.load_latest_closed(now, lookback_bars=5, freq="1min")
    assert str(win.index[-1]) == "2024-03-10 05:08:00"
    assert (pd.Timestamp(win.index[-1]) + pd.Timedelta("1min")) <= now


# ---------- гейт эквивалентности: логика на общих данных ----------

def _window(index, columns, data, src="s"):
    """Собрать WindowData из массивов (хелпер тестов)."""
    return WindowData(pd.DatetimeIndex(index), tuple(columns),
                      np.asarray(data, dtype=np.float64), src)


def test_gate_intersection_of_columns_zero_diff():
    """Гейт сравнивает пересечение колонок; общие идентичны → 0 различий."""
    idx = pd.date_range("2024-03-10 05:00", periods=8, freq="1min")
    a = _window(idx, OHLCV_COLUMNS, np.arange(40).reshape(8, 5))
    bdata = np.column_stack([np.arange(40).reshape(8, 5), np.arange(8)])
    b = _window(idx, OHLCV_COLUMNS + ("trades",), bdata)
    res = bitwise_gate(a, b)
    assert isinstance(res, GateResult)
    assert res.passed and res.diff_elements == 0 and res.max_abs_diff == 0.0
    assert res.columns_compared == OHLCV_COLUMNS
    assert res.columns_only_b == ("trades",)


def test_gate_detects_single_bit_diff():
    """Гейт ловит одно-битовое расхождение и указывает колонку."""
    idx = pd.date_range("2024-03-10 05:00", periods=4, freq="1min")
    a = _window(idx, OHLCV_COLUMNS, np.arange(20).reshape(4, 5))
    bd = np.arange(20, dtype=float).reshape(4, 5)
    bd[2, 3] += 1e-9
    b = _window(idx, OHLCV_COLUMNS, bd)
    res = bitwise_gate(a, b)
    assert not res.passed and res.diff_elements == 1
    assert res.first_mismatch[0] == 2 and res.first_mismatch[1] == "close"


def test_gate_boundary_bar_not_a_value_diff():
    """Лишний бар на границе идёт в n_only_a, гейт значений не валит."""
    idx_a = pd.date_range("2024-03-10 05:00", periods=5, freq="1min")
    idx_b = idx_a[:4]
    a = _window(idx_a, OHLCV_COLUMNS, np.arange(25).reshape(5, 5))
    b = _window(idx_b, OHLCV_COLUMNS, np.arange(20).reshape(4, 5))
    res = bitwise_gate(a, b)
    assert res.passed and res.n_only_a == 1 and res.n_only_b == 0
    assert not res.index_equal


# ---------- реальный slow-тест на снимке (без базы) ----------

@pytest.mark.slow
@pytest.mark.skipif(not _HAS_REF, reason="нет parquet-снимка sunday")
def test_real_parquet_snapshot_loads():
    """ParquetSource на РЕАЛЬНОМ снимке: признаки, dtype, tz-naive, монотонность."""
    win = ParquetSource(REF_STATE).load_window(
        "2024-03-10 05:00", "2024-03-11 05:00")
    assert len(win) == 1441
    assert win.has_features() and win.data.dtype == np.float64
    idx = pd.DatetimeIndex(win.index)
    assert idx.tz is None and idx.is_monotonic_increasing
    assert len(win.to_state_dataset()) == 1441


# ---------- ГЛАВНЫЙ ГЕЙТ: реальный поход в PG ----------

@pytest.mark.db
@pytest.mark.skipif(not _HAS_REF, reason="нет parquet-снимка sunday")
def test_pg_vs_parquet_bitwise_gate():
    """ГЛАВНЫЙ ГЕЙТ: снимок и база ПОБИТОВО равны на пересечении колонок.

    Пересечение — extended kline (OHLCV + поток ордеров), т.к. снимок эти
    сырые колонки тоже содержит. Ключи PG грузятся PgSource из sunday/*.env.
    Без базы/ключей — честный skip.
    """
    try:
        pg = PgSource()  # extended по умолчанию
        pg._get_fetcher()
        parquet = ParquetSource(REF_STATE, columns=PG_EXTENDED_COLUMNS)
        start, end = "2024-03-10 05:00", "2024-03-11 05:00"
        wp = parquet.load_window(start, end)
        wg = pg.load_window(start, end)
    except Exception as exc:  # нет базы/ключей/подключения
        pytest.skip(f"PG недоступна: {exc!r}")
    res = bitwise_gate(wp, wg)
    print(f"GATE common_bars={res.n_bars} cols={list(res.columns_compared)} "
          f"diff={res.diff_elements} max_abs={res.max_abs_diff} "
          f"per_col={res.per_column} only_parquet={res.n_only_a} "
          f"only_pg={res.n_only_b} first={res.first_mismatch}")
    for c in ("quote_asset_volume", "trades", "taker_buy_base",
              "taker_buy_quote"):
        assert c in res.columns_compared
        assert not np.isnan(wg.column(c)).any()
    assert res.n_bars > 1000, "слишком мало общих баров"
    assert res.passed, f"не побитово равны: {res.per_column}"
    assert res.n_only_b == 0 and res.n_only_a <= 1
