"""Тесты кэша окон :class:`WindowCache`: попадание/промах, инвалидация по
каждому компоненту ключа, атомарность записи, отказ кэшировать живой край и
побитовый гейт ``cold == warm`` с подсчётом обращений к базе.

Быстрый набор (≤60 с) не ходит в базу: PgSource получает фейковый fetcher с
управляемой задержкой, эмулирующей round-trip в PostgreSQL. Число ускорения
отражает СТОИМОСТЬ ИЗБЕГАЕМОГО запроса, а не parquet-IO (стенд — в отчёте).
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spotrl.data import cache as cache_mod
from spotrl.data.cache import WindowCache
from spotrl.data.sources import PG_EXTENDED_COLUMNS, ParquetSource, PgSource

# now в 2026 — синтетика 2024 гарантированно «в прошлом» и кэшируется.
NOW_2026 = pd.Timestamp("2026-01-01 00:00")


# ---------- фейковый fetcher базы с задержкой и счётчиком ----------

class _SlowFetcher:
    """Мок ресемплера PG: отдаёт заданный DataFrame, считает вызовы, спит."""

    def __init__(self, frame: pd.DataFrame, delay_s: float = 0.0):
        """Args: frame — ресемпл базы; delay_s — эмуляция round-trip в PG."""
        self._frame = frame
        self._delay = delay_s
        self.calls = 0

    def pg_resample_to_timeframe(self, table_name, start, end, to_timeframe,
                                 origin, open_time_index, last_full_bar,
                                 **kwargs):
        """Мок ресемпла базы: имитирует last_full_bar и задержку сети."""
        self.calls += 1
        if self._delay:
            time.sleep(self._delay)
        df = self._frame
        if last_full_bar and len(df):
            step = pd.Timedelta(to_timeframe.replace("m", "min"))
            idx = pd.DatetimeIndex(df.index)
            end_ts = pd.Timestamp(end)
            if end_ts.tzinfo is None and idx.tz is not None:
                end_ts = end_ts.tz_localize(idx.tz)
            df = df[(idx + step) <= end_ts]
        return df


def _pg_frame(n=60, start="2024-03-10 05:00"):
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


def _spy_pg(frame=None, delay_s=0.0, **pg_kwargs):
    """PgSource с фейковым fetcher и счётчиками load_window/_get_fetcher."""
    fetcher = _SlowFetcher(frame if frame is not None else _pg_frame(),
                           delay_s=delay_s)
    pg = PgSource(fetcher=fetcher, **pg_kwargs)
    counters = {"load": 0, "get_fetcher": 0}

    orig_load = pg.load_window
    orig_gf = pg._get_fetcher

    def load_spy(*a, **k):
        """Счётчик обёртки над load_window."""
        counters["load"] += 1
        return orig_load(*a, **k)

    def gf_spy():
        """Счётчик обёртки над _get_fetcher."""
        counters["get_fetcher"] += 1
        return orig_gf()

    pg.load_window = load_spy
    pg._get_fetcher = gf_spy
    return pg, fetcher, counters


def _windows_identical(a, b) -> bool:
    """Побитовая идентичность двух WindowData (0 различающихся элементов)."""
    return (pd.DatetimeIndex(a.index).equals(pd.DatetimeIndex(b.index))
            and a.columns == b.columns
            and a.data_hash() == b.data_hash()
            and a.source == b.source
            and a.meta == b.meta)


# ---------- гейт: побитово cold == warm, без обращения к базе ----------

def test_gate_cold_equals_warm_no_db_access(tmp_path, capsys):
    """ГЕЙТ: cold и warm побитово равны; тёплое чтение не ходит в базу."""
    pg, fetcher, counters = _spy_pg()
    wc = WindowCache(pg, cache_dir=tmp_path, now=NOW_2026)

    cold = wc.load_window("2024-03-10 05:00", "2024-03-10 05:59",
                          last_full_bar=False)
    assert counters["load"] == 1 and counters["get_fetcher"] == 1
    fetch_after_cold = fetcher.calls

    warm = wc.load_window("2024-03-10 05:00", "2024-03-10 05:59",
                          last_full_bar=False)
    # Тёплый путь не трогает inner и его fetcher.
    assert counters["load"] == 1, "тёплое чтение вызвало inner.load_window"
    assert counters["get_fetcher"] == 1, "тёплое чтение вызвало _get_fetcher"
    assert fetcher.calls == fetch_after_cold, "тёплое чтение сходило в базу"

    assert _windows_identical(cold, warm)
    # dtype/раскладка удержаны parquet round-trip.
    assert warm.data.dtype == np.float64 and warm.data.flags["C_CONTIGUOUS"]
    assert pd.DatetimeIndex(warm.index).dtype == np.dtype("datetime64[ns]")

    diff = int((cold.data != warm.data).sum())
    print(f"CACHE_GATE diff_elements={diff} "
          f"warm_get_fetcher={counters['get_fetcher'] - 1} "
          f"warm_fetch_calls={fetcher.calls - fetch_after_cold}")
    assert diff == 0


# ---------- инвалидация по каждому компоненту ключа ----------

def test_invalidation_window_bounds(tmp_path):
    """Другие границы окна → промах."""
    pg, _, counters = _spy_pg()
    wc = WindowCache(pg, cache_dir=tmp_path, now=NOW_2026)
    wc.load_window("2024-03-10 05:00", "2024-03-10 05:30", last_full_bar=False)
    wc.load_window("2024-03-10 05:00", "2024-03-10 05:31", last_full_bar=False)
    assert counters["load"] == 2


def test_invalidation_last_full_bar(tmp_path):
    """Другой last_full_bar меняет набор баров → промах."""
    pg, _, counters = _spy_pg()
    wc = WindowCache(pg, cache_dir=tmp_path, now=NOW_2026)
    wc.load_window("2024-03-10 05:00", "2024-03-10 05:59", last_full_bar=False)
    wc.load_window("2024-03-10 05:00", "2024-03-10 05:59", last_full_bar=True)
    assert counters["load"] == 2


def test_invalidation_columns(tmp_path):
    """Другой набор колонок → промах (разные источники, общая директория)."""
    frame = _pg_frame()
    pg_a, _, ca = _spy_pg(frame=frame)
    pg_b, _, cb = _spy_pg(frame=frame, columns=("open", "close", "trades"))
    wc_a = WindowCache(pg_a, cache_dir=tmp_path, now=NOW_2026)
    wc_b = WindowCache(pg_b, cache_dir=tmp_path, now=NOW_2026)
    wc_a.load_window("2024-03-10 05:00", "2024-03-10 05:30",
                     last_full_bar=False)
    wc_b.load_window("2024-03-10 05:00", "2024-03-10 05:30",
                     last_full_bar=False)
    assert ca["load"] == 1 and cb["load"] == 1  # оба холодные, ключи разные


def test_invalidation_dbbinance_version(tmp_path, monkeypatch):
    """Другая версия dbbinance → промах."""
    pg, _, counters = _spy_pg()
    wc = WindowCache(pg, cache_dir=tmp_path, now=NOW_2026)
    monkeypatch.setattr(cache_mod, "dbbinance_version", lambda: "1.0.0")
    wc.load_window("2024-03-10 05:00", "2024-03-10 05:30", last_full_bar=False)
    monkeypatch.setattr(cache_mod, "dbbinance_version", lambda: "9.9.9")
    wc.load_window("2024-03-10 05:00", "2024-03-10 05:30", last_full_bar=False)
    assert counters["load"] == 2


def test_invalidation_feature_builder_version(tmp_path, monkeypatch):
    """Другая версия билдера признаков → промах."""
    pg, _, counters = _spy_pg()
    wc = WindowCache(pg, cache_dir=tmp_path, now=NOW_2026)
    monkeypatch.setattr(cache_mod, "FEATURE_BUILDER_VERSION", "state_v0")
    wc.load_window("2024-03-10 05:00", "2024-03-10 05:30", last_full_bar=False)
    monkeypatch.setattr(cache_mod, "FEATURE_BUILDER_VERSION", "state_v1")
    wc.load_window("2024-03-10 05:00", "2024-03-10 05:30", last_full_bar=False)
    assert counters["load"] == 2


def test_invalidation_cache_format_version(tmp_path, monkeypatch):
    """Другая версия формата кэша → промах (глобальная инвалидация)."""
    pg, _, counters = _spy_pg()
    wc = WindowCache(pg, cache_dir=tmp_path, now=NOW_2026)
    monkeypatch.setattr(cache_mod, "CACHE_FORMAT_VERSION", "wc1")
    wc.load_window("2024-03-10 05:00", "2024-03-10 05:30", last_full_bar=False)
    monkeypatch.setattr(cache_mod, "CACHE_FORMAT_VERSION", "wc2")
    wc.load_window("2024-03-10 05:00", "2024-03-10 05:30", last_full_bar=False)
    assert counters["load"] == 2


def _write_snapshot(tmp_path: Path, n=60, bump=0.0) -> Path:
    idx = pd.date_range("2024-03-10 05:00", periods=n, freq="1min")
    close = np.arange(n, dtype=float) + 60000.0 + bump
    p = tmp_path / "snap.parquet"
    pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "volume": np.arange(n, dtype=float),
        "quote_asset_volume": np.arange(n, dtype=float) * 1000.0,
        "trades": np.arange(n, dtype=float) + 100,
        "taker_buy_base": np.arange(n, dtype=float) + 0.5,
        "taker_buy_quote": np.arange(n, dtype=float) * 500.0,
    }, index=idx).to_parquet(p)
    return p


def test_invalidation_parquet_rebuild(tmp_path):
    """Пересборка снимка по тому же пути (mtime+size) → промах."""
    snap = _write_snapshot(tmp_path)
    cache_dir = tmp_path / "cache"

    def load_counted(path):
        """Один прогон через кэш; вернуть число обращений к источнику."""
        src = ParquetSource(path, columns=PG_EXTENDED_COLUMNS)
        n = {"c": 0}
        orig = src.load_window

        def spy(*a, **k):
            """Счётчик обёртки над load_window снимка."""
            n["c"] += 1
            return orig(*a, **k)

        src.load_window = spy
        wc = WindowCache(src, cache_dir=cache_dir, now=NOW_2026)
        wc.load_window("2024-03-10 05:00", "2024-03-10 05:30")
        return n["c"]

    assert load_counted(snap) == 1              # холодная
    assert load_counted(snap) == 0              # тёплая (тот же файл)
    time.sleep(0.01)
    _write_snapshot(tmp_path, bump=1.0)         # пересобрали снимок
    assert load_counted(snap) == 1              # промах: файл изменился


# ---------- живой край ----------

def test_live_edge_not_cached(tmp_path):
    """Окно у живого края не кэшируется: повторный вызов снова идёт в базу."""
    now = pd.Timestamp("2024-03-10 06:00")  # край совпадает с концом данных
    frame = _pg_frame(n=60, start="2024-03-10 05:00")  # до 05:59
    pg, _, counters = _spy_pg(frame=frame)
    wc = WindowCache(pg, cache_dir=tmp_path, now=now, live_margin_bars=2)
    wc.load_window("2024-03-10 05:00", "2024-03-10 05:59", last_full_bar=False)
    wc.load_window("2024-03-10 05:00", "2024-03-10 05:59", last_full_bar=False)
    assert counters["load"] == 2, "живой край не должен кэшироваться"
    # Ни один parquet не записан.
    assert not list(Path(tmp_path).glob("*.parquet"))


def test_load_latest_closed_bypasses_cache(tmp_path):
    """Онлайн-путь идёт мимо кэша и ничего не пишет."""
    pg, _, counters = _spy_pg(frame=_pg_frame(n=20, start="2024-03-10 05:00"))
    wc = WindowCache(pg, cache_dir=tmp_path, now=NOW_2026)
    win = wc.load_latest_closed(pd.Timestamp("2024-03-10 05:19:30"),
                                lookback_bars=5, freq="1min")
    assert len(win) == 5
    assert not list(Path(tmp_path).glob("*.parquet"))


# ---------- атомарность ----------

def test_atomic_write_failure_leaves_no_final_file(tmp_path, monkeypatch):
    """Сбой os.replace после temp: финального файла нет, читатель — промах."""
    pg, _, counters = _spy_pg()
    wc = WindowCache(pg, cache_dir=tmp_path, now=NOW_2026)

    real_replace = cache_mod.os.replace

    def boom(src, dst):
        """Смоделировать сбой атомарной замены."""
        raise OSError("смоделированный сбой записи")

    monkeypatch.setattr(cache_mod.os, "replace", boom)
    with pytest.raises(OSError):
        wc.load_window("2024-03-10 05:00", "2024-03-10 05:30",
                       last_full_bar=False)
    monkeypatch.setattr(cache_mod.os, "replace", real_replace)

    # Финальных артефактов нет; читатель видит промах, а не битый .tmp.
    assert not list(Path(tmp_path).glob("*.parquet"))
    assert not list(Path(tmp_path).glob("*.manifest.json"))
    counters["load"] = 0
    wc.load_window("2024-03-10 05:00", "2024-03-10 05:30", last_full_bar=False)
    assert counters["load"] == 1  # промах — пошли в источник заново


def test_reader_ignores_stray_tmp(tmp_path):
    """Случайный *.tmp не подхватывается как валидный кэш."""
    pg, _, counters = _spy_pg()
    wc = WindowCache(pg, cache_dir=tmp_path, now=NOW_2026)
    (Path(tmp_path) / "garbage.parquet.tmp").write_bytes(b"not-a-parquet")
    wc.load_window("2024-03-10 05:00", "2024-03-10 05:30", last_full_bar=False)
    assert counters["load"] == 1


# ---------- ускорение (число для отчёта) ----------

def test_warm_read_speedup(tmp_path, capsys):
    """Тёплое чтение быстрее холодного (эмуляция round-trip в PG задержкой)."""
    pg, _, _ = _spy_pg(delay_s=0.05)
    wc = WindowCache(pg, cache_dir=tmp_path, now=NOW_2026)
    t0 = time.perf_counter()
    wc.load_window("2024-03-10 05:00", "2024-03-10 05:59", last_full_bar=False)
    t_cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    wc.load_window("2024-03-10 05:00", "2024-03-10 05:59", last_full_bar=False)
    t_warm = time.perf_counter() - t0
    speedup = t_cold / max(t_warm, 1e-9)
    print(f"CACHE_SPEEDUP t_cold={t_cold*1e3:.2f}ms "
          f"t_warm={t_warm*1e3:.2f}ms speedup={speedup:.1f}x "
          f"(задержка 50ms эмулирует запрос в PG)")
    assert speedup > 1.0


# ---------- продакшн-ветка now=None и каноничность ключа ----------

def test_default_now_caches_historical(tmp_path):
    """now=None (продакшн, utcnow): историческое окно 2024 кэшируется."""
    pg, _, counters = _spy_pg()
    wc = WindowCache(pg, cache_dir=tmp_path)  # now по умолчанию = utcnow
    wc.load_window("2024-03-10 05:00", "2024-03-10 05:30", last_full_bar=False)
    wc.load_window("2024-03-10 05:00", "2024-03-10 05:30", last_full_bar=False)
    assert counters["load"] == 1  # тёплый хит на дефолтном now


def test_key_canonical_str_equals_timestamp(tmp_path):
    """Строка и Timestamp одного момента дают ОДИН ключ → тёплый хит."""
    pg, _, counters = _spy_pg()
    wc = WindowCache(pg, cache_dir=tmp_path, now=NOW_2026)
    wc.load_window("2024-03-10 05:00", "2024-03-10 05:30", last_full_bar=False)
    wc.load_window(pd.Timestamp("2024-03-10 05:00"),
                   pd.Timestamp("2024-03-10 05:30"), last_full_bar=False)
    assert counters["load"] == 1


# ---------- параллельная запись одного ключа (потокобезопасность) ----------

def test_concurrent_cold_writes_same_key(tmp_path):
    """8 воркеров пишут ОДИН ключ: кэш не портится, warm без источника."""
    frame = _pg_frame(n=300, start="2024-03-10 05:00")
    results = []

    def worker(_):
        """Один холодный проход через собственный WindowCache на общий кэш."""
        pg, _f, _c = _spy_pg(frame=frame)
        wc = WindowCache(pg, cache_dir=tmp_path, now=NOW_2026)
        return wc.load_window("2024-03-10 05:00", "2024-03-10 09:59",
                              last_full_bar=False)

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(worker, range(8)))

    ref = results[0].data_hash()
    assert all(w.data_hash() == ref for w in results)
    # Никаких недописанных temp не осталось.
    assert not list(Path(tmp_path).glob("*.tmp"))

    # Свежий warm-read через кэш: источник не зовётся, данные те же.
    pg2, _f2, c2 = _spy_pg(frame=frame)
    wc2 = WindowCache(pg2, cache_dir=tmp_path, now=NOW_2026)
    warm = wc2.load_window("2024-03-10 05:00", "2024-03-10 09:59",
                           last_full_bar=False)
    assert c2["load"] == 0 and warm.data_hash() == ref
