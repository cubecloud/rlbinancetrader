"""Парность источников для рыночного строителя v2.

Требование части 3: строитель на колонках из ParquetSource и PgSource (extended
kline) даёт ОДИНАКОВЫЙ рыночный вектор — хотя бы по OHLCV-производным и торговым
числам (первые 9 признаков v2: 6 цен + 3 торговых числа). Непрерывные
составляющие v7 и булевы сигналы в сырой базе отсутствуют — они только в снимке,
поэтому сверяется рыночное ПОД-ядро, не полный v2-market-блок.

Быстрый набор не ходит в базу: PgSource получает инъектированный fetcher с теми
же extended-колонками, что и снимок. Реальный поход в PG — отдельный db-тест.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spotrl.data.sources import (PG_EXTENDED_COLUMNS, ParquetSource, PgSource)
from spotrl.data.state_dataset import from_frame
from spotrl.features.builder import precompute_market_features_v2
from spotrl.spec.observation import ObservationSpec

# Индексы рыночного ПОД-ядра, не зависящего от v7 (6 цен + 3 торговых числа).
_MARKET_CORE = tuple(range(9))

REF_STATE = Path(
    "~/Data/sunday_tests/state_v0/state_v0_2024-03_2026-07.parquet").expanduser()
_HAS_REF = REF_STATE.exists()


def _extended_frame(n: int = 400, start="2024-03-10 05:00") -> pd.DataFrame:
    """Ряд extended-kline (OHLCV + поток ордеров), tz-naive UTC."""
    idx = pd.date_range(start, periods=n, freq="1min")
    rng = np.random.default_rng(11)
    close = 60000 + np.cumsum(rng.normal(0, 5, n))
    base_vol = np.exp(rng.normal(1.0, 0.7, n))
    return pd.DataFrame({
        "open": close - 1.0, "high": close + 2.0, "low": close - 2.0,
        "close": close, "volume": base_vol,
        "quote_asset_volume": base_vol * close * rng.uniform(0.98, 1.02, n),
        "trades": np.maximum(1, rng.poisson(200, n)).astype(float),
        "taker_buy_base": base_vol * rng.uniform(0.2, 0.8, n),
        "taker_buy_quote": rng.uniform(3e5, 1.5e6, n),
    }, index=idx)


class _FakeFetcher:
    """Мок ресемплера базы: отдаёт заданный extended-DataFrame."""

    def __init__(self, frame: pd.DataFrame):
        """Args: frame — DataFrame, отдаваемый как ресемпл базы."""
        self._frame = frame

    def pg_resample_to_timeframe(self, **kwargs):
        """Мок ресемпла: возвращает исходный кадр как есть."""
        return self._frame


def _core_from_window(win) -> np.ndarray:
    """Рыночное ПОД-ядро v2 (цены+торговые числа) из WindowData источника.

    Строит StateDataset с нейтральными v7-колонками (в сырой базе их нет; на
    ядро цен/объёмов они не влияют) и считает строитель v2.
    """
    frame = pd.DataFrame(
        {c: win.column(c) for c in win.columns if c != "taker_buy_quote"},
        index=win.index)
    frame["q_buy"] = 0.0
    frame["q_sell"] = 0.0
    frame["regime_code"] = 0
    frame["leg_dn"] = False
    market = precompute_market_features_v2(from_frame(frame))
    return market[:, _MARKET_CORE]


def test_market_core_index_names():
    """Первые 9 признаков v2 — именно цены и торговые числа (не сигналы v7)."""
    market = ObservationSpec.v2().market
    assert market[:9] == (
        "m_ret_close", "m_ret_high", "m_ret_low", "m_vwap_ret",
        "m_hi_close", "m_close_lo",
        "m_aggr_buy_frac", "m_avg_trade_size", "m_rel_volume")


def test_builder_v2_source_parity_parquet_vs_pg(tmp_path):
    """Рыночное ядро v2 из снимка и из базы (те же extended-колонки) == побитово."""
    frame = _extended_frame(400)
    snap = tmp_path / "snap.parquet"
    frame.to_parquet(snap)
    win_pq = ParquetSource(snap, columns=PG_EXTENDED_COLUMNS).load_window(
        frame.index[0], frame.index[-1])
    win_pg = PgSource(fetcher=_FakeFetcher(frame)).load_window(
        frame.index[0], frame.index[-1], last_full_bar=False)
    core_pq = _core_from_window(win_pq)
    core_pg = _core_from_window(win_pg)
    diff = int((core_pq != core_pg).sum())
    assert diff == 0, f"парность источников: {diff} расхождений рыночного ядра"


@pytest.mark.db
@pytest.mark.skipif(not _HAS_REF, reason="нет parquet-снимка sunday")
def test_builder_v2_source_parity_real_pg():
    """Реальный поход в PG: рыночное ядро v2 снимок == база на общем окне."""
    try:
        pg = PgSource()
        pg._get_fetcher()
        start, end = "2024-03-10 05:00", "2024-03-11 05:00"
        win_pg = pg.load_window(start, end)
        win_pq = ParquetSource(REF_STATE, columns=PG_EXTENDED_COLUMNS).load_window(
            start, end)
    except Exception as exc:
        pytest.skip(f"PG недоступна: {exc!r}")
    common = win_pg.index.intersection(win_pq.index)
    core_pg = _core_from_window(win_pg)
    core_pq = _core_from_window(win_pq)
    pg_pos = pd.DatetimeIndex(win_pg.index).get_indexer(common)
    pq_pos = pd.DatetimeIndex(win_pq.index).get_indexer(common)
    diff = int((core_pg[pg_pos] != core_pq[pq_pos]).sum())
    print(f"SOURCE-PARITY-V2 common_bars={len(common)} diff={diff}")
    assert diff == 0, f"парность источников (real PG): {diff} расхождений"
