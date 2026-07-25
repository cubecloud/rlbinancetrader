"""Гейты рыночного строителя наблюдения v2: состав, каузальность, утечка.

Главная проверка части 3 — УТЕЧКА: каждый скользящий признак на префиксе
[0:t] обязан побитово совпасть с тем же признаком, посчитанным на полном ряде
в точке t. Раньше «мерить было нечего» (в v1 не было ни одного скользящего
признака); с вводом относительного объёма, среднего размера сделки и
Δlog-доходностей проверка стала содержательной.
"""
from __future__ import annotations

import numpy as np
import pytest

from spotrl.data.state_dataset import from_frame
from spotrl.data.synthetic import make_synthetic_frame
from spotrl.features.builder import precompute_market_features_v2
from spotrl.spec.observation import ObservationSpec


def test_market_v2_shape_matches_spec():
    """Строитель v2 отдаёт ровно столько колонок, сколько market в spec v2."""
    spec = ObservationSpec.v2()
    ds = from_frame(make_synthetic_frame(n_bars=500, seed=1))
    market = precompute_market_features_v2(ds)
    assert market.shape == (500, len(spec.market))
    assert market.dtype == np.float32
    assert np.isfinite(market).all()


def test_market_v2_aggr_buy_frac_range():
    """Доля агрессивных покупок в [0, 1]; при отсутствии объёма → нейтраль 0.5."""
    spec = ObservationSpec.v2()
    ds = from_frame(make_synthetic_frame(n_bars=300, seed=2))
    market = precompute_market_features_v2(ds)
    col = market[:, spec.market.index("m_aggr_buy_frac")]
    assert (col >= 0.0).all() and (col <= 1.0).all()


def test_market_v2_rel_volume_clipped():
    """Относительный объём зажат фиксированным порогом ±3 (не p1..p99)."""
    spec = ObservationSpec.v2()
    ds = from_frame(make_synthetic_frame(n_bars=3000, seed=3))
    market = precompute_market_features_v2(ds)
    col = market[:, spec.market.index("m_rel_volume")]
    assert col.min() >= -3.0 - 1e-6 and col.max() <= 3.0 + 1e-6


def test_market_v2_missing_trade_columns_degrade_to_neutral():
    """Без торговых колонок строитель не падает и не выдумывает объём.

    aggr_buy_frac → 0.5, avg_trade_size → 0, rel_volume → 0 (нейтрали дока).
    """
    spec = ObservationSpec.v2()
    frame = make_synthetic_frame(n_bars=200, seed=4).drop(
        columns=["quote_asset_volume", "trades", "taker_buy_base"])
    market = precompute_market_features_v2(from_frame(frame))
    assert np.isfinite(market).all()
    assert np.allclose(market[:, spec.market.index("m_aggr_buy_frac")], 0.5)
    assert np.allclose(market[:, spec.market.index("m_avg_trade_size")], 0.0)
    assert np.allclose(market[:, spec.market.index("m_rel_volume")], 0.0)


def test_market_v2_causal_prefix_equals_full():
    """Каузальность блоком: признак на [100:cut] не меняется от данных после cut."""
    frame = make_synthetic_frame(n_bars=5000, seed=0)
    full = precompute_market_features_v2(from_frame(frame))
    cut = 2000
    part = precompute_market_features_v2(from_frame(frame.iloc[:cut]))
    assert np.array_equal(full[100:cut], part[:cut][100:cut])


def test_market_v2_leak_gate_rolling_features_prefix_equals_full():
    """ГЛАВНЫЙ гейт утечки: скользящий признак на префиксе [0:t] == полный ряд в t.

    Для набора точек среза t строим StateDataset ТОЛЬКО из прошлого [0:t] и
    сверяем ПОСЛЕДНЮЮ строку рыночного вектора с полным рядом в точке t —
    побитово. Ноль расхождений = ни один скользящий признак (относительный
    объём, средний размер сделки, доходности, vwap) не заглядывает в текущий/
    будущий бар. Сэмплируем ~200 точек (бюджет ≤60 с, не O(n^2)).
    """
    spec = ObservationSpec.v2()
    frame = make_synthetic_frame(n_bars=4000, seed=7)
    full = precompute_market_features_v2(from_frame(frame))
    rng = np.random.default_rng(0)
    cuts = np.unique(rng.integers(1, 4000, size=250))
    diffs = 0
    first = None
    for t in cuts:
        prefix = precompute_market_features_v2(from_frame(frame.iloc[:t + 1]))
        if not np.array_equal(prefix[-1], full[t]):
            diffs += 1
            if first is None:
                bad = np.flatnonzero(prefix[-1] != full[t])
                first = (int(t), [spec.market[i] for i in bad])
    assert diffs == 0, f"утечка: {diffs} расхождений, первое {first}"


def test_market_v2_leak_gate_covers_rolling_columns():
    """Санити: точки среза реально попадают в бары, где скользящие окна не пусты.

    Иначе гейт утечки прошёл бы вхолостую на одних нейтралях. Проверяем, что у
    относительного объёма и среднего размера сделки есть НЕнулевая вариация.
    """
    spec = ObservationSpec.v2()
    market = precompute_market_features_v2(
        from_frame(make_synthetic_frame(n_bars=4000, seed=7)))
    for name in ("m_rel_volume", "m_avg_trade_size"):
        col = market[:, spec.market.index(name)]
        assert np.std(col) > 1e-4, f"{name} без вариации — гейт утечки пуст"
