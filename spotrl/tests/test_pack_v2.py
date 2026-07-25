"""Упаковка полного вектора наблюдения v2 (write_observation_v2/build_observation).

Проверяет расположение и нормировку 14 полей агента и расширенного блока мира:
one-hot pos_tag, tanh(bars/median_hold), clip чистого PnL ±0.5, слот просадки =
то, что подала среда (в режиме копирования — МИРОВАЯ CB-просадка). Чистота: два
вызова побитово равны, состояние агента не мутируется.
"""
from __future__ import annotations

import numpy as np

from spotrl.features.builder import (AgentState, WorldState, build_observation,
                                     precompute_market_features_v2)
from spotrl.data.state_dataset import from_frame
from spotrl.data.synthetic import make_synthetic_frame
from spotrl.spec.observation import ObservationSpec, SPEC_CONSTANTS_V2


def _market_row():
    """Одна строка рыночного вектора v2 (18) на синтетике."""
    ds = from_frame(make_synthetic_frame(n_bars=100, seed=1))
    return precompute_market_features_v2(ds)[50]


def test_build_observation_v2_size_and_purity():
    """Размер = spec.size; два вызова побитово равны; агент не мутируется."""
    spec = ObservationSpec.v2()
    row = _market_row()
    agent = AgentState(in_position=True, unreal_pnl=0.02, peak_unreal=0.05,
                       bars_in_trade=1221, dist_to_sl=0.03, dist_to_tp=0.07,
                       entered_on_up=True, pos_tag="dip", price_drawdown=-0.01)
    world = WorldState(cb_active=True, cooldown_active=True, cooldown_remain=0.5,
                       equity_drawdown=-0.12)
    a = build_observation(row, agent, world, spec)
    b = build_observation(row, agent, world, spec)
    assert a.shape == (spec.size,) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert agent.peak_unreal == 0.05 and agent.pos_tag == "dip"


def test_v2_pos_tag_one_hot_and_transforms():
    """one-hot pos_tag, tanh(days), clip PnL ±0.5, слот мировой просадки."""
    spec = ObservationSpec.v2()
    row = _market_row()
    median_hold = float(SPEC_CONSTANTS_V2["median_hold_bars"])
    agent = AgentState(in_position=True, unreal_pnl=0.9, peak_unreal=-0.9,
                       bars_in_trade=median_hold, dist_to_sl=0.03,
                       dist_to_tp=0.07, entered_on_up=False,
                       pos_tag="transition", price_drawdown=-0.02)
    world = WorldState(cb_active=False, cb_cleared_today=True,
                       cooldown_remain=0.25, equity_drawdown=-0.3)
    obs = build_observation(row, agent, world, spec)
    idx = spec.index_of
    # clip чистого PnL ±0.5
    assert obs[idx("a_unreal_pnl")] == 0.5
    assert obs[idx("a_peak_unreal")] == -0.5
    # one-hot pos_tag = transition
    assert obs[idx("a_pos_tag_none")] == 0.0
    assert obs[idx("a_pos_tag_dip")] == 0.0
    assert obs[idx("a_pos_tag_transition")] == 1.0
    # tanh(median_hold/median_hold) = tanh(1)
    assert abs(obs[idx("a_days_in_trade")] - np.tanh(1.0)) < 1e-6
    assert obs[idx("a_cb_cleared_today")] == 1.0
    assert abs(obs[idx("a_cooldown_remain")] - 0.25) < 1e-6
    # слот просадки эквити = то, что подала среда (мировая CB-просадка)
    assert abs(obs[idx("w_equity_drawdown")] - (-0.3)) < 1e-6


def test_v2_pos_tag_none_out_of_position():
    """Вне позиции pos_tag='none' → one-hot none=1, days_in_trade=tanh(0)=0."""
    spec = ObservationSpec.v2()
    obs = build_observation(_market_row(), AgentState(), WorldState(), spec)
    idx = spec.index_of
    assert obs[idx("a_pos_tag_none")] == 1.0
    assert obs[idx("a_pos_tag_dip")] == 0.0
    assert obs[idx("a_days_in_trade")] == 0.0
    assert obs[idx("a_in_position")] == 0.0
