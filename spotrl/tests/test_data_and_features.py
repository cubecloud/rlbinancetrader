"""Проверки слоя данных и сборки наблюдения."""
from __future__ import annotations

import numpy as np
import pytest

from spotrl.config import WorldConfig
from spotrl.data.state_dataset import from_frame, load_state
from spotrl.data.synthetic import make_synthetic_frame
from spotrl.envs.tradebook import TradeBook
from spotrl.envs.worldrules import breaker_armed, check_forced_exit
from spotrl.features.builder import (AgentState, WorldState, build_observation,
                                     precompute_market_features)
from spotrl.spec.observation import ObservationSpec


def test_missing_column_is_an_error():
    """Отсутствие обязательной колонки state — ошибка на границе."""
    frame = make_synthetic_frame(n_bars=10).drop(columns=["q_buy"])
    with pytest.raises(KeyError):
        from_frame(frame)


def test_non_monotonic_index_is_an_error():
    """Неупорядоченный по времени индекс не принимается."""
    frame = make_synthetic_frame(n_bars=10).iloc[::-1]
    with pytest.raises(ValueError):
        from_frame(frame)


def test_load_state_missing_file():
    """Отсутствующий файл state — ошибка чтения, а не пустой датасет."""
    with pytest.raises((FileNotFoundError, OSError)):
        load_state("/nonexistent/state.parquet")


def test_state_signals_default_to_false():
    """Без колонок сигналов v7 поля StateDataset = массивы False длины n."""
    ds = from_frame(make_synthetic_frame(n_bars=20))
    for name in ("entry_signal", "trans_entry_signal", "exit_sig"):
        arr = getattr(ds, name)
        assert arr.dtype == bool and len(arr) == 20 and not arr.any()


def test_state_signals_loaded_from_columns():
    """Булевы сигналы v7 читаются из кадра, если колонки присутствуют."""
    frame = make_synthetic_frame(n_bars=12)
    frame["entry_signal"] = ([True, False] * 6)
    frame["trans_entry_signal"] = False
    frame["exit_sig"] = ([False, True] * 6)
    ds = from_frame(frame)
    assert ds.entry_signal.sum() == 6 and ds.exit_sig.sum() == 6
    assert not ds.trans_entry_signal.any()


def test_market_features_are_causal(state):
    """Признак на баре t не меняется от данных после t."""
    full = precompute_market_features(state)
    cut = 1_000
    part = precompute_market_features(from_frame(
        make_synthetic_frame(n_bars=5_000, seed=0).iloc[:cut]))
    assert np.allclose(full[100:cut], part[100:cut], atol=1e-6)


def test_build_observation_is_pure_and_sized():
    """Размер вектора равен spec.size, аргументы не изменяются."""
    spec = ObservationSpec()
    row = np.arange(len(spec.market), dtype=np.float32)
    agent = AgentState(in_position=True, unreal_pnl=0.01, peak_unreal=0.02,
                       bars_in_trade=5, dist_to_sl=0.06, dist_to_tp=0.09)
    obs_a = build_observation(row, agent, WorldState(), spec)
    obs_b = build_observation(row, agent, WorldState(), spec)
    assert obs_a.shape == (spec.size,)
    assert np.array_equal(obs_a, obs_b)
    assert agent.peak_unreal == 0.02


def test_build_observation_rejects_wrong_row():
    """Строка рыночных признаков неверной длины — ошибка."""
    with pytest.raises(ValueError):
        build_observation(np.zeros(3), AgentState(), WorldState(), ObservationSpec())


def test_forced_exit_rules():
    """Стоп срабатывает по Low, тейк — по Close, конец данных закрывает сделку."""
    world = WorldConfig(stop_loss_frac=0.05)
    book = TradeBook(fee_side=world.fee_side)
    book.open(bar=10, price=100.0, sl_frac=0.05, tp_frac=0.10)
    assert check_forced_exit(book, world, high=101, low=94, close=100,
                             is_last_bar=False).reason == "sl"
    assert check_forced_exit(book, world, high=112, low=99, close=111,
                             is_last_bar=False).reason == "tp"
    assert check_forced_exit(book, world, high=101, low=99, close=100,
                             is_last_bar=False) is None
    assert check_forced_exit(book, world, high=101, low=99, close=100,
                             is_last_bar=True).reason == "end"


def test_forced_exit_without_position():
    """Вне позиции правила мира ничего не закрывают."""
    world = WorldConfig()
    assert check_forced_exit(TradeBook(fee_side=world.fee_side), world,
                             high=1, low=1, close=1, is_last_bar=True) is None


def test_breaker_threshold():
    """Circuit breaker взводится ровно на пороге просадки."""
    world = WorldConfig(breaker_drawdown_frac=0.5)
    assert breaker_armed(0.5, world) is True
    assert breaker_armed(0.49, world) is False


def test_tradebook_guards():
    """Двойное открытие и закрытие пустой позиции запрещены."""
    book = TradeBook(fee_side=0.001)
    with pytest.raises(RuntimeError):
        book.close(1, 100.0, "agent")
    book.open(1, 100.0, 0.05, 0.1)
    with pytest.raises(RuntimeError):
        book.open(2, 100.0, 0.05, 0.1)
    assert book.mark(110.0) == pytest.approx(0.1)
    closed = book.close(5, 110.0, "agent")
    assert closed.return_pct == pytest.approx((1.1 * 0.999 ** 2 - 1.0) * 100.0)
    assert book.mark(120.0) == 0.0
