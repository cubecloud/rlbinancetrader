"""Т8 — горячий путь шага: он обязан совпадать с медленным, читаемым путём.

`step` считает наблюдение и жёсткие правила «плоско», без создания объектов.
Рядом живут те же вычисления в читаемом виде: `agent_state`, `world_state`,
`observe` и `worldrules.check_forced_exit`. Эти тесты держат две реализации
вместе: разошлись — тест красный, а не «странные числа через месяц».

Здесь же закреплено соглашение о буфере наблюдения (требование скорости) и
непустой вид гейта чистоты Э1.2б: сравниваются ДВА РАЗНЫХ буфера, а не одна
и та же память дважды.
"""
from __future__ import annotations

import numpy as np
import pytest

from spotrl.config import ActionSpec, EnvConfig, FreedomConfig, WorldConfig
from spotrl.envs.spot_env import SpotFlipEnv
from spotrl.envs.tradebook import OpenTrade, TradeBook
from spotrl.envs.worldrules import check_forced_exit
from spotrl.features.builder import ObservationLayout, write_observation_v2_flat
from spotrl.spec.actions import FLIP, STAY
from spotrl.spec.observation import RESERVED_SLOT_VALUE


def _actions(n, p_flip, seed=11, nvec=(2, 2, 2)):
    """Поток действий с заданной долей FLIP."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 3), dtype=np.int64)
    out[:, 0] = (rng.random(n) < p_flip).astype(np.int64)
    out[:, 1] = rng.integers(nvec[1], size=n)
    out[:, 2] = rng.integers(nvec[2], size=n)
    return out


def test_step_obs_equals_slow_path(env):
    """Наблюдение из `step` побитово равно `observe()` на каждом баре."""
    env.reset(seed=3)
    for action in _actions(1_500, p_flip=0.03):
        obs, _, _, truncated, _ = env.step(action)
        assert np.array_equal(obs, env.observe()), "быстрый путь разошёлся с observe()"
        if truncated:
            env.reset(seed=4)


@pytest.fixture()
def tight_env(state):
    """Среда с узкими SL/TP: иначе правила мира на синтетике не срабатывают.

    У ряда из фикстуры шаг 0.1% в минуту, поэтому корзины 3-5% за 2000 баров
    не задеваются ни разу и тест правил был бы пустым.
    """
    config = EnvConfig(world=WorldConfig(stop_loss_frac=0.002),
                       freedom=FreedomConfig(exit_own=True, entry_own=True,
                                             params_own=True),
                       action_spec=ActionSpec(sl_buckets=(0.002, 0.005),
                                              tp_buckets=(0.003, 0.006),
                                              expert_sl_index=0, expert_tp_index=0),
                       episode_len=2_000)
    return SpotFlipEnv(state, config)


def test_forced_exit_matches_worldrules(tight_env):
    """Принудительные выходы шага совпадают с `check_forced_exit` по причине и цене."""
    env = tight_env
    env.reset(seed=5)
    seen = set()
    for action in _actions(4_000, p_flip=0.01, seed=21):
        trade = env.book.open_trade
        snapshot = None if trade is None else OpenTrade(
            entry_bar=trade.entry_bar, entry_price=trade.entry_price,
            sl_frac=trade.sl_frac, tp_frac=trade.tp_frac,
            peak_unreal=trade.peak_unreal)
        nxt = env._t + 1
        _, _, _, truncated, info = env.step(action)
        reason = info["exit_reason"]
        seen.add(reason)
        if reason in ("sl", "tp", "end"):
            book = TradeBook(fee_side=env.config.world.fee_side)
            book.open_trade = snapshot
            expected = check_forced_exit(
                book, env.config.world,
                high=float(env._high[nxt]), low=float(env._low[nxt]),
                close=float(env._close[nxt]), is_last_bar=nxt >= len(env.state) - 1)
            assert expected is not None and expected.reason == reason
            assert env.book.closed[-1].exit_price == expected.price
        if truncated:
            env.reset(seed=6)
    assert {"agent", "sl", "tp"} <= seen, f"ветки правил не задеты, увидели {seen}"


def test_step_returns_buffer_and_fresh_array_on_truncation(env):
    """Шаг отдаёт свой буфер, а на усечении — свежий массив (для DummyVecEnv)."""
    env.reset(seed=7)
    first, _, _, _, _ = env.step(np.array([STAY, 0, 0]))
    second, _, _, _, _ = env.step(np.array([STAY, 0, 0]))
    assert first is second, "вне усечения буфер обязан переиспользоваться"
    truncated = False
    while not truncated:
        obs, _, _, truncated, _ = env.step(np.array([STAY, 0, 0]))
    assert obs is not first, "на усечении обязан вернуться отдельный массив"


def test_observe_returns_independent_arrays(env):
    """`observe()` каждый раз даёт НОВЫЙ массив — копия для внешнего кода."""
    env.reset(seed=8)
    env.step(np.array([FLIP, 0, 0]))
    first, second = env.observe(), env.observe()
    assert first is not second
    assert np.array_equal(first, second)


def test_write_observation_two_buffers_are_bitwise_equal(env_config):
    """Гейт Э1.2б в непустом виде: два РАЗНЫХ буфера v2 от одних входов равны."""
    spec = env_config.obs_spec
    layout = ObservationLayout.from_spec(spec)
    market_row = np.arange(len(spec.market), dtype=np.float32) * 0.125
    # плоское ядро v2: in_position, unreal, peak, price_dd, dist_sl, dist_tp,
    # entered_on_up, pos_tag, cb_active, cb_cleared, cooldown_remain, bars,
    # data_age, breaker, equity_drawdown
    args = (market_row, 1.0, 0.031, 0.042, -0.01, 0.081, 0.059, 1.0, "dip",
            0.0, 0.0, 0.4, 17.0, 1.0, 0.0, -0.12)
    left = write_observation_v2_flat(
        np.empty(layout.size, dtype=np.float32), *args, layout, spec.constants)
    right = write_observation_v2_flat(
        np.empty(layout.size, dtype=np.float32), *args, layout, spec.constants)
    assert left is not right
    assert np.array_equal(left, right)


def test_write_observation_can_skip_reserved_slots(env_config):
    """С `write_reserved=False` слоты-заглушки v2 не переписываются."""
    spec = env_config.obs_spec
    layout = ObservationLayout.from_spec(spec)
    buffer = np.zeros(layout.size, dtype=np.float32)
    buffer[layout.reserved_at:] = 7.0
    args = (np.zeros(layout.n_market, dtype=np.float32),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "none",
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    write_observation_v2_flat(buffer, *args, layout, spec.constants, False)
    assert np.all(buffer[layout.reserved_at:] == 7.0)
    write_observation_v2_flat(buffer, *args, layout, spec.constants, True)
    assert np.all(buffer[layout.reserved_at:] == RESERVED_SLOT_VALUE)


def test_invalid_action_indices_raise(env):
    """Недопустимые индексы голов — исключение, а не молчаливый разбор."""
    env.reset(seed=9)
    with pytest.raises(ValueError):
        env.step(np.array([2, 0, 0]))
    with pytest.raises(ValueError):
        env.step(np.array([FLIP, -1, 0]))


def test_frozen_freedoms_are_no_ops(state):
    """При выключенных свободах FLIP не открывает и не закрывает позицию.

    Гейт эквивалентности гоняется на включённых свободах, поэтому ветка
    «среда исполняет решение эксперта, а не агента» закрывается здесь.
    """
    config = EnvConfig(freedom=FreedomConfig(exit_own=False, entry_own=False),
                       episode_len=500)
    env = SpotFlipEnv(state, config)
    env.reset(seed=12)
    for action in _actions(400, p_flip=0.5, seed=31, nvec=config.action_spec.nvec):
        obs, _, _, truncated, info = env.step(action)
        assert not env.book.in_position, "вход запрещён флагом entry_own"
        assert info["exit_reason"] == ""
        assert np.array_equal(obs, env.observe())
        if truncated:
            break
