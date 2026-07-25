"""Юнит-тесты машины скрытого состояния мира среды (шаг 2 плана obs v2).

Проверяют на СИНТЕТИЧЕСКИХ данных, что среда ведёт cooldown, circuit breaker,
entered_on_up, pos_tag и откат от пика цены по правилам v7, а геттеры
`agent_state`/`world_state` чистые (мутация только в step) — это условие Т2 для
расширенного состояния.

Рыночные v1-признаки требуют не меньше 61 бара разгона (ret_60), поэтому каждый
набор начинается с `WARMUP` постоянных баров, которые прогоняются действием STAY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spotrl.config import EnvConfig, FreedomConfig, WorldConfig
from spotrl.data.state_dataset import from_frame
from spotrl.envs.spot_env import SpotFlipEnv
from spotrl.spec.actions import ActionSpec, FLIP, STAY

WARMUP = 61  # баров разгона перед сценарием (ret_60 требует >= 61 бара)
_STAY = np.array([STAY, 0, 0])
_FLIP = np.array([FLIP, 0, 0])


def _dataset(scenario_close, *, leg_dn=None, entry_signal=None,
             trans_entry_signal=None, freq="min"):
    """Собрать синтетический StateDataset: WARMUP баров по 100 + сценарий.

    Args:
        scenario_close: цены закрытия сценария (после разгона), длина m.
        leg_dn/entry_signal/trans_entry_signal: булевы ряды ДЛЯ СЦЕНАРИЯ (длины
            m); разгонная часть у них False.
        freq: шаг индекса ('min' или 'D' для теста календарного сброса CB).

    Returns:
        StateDataset длины WARMUP+m.
    """
    scen = np.asarray(scenario_close, dtype=float)
    m = len(scen)
    close = np.concatenate([np.full(WARMUP, 100.0), scen])
    n = len(close)
    idx = pd.date_range("2021-01-01", periods=n, freq=freq)

    def _pad(arr):
        """Дополнить булев ряд сценария разгонными False слева."""
        tail = np.zeros(m, dtype=bool) if arr is None else np.asarray(arr, dtype=bool)
        return np.concatenate([np.zeros(WARMUP, dtype=bool), tail])

    frame = pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": np.ones(n),
        "q_buy": np.zeros(n), "q_sell": np.zeros(n),
        "regime_code": np.full(n, 2, dtype=int),
        "leg_dn": _pad(leg_dn), "entry_signal": _pad(entry_signal),
        "trans_entry_signal": _pad(trans_entry_signal), "exit_sig": _pad(None),
    }, index=idx)
    return from_frame(frame)


def _env(dataset, *, sl=0.10, cooldown_bars=5, cb_dd=0.50, entry_own=True):
    """Среда на синтетическом наборе с управляемыми правилами; разогнана STAY."""
    world = WorldConfig(stop_loss_frac=sl, cooldown_bars=cooldown_bars,
                        breaker_drawdown_frac=cb_dd, close_at_end=True)
    freedom = FreedomConfig(exit_own=True, entry_own=entry_own, params_own=False)
    # TP отключаем гигантским порогом: сценарии закрывают позицию только SL или
    # явным FLIP, чтобы рост цены не срабатывал дефолтным тейком (10%).
    action_spec = ActionSpec(tp_buckets=(1e9,), expert_tp_index=0)
    config = EnvConfig(world=world, freedom=freedom, action_spec=action_spec,
                       episode_len=len(dataset) + 10)
    env = SpotFlipEnv(dataset, config)
    env.reset(seed=0)
    for _ in range(WARMUP):
        env.step(_STAY)          # разгон: env._t доходит до WARMUP
    assert env._t == WARMUP
    return env


def test_cooldown_arms_after_stop_loss():
    """SL взводит cooldown на cooldown_bars включительно, потом снимается."""
    # сценарий: вход на баре WARMUP, обвал на 3-м баре сценария -> SL
    scen = [100., 100., 100., 80., 100., 100., 100., 100., 100., 100., 100., 100.]
    env = _env(_dataset(scen, entry_signal=[True] * len(scen)),
               sl=0.10, cooldown_bars=3)
    env.step(_FLIP)                          # t=WARMUP -> вход по open(WARMUP+1)
    env.step(_STAY)                          # держим
    env.step(_STAY)                          # исполнение -> SL на баре с ценой 80
    close_bar = env._t                       # env._t == бар закрытия
    ws = env.world_state()
    assert ws.cooldown_active is True and ws.cooldown_remain > 0.0
    # cooldown_until = close_bar + 3; активен на close_bar..close_bar+3
    for _ in range(3):
        env.step(_STAY)
    assert env._t == close_bar + 3
    assert env.world_state().cooldown_active is True
    env.step(_STAY)
    assert env.world_state().cooldown_active is False
    assert env.world_state().cooldown_remain == 0.0


def test_signal_exit_does_not_arm_cooldown():
    """Сигнальное закрытие (директива _exit_is_signal) НЕ взводит cooldown."""
    env = _env(_dataset([100.] * 10, entry_signal=[True] * 10), cooldown_bars=4)
    env.step(_FLIP)
    env.step(_STAY)
    env._exit_is_signal = True
    env.step(_FLIP)
    assert env.book.open_trade is None
    assert env.world_state().cooldown_active is False


def test_agent_exit_arms_cooldown():
    """Обычный агентский выход (НЕсигнальный) взводит cooldown, как SL в v7."""
    env = _env(_dataset([100.] * 10, entry_signal=[True] * 10), cooldown_bars=4)
    env.step(_FLIP)
    env.step(_STAY)
    env.step(_FLIP)
    assert env.book.open_trade is None
    assert env.world_state().cooldown_active is True


def test_pos_tag_dip_vs_transition():
    """pos_tag = dip при entry_signal, transition при только trans_entry."""
    dip = _env(_dataset([100.] * 8, entry_signal=[True] * 8))
    dip.step(_FLIP)
    assert dip.agent_state().pos_tag == "dip"

    trans = _env(_dataset([100.] * 8, trans_entry_signal=[True] * 8))
    trans.step(_FLIP)
    assert trans.agent_state().pos_tag == "transition"


def test_entered_on_up_from_leg_dn_at_decision_bar():
    """entered_on_up = not leg_dn на баре РЕШЕНИЯ (t), а не на баре входа."""
    up = _env(_dataset([100.] * 8, entry_signal=[True] * 8,
                       leg_dn=[False] * 8))
    up.step(_FLIP)                            # решение на баре WARMUP: leg_dn False
    assert up.agent_state().entered_on_up is True

    down = _env(_dataset([100.] * 8, entry_signal=[True] * 8,
                         leg_dn=[True] * 8))
    down.step(_FLIP)
    assert down.agent_state().entered_on_up is False


def test_price_drawdown_tracks_peak():
    """Откат от пика цены = price/peak_price − 1 (≤0), пик из книги."""
    scen = [100., 100., 120., 110., 110., 110.]
    env = _env(_dataset(scen, entry_signal=[True] * len(scen)), sl=0.90)
    env.step(_FLIP)                           # вход по open(WARMUP+1)=100
    env.step(_STAY)                           # close=120 -> пик
    env.step(_STAY)                           # close=110
    a = env.agent_state()
    assert a.price_drawdown < 0.0
    assert abs(a.price_drawdown - (110.0 / 120.0 - 1.0)) < 1e-9


def test_circuit_breaker_trips_on_world_equity_drawdown():
    """CB взводится по просадке МИРОВОЙ эквити (леджер v7), не наградной.

    Сценарий: вход по 100, обвал до 60, агентский выход. Мировой леджер
    (cash=10M, sizing 0.9999, целые лоты, односторонняя комиссия) проседает на
    ~40% > порога 30% -> на плоском баре выхода CB взводится. sl=0 (cooldown не
    вооружается, CB оценивается сразу), cooldown_bars=0.
    """
    scen = [100., 100., 60., 60., 60.]
    env = _env(_dataset(scen, entry_signal=[True] * len(scen)),
               sl=0.0, cooldown_bars=0, cb_dd=0.30)
    # мировой леджер стартует на конвенции v7, не на наградной 1.0
    assert env._world_cash == 10_000_000.0
    env.step(_FLIP)                           # вход по open(WARMUP+1)=100
    assert env._world_size == int(10_000_000.0 * 0.9999 // 100.1)
    assert env.world_state().cb_active is False
    env.step(_STAY)                           # в позиции, close=60
    env.step(_FLIP)                           # выход по 60 -> мировая эквити −40%
    # мировая эквити просела, CB взведён; наградная эквити тут ни при чём
    assert env._world_cash < 7_000_000.0
    assert env.world_state().cb_active is True


def test_circuit_breaker_clears_next_calendar_day():
    """CB держится в день halt и снимается на следующий календарный день.

    Индекс строится вручную: бары разгона и текущий — на 2021-01-01, а
    следующий бар — на 2021-01-02, чтобы поймать смену календарного дня.
    """
    n = WARMUP + 3
    times = ([pd.Timestamp("2021-01-01") + pd.Timedelta(minutes=k)
              for k in range(n - 1)] + [pd.Timestamp("2021-01-02")])
    close = np.full(n, 100.0)
    zeros = np.zeros(n, dtype=bool)
    frame = pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": np.ones(n), "q_buy": np.zeros(n),
        "q_sell": np.zeros(n), "regime_code": np.full(n, 2, dtype=int),
        "leg_dn": zeros, "entry_signal": zeros, "trans_entry_signal": zeros,
        "exit_sig": zeros}, index=pd.DatetimeIndex(times))
    env = _env(from_frame(frame), sl=0.0, cb_dd=0.30, entry_own=False)
    assert env._dates[env._t] == pd.Timestamp("2021-01-01").date()
    env._cb_triggered = True
    env._cb_halt_date = env._dates[env._t]    # halt в день 2021-01-01
    env.step(_STAY)                           # ещё бар того же дня -> держится
    assert env.world_state().cb_active is True
    env.step(_STAY)                           # бар 2021-01-02 -> снят сегодня
    ws = env.world_state()
    assert ws.cb_active is False
    assert ws.cb_cleared_today is True


def test_getters_are_pure():
    """agent_state/world_state не мутируют состояние среды (условие Т2 для v2)."""
    scen = [100., 100., 120., 130., 120.]
    env = _env(_dataset(scen, entry_signal=[True] * len(scen)), sl=0.90)
    env.step(_FLIP)
    env.step(_STAY)
    snap = (env._t, env._cooldown_until, env._cb_triggered,
            env.book.open_trade.peak_price)
    a1, w1 = env.agent_state(), env.world_state()
    a2, w2 = env.agent_state(), env.world_state()
    assert a1 == a2 and w1 == w2
    assert snap == (env._t, env._cooldown_until, env._cb_triggered,
                    env.book.open_trade.peak_price)
