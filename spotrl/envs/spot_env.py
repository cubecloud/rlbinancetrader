"""Среда спота long/flat: один бит решения на бар плюс головы SL/TP.

Ключевые решения (PLAN v0.5):
  действия  — MultiDiscrete([2, n_SL, n_TP]); голова 0 = {STAY, FLIP},
              головы SL/TP содержательны только на баре входа (3.5);
  награда   — поминутный лог-прирост эквити, ПАКЕТОМ с gamma = 1.0 (5.1);
  исполнение— решение на баре t исполняется по open(t+1) (П7);
  эпизод    — обрыв по длине даёт truncated=True, а не terminated (5.2, Т5).

Чего в среде НЕТ: второй реализации механики v7 (П3). Сравнение с v7 идёт
через bridge/, а не через копию стратегии рядом со средой.

Про скорость шага. Горячий путь (`step`) намеренно написан «плоско»: без
создания объектов на бар, с предвыделенным буфером наблюдения и с
параметрами конфигурации, снятыми в атрибуты в `__init__`. Смысловые
функции (`agent_state`, `world_state`, `worldrules.check_forced_exit`)
остаются публичными и проверяются тестами; шаг повторяет их вычисления
числом-в-число — это закреплено гейтом эквивалентности на 100 000 шагов.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from spotrl.config import EnvConfig
from spotrl.data.state_dataset import StateDataset
from spotrl.features.builder import (AgentState, ObservationLayout, WorldState,
                                     build_observation, precompute_market_features,
                                     write_observation)
from spotrl.envs.tradebook import TradeBook
from spotrl.envs.worldrules import breaker_armed
from spotrl.spec.actions import FLIP
from spotrl.spec.observation import RESERVED_SLOT_VALUE


class SpotFlipEnv(gym.Env):
    """gym.Env спота: long/flat, размер позиции — константа (плеча нет).

    Attributes:
        config: параметры среды (правила мира, свободы, спецификации).
        state: источник рыночных данных.
        book: книга сделок текущего эпизода.
    """

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(self, state: StateDataset, config: Optional[EnvConfig] = None) -> None:
        """Создать среду поверх готового StateDataset."""
        super().__init__()
        self.config = config or EnvConfig()
        self.state = state
        self._market = precompute_market_features(state)
        self._close = state.close
        self._open = state.open
        self._high = state.ohlcv[:, 1]
        self._low = state.ohlcv[:, 2]
        self._n_bars = len(state)
        if self._n_bars < 3:
            raise ValueError("нужно хотя бы 3 бара")
        self.action_space = gym.spaces.MultiDiscrete(np.array(self.config.action_spec.nvec))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.config.obs_spec.size,), dtype=np.float32)
        self.book = TradeBook(fee_side=self.config.world.fee_side)
        self._t = 0
        self._start = 0
        self._equity = 1.0
        self._peak_equity = 1.0
        self._qty = 0.0
        # Снимки конфигурации для горячего шага: цепочка `self.config.world.x`
        # стоит 0.043 мкс за обращение, а таких обращений на шаг больше десяти.
        world, freedom, spec = self.config.world, self.config.freedom, self.config.action_spec
        self._fee = world.fee_side
        self._fee_close = 1.0 - world.fee_side
        self._stop_loss_on = world.stop_loss_frac > 0.0
        self._close_at_end = world.close_at_end
        self._breaker_dd = world.breaker_drawdown_frac
        self._exit_own = freedom.exit_own
        self._entry_own = freedom.entry_own
        self._params_own = freedom.params_own
        self._sl_buckets = spec.sl_buckets
        self._tp_buckets = spec.tp_buckets
        self._n_sl = len(spec.sl_buckets)
        self._n_tp = len(spec.tp_buckets)
        self._default_sl = world.stop_loss_frac
        self._default_tp = spec.tp_buckets[spec.expert_tp_index]
        self._episode_len = self.config.episode_len
        self._last_bar = self._n_bars - 1
        self._layout = ObservationLayout.from_spec(self.config.obs_spec)
        self._obs_buf = np.zeros(self._layout.size, dtype=np.float32)
        # Слоты-заглушки константны: заполняются один раз, шаг их не трогает.
        self._obs_buf[self._layout.reserved_at:] = RESERVED_SLOT_VALUE

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Начать эпизод. Стартовый бар выбирается случайно по сиду."""
        super().reset(seed=seed)
        horizon = min(self.config.episode_len, self._n_bars - 2)
        last_start = max(0, self._n_bars - horizon - 2)
        self._start = int(self.np_random.integers(0, last_start + 1))
        self._t = self._start
        self.book = TradeBook(fee_side=self.config.world.fee_side)
        self._equity = 1.0
        self._peak_equity = 1.0
        self._qty = 0.0
        return self._obs(), {"start_bar": self._start}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Один бар: решение на t -> исполнение по open(t+1) -> награда.

        Возвращаемое наблюдение — ВНУТРЕННИЙ буфер среды: он действителен до
        следующего вызова `step`/`reset` (обычное соглашение gymnasium; SB3
        `DummyVecEnv._save_obs` копирует его сразу же). На шаге с
        `truncated=True` возвращается свежий массив, иначе `DummyVecEnv`
        положил бы в `info["terminal_observation"]` ссылку на буфер и затёр
        бы его немедленным `reset` (dummy_vec_env.py:69-71) — а PPO берёт
        оттуда значение для бутстрапа усечённого эпизода
        (on_policy_algorithm.py:216-222). Кому нужна копия на каждом шаге —
        `observe()` без аргументов.
        """
        pos = int(action[0])
        book = self.book
        trade = book.open_trade
        nxt = self._t + 1
        equity_before = self._equity
        exit_reason = ""

        if pos == FLIP:
            if trade is not None:
                if self._exit_own:
                    self._close_position(nxt, self._open.item(nxt), "agent")
                    trade = None
                    exit_reason = "agent"
            elif self._entry_own:
                if self._params_own:
                    i_sl, i_tp = int(action[1]), int(action[2])
                    if not 0 <= i_sl < self._n_sl or not 0 <= i_tp < self._n_tp:
                        raise ValueError(f"головы SL/TP: индексы вне корзин {i_sl}, {i_tp}")
                    sl_frac = self._sl_buckets[i_sl]
                    tp_frac = self._tp_buckets[i_tp]
                else:
                    sl_frac = self._default_sl
                    tp_frac = self._default_tp
                self._open_position(nxt, self._open.item(nxt),
                                    sl_frac=sl_frac, tp_frac=tp_frac)
                trade = book.open_trade
        elif pos != 0:
            raise ValueError(f"голова позиции: недопустимый индекс {pos}")

        is_last = nxt >= self._last_bar
        close_next = self._close.item(nxt)
        if trade is not None:
            # Порядок проверок — как в worldrules.check_forced_exit: стоп
            # внутрибарно по Low, тейк по close, закрытие на конце данных.
            entry_price = trade.entry_price
            stop_price = entry_price * (1.0 - trade.sl_frac)
            if self._stop_loss_on and self._low.item(nxt) <= stop_price:
                self._close_position(nxt, stop_price, "sl")
                trade, exit_reason = None, "sl"
            elif close_next >= entry_price * (1.0 + trade.tp_frac):
                self._close_position(nxt, close_next, "tp")
                trade, exit_reason = None, "tp"
            elif is_last and self._close_at_end:
                self._close_position(nxt, close_next, "end")
                trade, exit_reason = None, "end"

        fee = self.config.world.fee_side
        roundtrip = (1.0 - fee) ** 2  # двусторонняя комиссия (вход+выход)
        if trade is not None:
            unreal = close_next / trade.entry_price - 1.0   # валовое движение цены
            if unreal > trade.peak_unreal:
                trade.peak_unreal = unreal
            equity = self._qty * close_next
            self._equity = equity
            in_position = 1.0
            # в наблюдение — ЧИСТАЯ нереализованная прибыль (что агент реально
            # получит при выходе сейчас), согласованная с equity/наградой (реш. «а»)
            obs_unreal = roundtrip * (1.0 + unreal) - 1.0
            obs_peak = roundtrip * (1.0 + trade.peak_unreal) - 1.0
            bars_in_trade = float(nxt - trade.entry_bar)
            dist_to_sl = unreal + trade.sl_frac   # расстояние до цены стопа — валовое
            dist_to_tp = trade.tp_frac - unreal   # расстояние до цены тейка — валовое
        else:
            equity = self._equity
            obs_unreal = obs_peak = dist_to_sl = dist_to_tp = 0.0
            in_position = bars_in_trade = 0.0

        if equity > self._peak_equity:
            self._peak_equity = equity
        # знаковая просадка (форма observation_design: equity/пик − 1, ≤0);
        # breaker_armed принимает МОДУЛЬ просадки (≥0), поэтому передаём -drawdown
        equity_drawdown = equity / self._peak_equity - 1.0
        reward = math.log(equity / equity_before)
        self._t = nxt
        truncated = is_last or (nxt - self._start) >= self._episode_len
        info = {"equity": equity, "exit_reason": exit_reason,
                "n_closed": len(book.closed)}

        layout = self._layout
        obs = write_observation(
            np.empty(layout.size, dtype=np.float32) if truncated else self._obs_buf,
            self._market[nxt], in_position, obs_unreal, obs_peak, bars_in_trade,
            dist_to_sl, dist_to_tp, 1.0, float(-equity_drawdown >= self._breaker_dd),
            equity_drawdown, layout, truncated)
        # terminated всегда False: эпизод не имеет поглощающего состояния,
        # обрыв по длине — это усечение (PLAN 5.2, тест Т5).
        return obs, reward, False, truncated, info

    def agent_state(self) -> AgentState:
        """Состояние агента на текущем баре — чистая выборка из книги сделок."""
        trade = self.book.open_trade
        if trade is None:
            return AgentState()
        price = float(self._close[self._t])
        unreal = price / trade.entry_price - 1.0            # валовое движение цены
        roundtrip = (1.0 - self.config.world.fee_side) ** 2  # двусторонняя комиссия
        return AgentState(in_position=True,
                          # ЧИСТАЯ нереализованная прибыль (реш. «а»), как в step
                          unreal_pnl=roundtrip * (1.0 + unreal) - 1.0,
                          peak_unreal=roundtrip * (1.0 + trade.peak_unreal) - 1.0,
                          bars_in_trade=self._t - trade.entry_bar,
                          dist_to_sl=unreal + trade.sl_frac,   # валовое расстояние до стопа
                          dist_to_tp=trade.tp_frac - unreal)   # валовое расстояние до тейка

    def world_state(self) -> WorldState:
        """Наблюдаемая часть правил мира на текущем баре."""
        equity_drawdown = self._equity / self._peak_equity - 1.0
        return WorldState(data_age_bars=1,
                          breaker_armed=breaker_armed(-equity_drawdown, self.config.world),
                          equity_drawdown=equity_drawdown)

    def observe(self) -> np.ndarray:
        """Наблюдение на текущем баре в НОВОМ массиве (буфер не задействован).

        Публичная точка для тех, кому нужна собственная копия наблюдения:
        результат `step` живёт только до следующего шага. Вызов не меняет
        состояние среды (гейт Э1.2б, тест Т2).
        """
        return build_observation(self._market[self._t], self.agent_state(),
                                 self.world_state(), self.config.obs_spec)

    def _obs(self) -> np.ndarray:
        """Наблюдение на текущем баре; вызов не меняет состояние среды (Т2)."""
        return self.observe()

    def _open_position(self, bar: int, price: float, sl_frac: float,
                       tp_frac: float) -> None:
        """Купить на всю эквити по цене `price`; комиссия списывается сразу.

        Учёт через количество (`_qty`) делает сумму поминутных наград за
        сделку тождественной `log(1 + Return)` (тест Т7).
        """
        fee = self.config.world.fee_side
        self._qty = self._equity * (1.0 - fee) / float(price)
        self._equity = self._qty * float(price)
        self.book.open(bar, price, sl_frac=sl_frac, tp_frac=tp_frac)

    def _close_position(self, bar: int, price: float, reason: str) -> None:
        """Продать позицию по цене `price`; комиссия списывается сразу."""
        if self.book.open_trade is None:
            return
        fee = self.config.world.fee_side
        self._equity = self._qty * float(price) * (1.0 - fee)
        self._qty = 0.0
        self.book.close(bar, price, reason)


__all__ = ["SpotFlipEnv", "FLIP"]
