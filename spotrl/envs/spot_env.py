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
        # ЛЕСА КОПИРОВАНИЯ (v7-clone scaffolding): режим эквити circuit breaker.
        # v7_ledger — CB по МИРОВОЙ эквити конвенции v7 (для паритета 6/6);
        # reward — CB по наградной эквити среды (самостоятельный режим, мировой
        # леджер не ведётся). Флаг снимается ДО _init_world_machine (пик CB
        # инициализируется по режиму). См. WorldConfig.cb_equity_mode.
        self._cb_uses_v7_ledger = self.config.world.cb_equity_mode == "v7_ledger"
        self._init_world_machine()
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
        # Входы машины скрытого состояния мира (шаг 2 плана obs v2).
        self._breaker_cb_dd = world.breaker_drawdown_frac  # порог CB (в v7 = 0.15)
        self._cooldown_bars = world.cooldown_bars          # длина post-exit cooldown
        # Мировой леджер эквити для правила circuit breaker (конвенция движка v7):
        # стартовый капитал cb_equity_cash, сайзинг cb_position_pct целыми лотами
        # (floor), комиссия ОДНОсторонняя (вход по adjusted = price*(1+fee), выход
        # по сырой цене). Считается из СОБСТВЕННОЙ книги среды, поэтому корректен
        # и в паритете (последовательность сделок = v7 → мировая эквити = v7), и
        # под живым агентом (его сделки в ту же конвенцию — не прекомпьют-ложь).
        # Наградная эквити (компаунд от 1.0, двусторонняя) — ОТДЕЛЬНО, не трогаем.
        self._cb_cash = world.cb_equity_cash
        self._cb_pos_pct = world.cb_position_pct
        self._entry_sig = state.entry_signal               # булев self._entry v7
        self._trans_sig = state.trans_entry_signal         # булев self._trans_entry v7
        self._leg_dn = state.leg_dn                         # каузальный leg_dn
        # Календарные даты баров для снятия CB по смене дня (regimeb:240-245).
        self._dates = np.array([ts.date() for ts in state.index], dtype=object)
        # Директива драйвера паритета: пометить БЛИЖАЙШЕЕ агентское закрытие как
        # сигнальное (штатный выход v7 → cooldown НЕ взводится). По умолчанию
        # False: обычный агентский выход НЕсигнальный и взводит cooldown, как в v7
        # (v7hook: агентский close не помечается сигнальным). Сбрасывается на шаге.
        self._exit_is_signal = False
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
        self._init_world_machine()
        return self._obs(), {"start_bar": self._start}

    def _init_world_machine(self) -> None:
        """Сбросить машину скрытого состояния мира (cooldown и circuit breaker).

        Состояние машины мутируется ТОЛЬКО здесь и в `step`; геттеры
        `agent_state`/`world_state` его лишь читают (гейт чистоты Т2).

        Attributes машины:
            _cooldown_until: бар, до которого включительно активен post-exit
                cooldown (−1 = не активен).
            _cb_triggered: circuit breaker сейчас держит halt.
            _cb_halt_date: календарная дата взведения CB (снятие — на следующий
                день).
            _cb_cleared_date: дата, в которую CB был снят (для «снят сегодня»).
            _cb_peak: пик эквити для расчёта просадки CB (переякоривается при
                снятии CB, как regimeb:245). Собственный пик машины CB, НЕ
                `_peak_equity` (тот монотонен и обслуживает наблюдательный слот и
                breaker_armed — переякоривать его на снятии CB нельзя). Стартовый
                пик зависит от режима: v7_ledger → мировой кэш; reward → наградная
                эквити (1.0 к моменту вызова и в __init__, и в reset).
            _world_cash: ЛЕСА КОПИРОВАНИЯ — мировая эквити CB (конвенция v7); на
                плоском баре равна кэшу счёта движка. Меняется только при закрытии
                сделки. В режиме reward НЕ ведётся (замирает на старте) и CB не
                читается.
            _world_size: ЛЕСА КОПИРОВАНИЯ — целые лоты открытой сделки в мировом
                леджере (floor). В режиме reward не ведётся.
            _world_entry_adj: ЛЕСА КОПИРОВАНИЯ — цена входа открытой сделки с
                односторонней комиссией (adjusted_price движка = price*(1+fee_side)).
                В режиме reward не ведётся.
        """
        self._cooldown_until = -1
        self._cb_triggered = False
        self._cb_halt_date = None
        self._cb_cleared_date = None
        self._world_cash = self.config.world.cb_equity_cash
        self._world_size = 0
        self._world_entry_adj = 0.0
        # Источник пика CB по режиму (см. WorldConfig.cb_equity_mode): в режиме
        # копирования — мировой кэш конвенции v7; в самостоятельном — наградная
        # эквити (компаунд от 1.0). Отклонение от буквального текста задачи
        # («_peak_equity»): _peak_equity брать НЕЛЬЗЯ, он монотонный и его нельзя
        # понизить при снятии CB — иначе CB мгновенно взводится обратно.
        self._cb_peak = self._world_cash if self._cb_uses_v7_ledger else self._equity
        self._exit_is_signal = False

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
                # Разделители сделки фиксируются на баре РЕШЕНИЯ t (= self._t):
                # вход исполняется по open(t+1)=nxt, entry_bar−1 = t.
                dec = self._t
                entered_on_up = not bool(self._leg_dn[dec])
                pos_tag = ("dip" if bool(self._entry_sig[dec])
                           else "transition" if bool(self._trans_sig[dec])
                           else "dip")
                self._open_position(nxt, self._open.item(nxt),
                                    sl_frac=sl_frac, tp_frac=tp_frac,
                                    entered_on_up=entered_on_up, pos_tag=pos_tag)
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

        # Машина cooldown: НЕсигнальное закрытие (SL или агентский выход)
        # взводит cooldown на nxt+cooldown_bars (regimeb:276, условие
        # sl_pct>0 и cooldown_bars>0). Сигнальное закрытие (в v7 — штатный
        # выход; в паритет-драйвере помечается _exit_is_signal) НЕ взводит.
        # TP/end не взводят (в v7 TP помечается сигнальным закрытием).
        triggers_cooldown = (exit_reason == "sl"
                             or (exit_reason == "agent" and not self._exit_is_signal))
        if (triggers_cooldown and self._cooldown_bars > 0 and self._stop_loss_on):
            self._cooldown_until = nxt + self._cooldown_bars

        fee = self.config.world.fee_side
        roundtrip = (1.0 - fee) ** 2  # двусторонняя комиссия (вход+выход)
        if trade is not None:
            unreal = close_next / trade.entry_price - 1.0   # валовое движение цены
            if unreal > trade.peak_unreal:
                trade.peak_unreal = unreal
            if close_next > trade.peak_price:
                trade.peak_price = close_next
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
        # Машина circuit breaker (мирроринг regimeb:232-256): проверяется на
        # ПЛОСКОМ баре (в v7 cb-проверка недостижима внутри позиции), halt по
        # просадке МИРОВОЙ эквити ≥ порога, снятие — на СЛЕДУЮЩИЙ календарный день
        # с переякориванием пика. Мировая эквити (self._world_cash, конвенция v7:
        # cash=10M, sizing 0.9999, целые лоты, односторонняя комиссия) на плоском
        # баре равна кэшу счёта движка (equity = cash + Σ pl; вне позиции Σ=0,
        # backtesting.py:790). Это ТА ЖЕ величина, что видит _check_circuit_breaker
        # v7 (self.equity), поэтому cb срабатывает бар-в-бар. Наградная эквити
        # (компаунд от 1.0, двусторонняя) для CB НЕ используется — она про reward.
        now_date = self._dates[nxt]
        # Источник эквити CB по режиму (ЛЕСА КОПИРОВАНИЯ, см. cb_equity_mode):
        #   v7_ledger — мировая эквити конвенции v7 (`_world_cash`), для паритета;
        #   reward     — наградная эквити среды (`_equity`, на плоском баре ==
        #                локальному `equity`; берём атрибут явно). Мировой леджер в
        #                режиме reward не ведётся и здесь не читается.
        world_equity = self._world_cash if self._cb_uses_v7_ledger else self._equity
        # v7 оценивает cb ТОЛЬКО на плоском баре ВНЕ cooldown: в next() ранний
        # выход `if i <= _cooldown_until: return` (regimeb:306) стоит ДО проверки
        # cb (regimeb:318). Поэтому во время cooldown cb не трогаем.
        in_cooldown = self._cooldown_until >= nxt
        if trade is None and not in_cooldown:
            if self._cb_triggered:
                if (self._cb_halt_date is not None
                        and now_date > self._cb_halt_date):
                    self._cb_triggered = False
                    self._cb_halt_date = None
                    self._cb_cleared_date = now_date
                    self._cb_peak = world_equity
            if not self._cb_triggered:
                if world_equity > self._cb_peak:
                    self._cb_peak = world_equity
                dd = (self._cb_peak - world_equity) / self._cb_peak
                if dd >= self._breaker_cb_dd:
                    self._cb_triggered = True
                    self._cb_halt_date = now_date
        self._exit_is_signal = False  # директива действует только на этот шаг
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
        peak_price = max(trade.peak_price, price)  # чистое чтение, книгу не мутируем
        return AgentState(in_position=True,
                          # ЧИСТАЯ нереализованная прибыль (реш. «а»), как в step
                          unreal_pnl=roundtrip * (1.0 + unreal) - 1.0,
                          peak_unreal=roundtrip * (1.0 + trade.peak_unreal) - 1.0,
                          bars_in_trade=self._t - trade.entry_bar,
                          dist_to_sl=unreal + trade.sl_frac,   # валовое расстояние до стопа
                          dist_to_tp=trade.tp_frac - unreal,   # валовое расстояние до тейка
                          entered_on_up=trade.entered_on_up,
                          pos_tag=trade.pos_tag,
                          price_drawdown=price / peak_price - 1.0)

    def world_state(self) -> WorldState:
        """Наблюдаемая часть правил мира на текущем баре — чистое чтение машины.

        cooldown/circuit breaker ведёт `step` (мутация только там); здесь
        значения лишь читаются и нормируются (гейт чистоты Т2).
        """
        equity_drawdown = self._equity / self._peak_equity - 1.0
        # cooldown активен, пока текущий бар <= _cooldown_until включительно
        # (regimeb:306 `if i <= self._cooldown_until: return`).
        active = self._cooldown_until >= self._t
        remaining = self._cooldown_until - self._t + 1 if active else 0
        cd_len = self._cooldown_bars if self._cooldown_bars > 0 else 1
        cooldown_remain = min(1.0, remaining / cd_len)
        return WorldState(data_age_bars=1,
                          breaker_armed=breaker_armed(-equity_drawdown, self.config.world),
                          equity_drawdown=equity_drawdown,
                          cb_active=self._cb_triggered,
                          cb_cleared_today=(self._cb_cleared_date is not None
                                            and self._cb_cleared_date == self._dates[self._t]),
                          cooldown_active=active,
                          cooldown_remain=cooldown_remain)

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
                       tp_frac: float, entered_on_up: bool = False,
                       pos_tag: str = "dip") -> None:
        """Купить на всю эквити по цене `price`; комиссия списывается сразу.

        Учёт через количество (`_qty`) делает сумму поминутных наград за
        сделку тождественной `log(1 + Return)` (тест Т7). Разделители сделки
        (entered_on_up, pos_tag) фиксируются на баре входа и пишутся в книгу.
        """
        fee = self.config.world.fee_side
        self._qty = self._equity * (1.0 - fee) / float(price)
        self._equity = self._qty * float(price)
        # ЛЕСА КОПИРОВАНИЯ — мировой леджер CB (конвенция v7): целые лоты по
        # adjusted-цене входа. size = floor(cash*pct/adjusted); backtesting.py:
        # 898,937 (adjusted=892). В режиме reward леджер не ведётся — не тратим
        # шаг на него (CB его не читает).
        if self._cb_uses_v7_ledger:
            adjusted = float(price) * (1.0 + fee)
            self._world_entry_adj = adjusted
            self._world_size = int((self._world_cash * self._cb_pos_pct) // adjusted)
        self.book.open(bar, price, sl_frac=sl_frac, tp_frac=tp_frac,
                       entered_on_up=entered_on_up, pos_tag=pos_tag)

    def _close_position(self, bar: int, price: float, reason: str) -> None:
        """Продать позицию по цене `price`; комиссия списывается сразу."""
        if self.book.open_trade is None:
            return
        fee = self.config.world.fee_side
        self._equity = self._qty * float(price) * (1.0 - fee)
        self._qty = 0.0
        # ЛЕСА КОПИРОВАНИЯ — мировой леджер CB: выход по СЫРОЙ цене (комиссия
        # односторонняя, взята на входе); pl = size*(exit − entry_adj), cash += pl
        # (backtesting.py: 919 закрытие по сырой цене, 641 pl, 998 cash += pl).
        # В режиме reward леджер не ведётся — не тратим шаг (CB его не читает).
        if self._cb_uses_v7_ledger:
            self._world_cash += self._world_size * (float(price) - self._world_entry_adj)
            self._world_size = 0
        self.book.close(bar, price, reason)


__all__ = ["SpotFlipEnv", "FLIP"]
