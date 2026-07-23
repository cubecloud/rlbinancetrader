"""Каркас среды этапа 1.1 (snapshot-режим).

Слои (PLAN П1-П9):
  данные   — StateCache -> (ohlcv_df, features_df), предвычислено, каузально;
  стратегия— StrategyCore (интерфейс ниже; реализация = этап 0.3, ЖДЁТ базу
             от sunday — какая механика станет базовой, решается после
             paper-вердикта 2026-08-02);
  RL       — этот gym; контекст сделки среда считает САМА из своего
             трейдбука (П4), tb_-колонки parquet не используются.

Каркасные решения, ПОМЕЧЕННЫЕ как временные (финализируются после 1.2*):
  - action space: Discrete(2) {hold, exit} c маской (exit валиден только в
    позиции). Расширяемо (masking-инфраструктура как в binanceenv).
  - reward: log(1+r_net) закрытой сделки (П8/1.4), иначе 0. Shaping запрещён.
  - исполнение: решение на баре t -> цена open(t+1) (конвенция П7).

Онлайн-режим (poверх свежих строк PG) — отдельная реализация DataFeed позже;
интерфейс среды к этому готов (она видит только массивы и указатель времени).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
except ImportError:  # rlbinancegym env использует gymnasium 0.29 — есть всегда
    raise

FEE_SIDE = 0.001  # комиссия на сторону, как в зеркале sunday


class StrategyCore:
    """Интерфейс механики базовой стратегии (реализация — этап 0.3).

    Контракт: на баре t методы видят ТОЛЬКО данные [<=t] (каузальность —
    ответственность реализации; проверяется parity-тестом гейта 0.3).
    """

    def entry_signal(self, t: int) -> bool:
        """Механика хочет войти на баре t (исполнение по open(t+1))."""
        raise NotImplementedError

    def exit_signal(self, t: int, entry_t: int) -> bool:
        """Штатный выход механики (сигнальный/legflip/TP/SL/...)."""
        raise NotImplementedError

    def hard_stop(self, t: int, entry_t: int, unreal_pnl: float) -> bool:
        """SL/circuit-breaker — у агента НЕ отбираются (PLAN 1.2)."""
        raise NotImplementedError


class TradeBook:
    """Трейдбук агента: контекст сделки и агрегаты считаются отсюда (П4)."""

    def __init__(self):
        self.trades: list[dict] = []      # закрытые
        self.entry_t: int | None = None   # открытая позиция
        self.entry_price: float = np.nan
        self.peak_unreal: float = 0.0

    @property
    def in_position(self) -> bool:
        return self.entry_t is not None

    def open(self, t: int, price: float) -> None:
        assert not self.in_position
        self.entry_t, self.entry_price, self.peak_unreal = t, price, 0.0

    def unreal_pnl(self, price: float) -> float:
        if not self.in_position:
            return 0.0
        gross = price / self.entry_price - 1.0
        return (1.0 + gross) * (1 - FEE_SIDE) ** 2 - 1.0

    def close(self, t: int, price: float, reason: str) -> float:
        r = self.unreal_pnl(price)
        self.trades.append(dict(entry_t=self.entry_t, exit_t=t,
                                ret=r, reason=reason))
        self.entry_t = None
        self.entry_price = np.nan
        return r

    def context(self, t: int, price: float) -> np.ndarray:
        """Фичи текущей сделки для observation (группа 1.3.1-«сделка»)."""
        if not self.in_position:
            return np.zeros(4, dtype=np.float32)
        u = self.unreal_pnl(price)
        self.peak_unreal = max(self.peak_unreal, u)
        return np.array([1.0,
                         (t - self.entry_t) / 1440.0,     # дни в позиции
                         u,
                         u - self.peak_unreal],           # откат от пика сделки
                        dtype=np.float32)


class ExitTimingEnv(gym.Env):
    """Snapshot-среда: механика ведёт входы, агент решает выход.

    До этапа 0.3 (нет StrategyCore) работает в режиме STUB: вход по
    простому правилу-заглушке — только чтобы гонять пайплайн end-to-end.
    Любые экономические выводы на заглушке ЗАПРЕЩЕНЫ.
    """

    metadata = {"render_modes": []}
    HOLD, EXIT = 0, 1

    def __init__(self, ohlcv_df: pd.DataFrame, features_df: pd.DataFrame,
                 strategy: StrategyCore | None = None,
                 episode_len: int = 20_000, seed: int | None = None):
        assert len(ohlcv_df) == len(features_df)
        self.open_arr = ohlcv_df["open"].to_numpy(np.float64)
        self.close_arr = ohlcv_df["close"].to_numpy(np.float64)
        # bool-колонки (leg_dn, bounce_ok) обязаны попасть в obs -> float
        num = features_df.select_dtypes(include=[np.number, bool]).astype(np.float32)
        dropped = [c for c in features_df.columns if c not in num.columns]
        assert not dropped, f"non-numeric features dropped: {dropped}"
        self.feature_names = list(num.columns)
        self.feat = np.nan_to_num(num.to_numpy(np.float32),
                                  nan=0.0, posinf=0.0, neginf=0.0)
        self.strategy = strategy
        self.episode_len = episode_len
        n_feat = self.feat.shape[1] + 4          # + контекст сделки
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                shape=(n_feat,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.book = TradeBook()
        self._rng = np.random.default_rng(seed)

    # -- masking (интерфейс MaskablePPO, как в binanceenv) --
    def action_masks(self) -> np.ndarray:
        return np.array([True, self.book.in_position])

    def _entry_stub(self, t: int) -> bool:
        """ЗАГЛУШКА до 0.3 (см. докстринг класса)."""
        return t % 720 == 0

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.feat[self.t],
                               self.book.context(self.t, self.close_arr[self.t])])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        hi = len(self.close_arr) - self.episode_len - 2
        self.t0 = int(self._rng.integers(0, max(hi, 1)))
        self.t = self.t0
        self.book = TradeBook()
        return self._obs(), {}

    def step(self, action: int):
        reward = 0.0
        price_next_open = self.open_arr[self.t + 1]

        if self.book.in_position:
            want_exit = bool(action == self.EXIT)
            hard = (self.strategy.hard_stop(self.t, self.book.entry_t,
                                            self.book.unreal_pnl(self.close_arr[self.t]))
                    if self.strategy else
                    self.book.unreal_pnl(self.close_arr[self.t]) <= -0.07)
            mech = (self.strategy.exit_signal(self.t, self.book.entry_t)
                    if self.strategy else
                    self.t - self.book.entry_t >= 2880)
            if want_exit or hard or mech:
                reason = "agent" if want_exit and not (hard or mech) else \
                         ("hard" if hard else "mech")
                r = self.book.close(self.t + 1, price_next_open, reason)
                reward = float(np.log1p(r))
        else:
            enter = (self.strategy.entry_signal(self.t) if self.strategy
                     else self._entry_stub(self.t))
            if enter:
                self.book.open(self.t + 1, price_next_open)

        self.t += 1
        terminated = self.t - self.t0 >= self.episode_len \
            or self.t >= len(self.close_arr) - 2
        return self._obs(), reward, terminated, False, {}
