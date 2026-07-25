"""Гейт паритета env↔expert-log (главный чек копирования части 1 obs v2).

Что доказывает: СРЕДА, прогоняемая потоком экспертных действий v7 (повторяет
каждый вход/выход эксперта), кладёт в `agent_state()`/`world_state()` те же
разделители, что видел эксперт в своём логе — бар-в-бар. Это сильнее гейтов
1.2а: те доказывали разделимость решения ЭКСПЕРТА, но не что среда воспроизводит
те же числа своей машиной скрытого состояния.

Сверяются побарно (на баре РЕШЕНИЯ t, до шага): in_position, pos_tag,
cooldown_active, cb_active. entered_on_up сверяется на баре входа против
not leg_dn[entry_bar−1] из trades. Число расхождений печатается по КАЖДОМУ
разделителю и по КАЖДОЙ эпохе.

Оговорка (подтверждена дизайном): числовой паритет cb_active НЕ гарантирован в
части 1 — cb-эквити среды (компаунд от 1.0, двусторонняя комиссия) отличается от
портфельной эквити v7 (0.9999, целые лоты, односторонняя комиссия). Логика cb
(взвод → halt → снятие по календарному дню) воспроизведена; числовой паритет cb
относится к части 3 (паритет эквити/источников). Расхождение cb печатается
ЧЕСТНО отдельным числом, а не прячется.

Запуск (env rlbinancetrader — есть pyarrow, sunday не требуется):
  python -m spotrl.tests.parity_env_expert
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from spotrl.config import EnvConfig, FreedomConfig, WorldConfig
from spotrl.data.state_dataset import attach_signals, load_state
from spotrl.envs.spot_env import SpotFlipEnv
from spotrl.spec.actions import ActionSpec, FLIP, STAY

# v7 (P7) параметры мира — среда обязана прогоняться на них, иначе тайминги
# закрытий и cooldown разъедутся ДО сверки состояния.
V7_SL = 0.07
V7_COOLDOWN_BARS = 423
V7_CB_DD = 0.15
# Причины выхода, которые в v7 закрывают сделку сигнально (штатный выход) — их
# драйвер исполняет как агентский FLIP-close с пометкой «сигнальное закрытие»
# (cooldown НЕ взводится). SL и forced_eod исполняет сама среда.
SIGNAL_EXIT_REASONS = frozenset({"tp", "legflip", "signal", "b2b", "agent"})
DRIVEN_BY_ENV = frozenset({"sl", "forced_eod"})


@dataclass
class ParityCounts:
    """Счётчики расхождений env↔expert по одной эпохе.

    Attributes:
        compared: сколько баров сверено.
        in_position: расхождений по флагу «в позиции».
        pos_tag: расхождений по типу сделки (dip/transition/none).
        cooldown: расхождений по cooldown_active.
        cb: расхождений по cb_active (числовой паритет — часть 3).
        entered_on_up: расхождений по entered_on_up на барах входа.
        first_mismatch: примеры первых расхождений по каждому полю.
    """

    compared: int = 0
    in_position: int = 0
    pos_tag: int = 0
    cooldown: int = 0
    cb: int = 0
    entered_on_up: int = 0
    ledger_size: int = 0
    ledger_price: int = 0
    trades_checked: int = 0
    first_mismatch: Dict[str, str] = field(default_factory=dict)

    def note(self, field_name: str, msg: str) -> None:
        """Запомнить первый пример расхождения по полю."""
        self.first_mismatch.setdefault(field_name, msg)

    def describe(self) -> dict:
        """Сериализуемая сводка для отчёта."""
        return {"compared": self.compared, "in_position": self.in_position,
                "pos_tag": self.pos_tag, "cooldown": self.cooldown, "cb": self.cb,
                "entered_on_up": self.entered_on_up,
                "trades_checked": self.trades_checked,
                "ledger_size": self.ledger_size, "ledger_price": self.ledger_price,
                "first_mismatch": self.first_mismatch}


def _make_env(state_path: str, signals_path: str) -> SpotFlipEnv:
    """Собрать среду на v7-параметрах для полного детерминированного прогона."""
    dataset = attach_signals(load_state(state_path), signals_path)
    n = len(dataset)
    world = WorldConfig(stop_loss_frac=V7_SL, cooldown_bars=V7_COOLDOWN_BARS,
                        breaker_drawdown_frac=V7_CB_DD, close_at_end=True)
    freedom = FreedomConfig(exit_own=True, entry_own=True, params_own=False)
    # TP отключаем (гигантский порог): все НЕ-SL выходы исполняет драйвер
    # экспертным FLIP-close, чтобы среда не срабатывала своим TP раньше v7.
    action_spec = ActionSpec(tp_buckets=(1e9,), expert_tp_index=0)
    config = EnvConfig(world=world, freedom=freedom, action_spec=action_spec,
                       episode_len=n + 10)
    return SpotFlipEnv(dataset, config)


def run_parity(state_path: str, signals_path: str, expert_path: str,
               trades_path: str) -> ParityCounts:
    """Прогнать среду потоком экспертных действий и сверить состояние.

    Args:
        state_path: parquet каузального state.
        signals_path: артефакт булевых сигналов v7 (dump_v7_signals).
        expert_path: parquet побарного экспертного лога v7.
        trades_path: CSV сделок v7 с колонками entry_bar/exit_bar/exit_reason.

    Returns:
        ParityCounts — число расхождений по каждому разделителю.
    """
    env = _make_env(state_path, signals_path)
    n = env._n_bars
    expert = pd.read_parquet(expert_path)
    trades = pd.read_csv(trades_path)
    leg_dn = env.state.leg_dn

    # Экспертные факты по бару решения.
    entry_placed = np.zeros(n, dtype=bool)
    exp_in_pos = np.zeros(n, dtype=bool)
    exp_pos_tag = np.array([""] * n, dtype=object)
    exp_cooldown = np.zeros(n, dtype=bool)
    exp_cb = np.zeros(n, dtype=bool)
    logged = np.zeros(n, dtype=bool)
    b = expert["bar"].to_numpy()
    entry_placed[b] = expert["entry_placed"].to_numpy().astype(bool)
    exp_in_pos[b] = expert["in_pos_before"].to_numpy().astype(bool)
    exp_pos_tag[b] = expert["pos_tag"].to_numpy().astype(str)
    exp_cooldown[b] = expert["cooldown_active"].to_numpy().astype(bool)
    exp_cb[b] = expert["cb_active"].to_numpy().astype(bool)
    logged[b] = True

    # Бар решения о НЕ-SL выходе (exit_bar−1) и признак «сигнального» закрытия.
    signal_exit_at = {}   # decision_bar -> True (все они сигнальные в P7)
    for tr in trades.itertuples():
        if str(tr.exit_reason) in SIGNAL_EXIT_REASONS:
            signal_exit_at[int(tr.exit_bar) - 1] = True
    # entered_on_up эталон по сделкам: not leg_dn[entry_bar−1].
    ref_entered_on_up = {int(tr.entry_bar): (not bool(leg_dn[int(tr.entry_bar) - 1]))
                         for tr in trades.itertuples()}
    # Эталон мирового леджера v7 по сделкам (для по-сделочного ассерта: size,
    # adjusted-цена входа, сырая цена выхода). Индекс — entry_bar.
    ref_ledger = {int(tr.entry_bar): (float(tr.size), float(tr.entry_price),
                                      float(tr.exit_price))
                  for tr in trades.itertuples()}
    env_ledger: Dict[int, tuple] = {}   # entry_bar -> (world_size, world_entry_adj)

    counts = ParityCounts()
    env.reset(seed=0)
    assert env._start == 0, f"ожидался старт с бара 0, получен {env._start}"
    seen_entry_bars = set()
    stay = np.array([STAY, 0, 0])
    flip = np.array([FLIP, 0, 0])

    while True:
        t = env._t
        trade = env.book.open_trade
        if logged[t]:
            counts.compared += 1
            env_in_pos = trade is not None
            if env_in_pos != bool(exp_in_pos[t]):
                counts.in_position += 1
                counts.note("in_position", f"bar {t}: env {env_in_pos} vs exp {exp_in_pos[t]}")
            env_tag = trade.pos_tag if trade is not None else "none"
            exp_tag = exp_pos_tag[t] if exp_pos_tag[t] else "none"
            if env_in_pos and bool(exp_in_pos[t]) and env_tag != exp_tag:
                counts.pos_tag += 1
                counts.note("pos_tag", f"bar {t}: env {env_tag} vs exp {exp_tag}")
            ws = env.world_state()
            if ws.cooldown_active != bool(exp_cooldown[t]):
                counts.cooldown += 1
                counts.note("cooldown", f"bar {t}: env {ws.cooldown_active} vs exp {exp_cooldown[t]}")
            if ws.cb_active != bool(exp_cb[t]):
                counts.cb += 1
                counts.note("cb", f"bar {t}: env {ws.cb_active} vs exp {exp_cb[t]}")
        # entered_on_up на баре входа
        if trade is not None and trade.entry_bar not in seen_entry_bars:
            seen_entry_bars.add(trade.entry_bar)
            # снимок мирового леджера на баре входа (size/adjusted-цена уже
            # посчитаны средой в _open_position).
            env_ledger[trade.entry_bar] = (env._world_size, env._world_entry_adj)
            ref = ref_entered_on_up.get(trade.entry_bar)
            if ref is not None and bool(trade.entered_on_up) != ref:
                counts.entered_on_up += 1
                counts.note("entered_on_up",
                            f"entry_bar {trade.entry_bar}: env {trade.entered_on_up} vs ref {ref}")

        # Решение драйвера на баре t.
        action = stay
        env._exit_is_signal = False
        if trade is None:
            if entry_placed[t]:
                action = flip
        else:
            if t in signal_exit_at:
                action = flip
                env._exit_is_signal = True
            # SL/forced_eod — среда закроет сама (STAY).

        _obs, _r, _term, trunc, _info = env.step(action)
        if trunc:
            break

    # По-сделочный ассерт мирового леджера против trades CSV: это ПРЯМАЯ проверка
    # того, что cb-паритет получен из совпадения эквити-траектории, а не из
    # компенсирующих ошибок (совет ревьюера). Сверяем size (целые лоты),
    # adjusted-цену входа (raw*(1+fee) ≈ CSV entry_price) и сырую цену выхода
    # (≈ CSV exit_price). Порог по цене — относительный, 1e-6.
    for closed in env.book.closed:
        ref = ref_ledger.get(closed.entry_bar)
        snap = env_ledger.get(closed.entry_bar)
        if ref is None or snap is None:
            continue
        counts.trades_checked += 1
        ref_size, ref_entry, ref_exit = ref
        env_size, env_entry_adj = snap
        if int(env_size) != int(ref_size):
            counts.ledger_size += 1
            counts.note("ledger_size",
                        f"entry_bar {closed.entry_bar}: env {env_size} vs ref {ref_size}")
        if abs(env_entry_adj - ref_entry) > 1e-6 * ref_entry:
            counts.ledger_price += 1
            counts.note("ledger_entry",
                        f"entry_bar {closed.entry_bar}: env_adj {env_entry_adj} vs ref {ref_entry}")
        # Терминальное принудительное закрытие ("end"): среда закрывает по CLOSE
        # последнего бара, а движок v7 — рыночным trade.close() по OPEN последнего
        # бара (backtesting.py:1221-1224 → :858, trade_on_close=False). Отсюда
        # расхождение цены выхода единственной терминальной сделки. После
        # последнего бара плоских баров нет — на CB это не влияет; из ценовой
        # сверки терминальную сделку исключаем (в 2021 её и так 0).
        if closed.exit_reason == "end":
            continue
        if abs(closed.exit_price - ref_exit) > 1e-6 * ref_exit:
            counts.ledger_price += 1
            counts.note("ledger_exit",
                        f"entry_bar {closed.entry_bar}: env {closed.exit_price} vs ref {ref_exit}")
    return counts


def main() -> None:
    """CLI: прогнать паритет по обеим эпохам и напечатать числа расхождений."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/cubecloud/Data")
    args = ap.parse_args()
    base = args.data
    epochs = [
        ("2021", f"{base}/sunday_tests/state_v0/state_v3_causal_2021-01_2024-03.parquet",
         f"{base}/rlbinancetrader/v7signals_2021.parquet",
         f"{base}/rlbinancetrader/expert_v7_2021-01_2024-03.parquet",
         f"{base}/rlbinancetrader/trades_v7_labeled_2021.csv"),
        ("2024", f"{base}/sunday_tests/state_v0/state_v3_causal_2024-03_2026-07.parquet",
         f"{base}/rlbinancetrader/v7signals_2024.parquet",
         f"{base}/rlbinancetrader/expert_v7_2024-03_2026-07.parquet",
         f"{base}/rlbinancetrader/trades_v7_labeled_2024.csv"),
    ]
    for tag, state, signals, expert, trades in epochs:
        counts = run_parity(state, signals, expert, trades)
        print(f"[{tag}] {counts.describe()}")


if __name__ == "__main__":
    main()
