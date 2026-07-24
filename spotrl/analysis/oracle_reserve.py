"""Frozen oracle-резерв свободы раннего выхода из dip-сделок (этап Э0.2).

Определения зафиксированы в pre-reg (handoff/prereg_oracle_reserve_2026-07-25.md):

Oracle видит будущее, но исполняется строго по семантике движка — выход на баре
решения ``d`` даёт цену ``open[d+1] * (1 - commission)`` (никакого ``max(High)``).
Для каждой dip-сделки ``[entry_bar, exit_bar]`` (границы чистой v7):

- hook-eligible бары ИСПОЛНЕНИЯ (fill) = ``[entry_bar+2 .. exit_bar-1]``
  (на входном баре позиция ещё не открыта; на баре механического решения
  ``exit_bar-1`` родитель уже закрыл — хук заблокирован);
- ``price_oracle = max(open[b] * (1 - commission))`` по этим барам;
- ранний выход берётся, только если он строго лучше штатного выхода v7
  (иначе держим до механики) — отсюда дельта ≥ 0 по построению.

Всё считается в frozen-пространстве (посделочные ``ReturnPct`` на исходных
границах v7). Компаунд движка живёт в других единицах и здесь НЕ смешивается.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DIP_TAG = "dip"
SL_REASON = "sl"
# причины выхода, которые в v7 НЕ взводят cooldown (ранний выход добавит НОВЫЙ)
NON_SL_REASONS = ("signal", "legflip", "tp")
EXCLUDED_REASONS = ("forced_eod",)   # анти-каузальная цена выхода, вне резерва


def frozen_trade_reserve(trades: pd.DataFrame, open_arr: np.ndarray,
                         high_arr: np.ndarray | None = None,
                         commission: float = 0.001) -> pd.DataFrame:
    """Посделочный frozen-резерв по dip-сделкам.

    Parameters
    ----------
    trades : DataFrame с колонками entry_bar, exit_bar, entry_price,
        return_pct, pos_tag, exit_reason (как из ``label_exits.label_trades``).
    open_arr : массив Open по позиционному бар-индексу движка.
    high_arr : опциональный массив High для справочного потолка tail-capture.
    commission : комиссия движка (0.001).

    Returns
    -------
    DataFrame по dip-сделкам (кроме исключённых причин) с полями:
        entry_bar, exit_bar, exit_reason, ret_v7, best_fill_bar, decision_bar,
        ret_oracle, delta, ret_ideal_high, delta_ideal.
    """
    n_bars = len(open_arr)
    rows = []
    dip = trades[(trades["pos_tag"] == DIP_TAG)
                 & (~trades["exit_reason"].isin(EXCLUDED_REASONS))]
    for t in dip.itertuples():
        entry_bar = int(t.entry_bar)
        exit_bar = int(t.exit_bar)
        entry_price = float(t.entry_price)
        ret_v7 = float(t.return_pct)
        lo = entry_bar + 2
        hi = min(exit_bar - 1, n_bars - 1)          # включительно
        best_fill = -1
        ret_oracle = ret_v7
        ret_ideal = ret_v7
        if hi >= lo:
            fills = np.arange(lo, hi + 1)
            sell_open = open_arr[fills] * (1.0 - commission)
            rets = sell_open / entry_price - 1.0
            j = int(np.argmax(rets))
            if rets[j] > ret_v7:
                ret_oracle = float(rets[j])
                best_fill = int(fills[j])
            if high_arr is not None:
                sell_high = high_arr[fills] * (1.0 - commission)
                rh = sell_high / entry_price - 1.0
                ret_ideal = float(max(rh.max(), ret_v7))
        delta = ret_oracle - ret_v7
        rows.append({
            "entry_bar": entry_bar, "exit_bar": exit_bar,
            "exit_reason": t.exit_reason, "ret_v7": ret_v7,
            "best_fill_bar": best_fill,
            "decision_bar": best_fill - 1 if best_fill >= 0 else -1,
            "ret_oracle": ret_oracle, "delta": delta,
            "ret_ideal_high": ret_ideal, "delta_ideal": ret_ideal - ret_v7,
        })
    cols = ["entry_bar", "exit_bar", "exit_reason", "ret_v7", "best_fill_bar",
            "decision_bar", "ret_oracle", "delta", "ret_ideal_high",
            "delta_ideal"]
    return pd.DataFrame(rows, columns=cols)


@dataclass
class FrozenDecomposition:
    """Декомпозиция frozen-резерва (в п.п. на сделку, если поделено на n_dip)."""

    n_dip: int
    n_dip_sl: int
    reserve_total_pp: float             # Σ delta * 100 (сумма п.п.)
    by_reason_pp: dict = field(default_factory=dict)
    component_b_sum_pp: float = 0.0     # экономия на лузерах-SL, Σ п.п.
    component_b_per_dip: float = 0.0    # /n_dip  (знаменатель по умолчанию)
    component_b_per_sl: float = 0.0     # /n_dip_sl (альтернативный знаменатель)
    reserve_per_dip: float = 0.0


def decompose(reserve_df: pd.DataFrame, n_dip: int,
              n_dip_sl: int) -> FrozenDecomposition:
    """Разложить frozen-резерв по причине выхода базы v7. Всё в п.п. (×100)."""
    delta_pp = reserve_df["delta"].to_numpy() * 100.0
    reasons = reserve_df["exit_reason"].to_numpy()
    by_reason = {}
    for r in np.unique(reasons):
        by_reason[str(r)] = float(delta_pp[reasons == r].sum())
    reserve_total = float(delta_pp.sum())
    comp_b = float(delta_pp[reasons == SL_REASON].sum())
    return FrozenDecomposition(
        n_dip=n_dip, n_dip_sl=n_dip_sl,
        reserve_total_pp=reserve_total, by_reason_pp=by_reason,
        component_b_sum_pp=comp_b,
        component_b_per_dip=comp_b / n_dip if n_dip else 0.0,
        component_b_per_sl=comp_b / n_dip_sl if n_dip_sl else 0.0,
        reserve_per_dip=reserve_total / n_dip if n_dip else 0.0,
    )


def partition_check(reserve_df: pd.DataFrame,
                    decomp: FrozenDecomposition, tol: float = 1e-9) -> bool:
    """Тождество: сумма бакетных дельт == полный frozen-резерв (внутри frozen)."""
    return abs(sum(decomp.by_reason_pp.values())
               - decomp.reserve_total_pp) <= tol


def tail_metrics(reserve_df: pd.DataFrame, k_max: int = 3) -> dict:
    """Хвостовые метрики: leave-top-k-out, tail-capture, сохранность TP."""
    delta_pp = np.sort(reserve_df["delta"].to_numpy() * 100.0)[::-1]
    total = float(delta_pp.sum())
    leave = {}
    for k in range(1, k_max + 1):
        rest = float(delta_pp[k:].sum()) if len(delta_pp) > k else 0.0
        leave[f"leave_top_{k}"] = rest
        leave[f"leave_top_{k}_frac_kept"] = (rest / total) if total > 0 else 0.0
    ideal = float((reserve_df["delta_ideal"].to_numpy() * 100.0).sum())
    tail_capture = (total / ideal) if ideal > 0 else float("nan")
    tp = reserve_df[reserve_df["exit_reason"] == "tp"]
    tp_preserved = (float((tp["decision_bar"] < 0).mean())
                    if len(tp) else float("nan"))
    return {"leave_top_k": leave, "tail_capture_ratio": tail_capture,
            "n_tp": int(len(tp)), "tp_preserved_frac": tp_preserved,
            "reserve_total_pp": total, "ideal_high_pp": ideal}


def sl_bucket_fragility(reserve_df: pd.DataFrame, n_dip: int,
                        threshold_pct: float = 0.5) -> dict:
    """Устойчивость вердикта по компоненте (б) к удалению SL-сделок.

    Гейт: `component_b_sum / n_dip < threshold` -> мертва. В п.п. линия смерти
    = `threshold * n_dip`. Считаем, сколько SL-сделок можно выбросить, прежде
    чем сумма (б) упадёт ниже линии — это и есть хрупкость вердикта (эпоха A
    держится на единицах сделок).
    """
    sl = reserve_df[reserve_df["exit_reason"] == SL_REASON]
    deltas_pp = np.sort(sl["delta"].to_numpy() * 100.0)[::-1]   # по убыванию
    total = float(deltas_pp.sum())
    dead_line = threshold_pct * n_dip                            # п.п.
    # сколько крупнейших сделок надо выбросить, чтобы уйти под линию
    drop = 0
    running = total
    while drop < len(deltas_pp) and running >= dead_line:
        running -= deltas_pp[drop]
        drop += 1
    dead_after = drop if running < dead_line else None
    return {
        "n_sl": int(len(deltas_pp)),
        "sl_sum_pp": total,
        "dead_line_pp": dead_line,
        "margin_pp": total - dead_line,
        "verdict_base": "ALIVE" if total >= dead_line else "DEAD",
        "drops_to_dead": dead_after,       # None = устойчива ко всем удалениям
        "dead_after_dropping_one": bool(dead_after == 1),
    }


def new_cooldowns_from_oracle(reserve_df: pd.DataFrame) -> int:
    """Число НОВЫХ cooldown, которые вводит oracle = ранние выходы на
    dip-сделках, чья база-причина ≠ sl (там cooldown в v7 НЕ взводился)."""
    fired = reserve_df["decision_bar"] >= 0
    non_sl = reserve_df["exit_reason"].isin(NON_SL_REASONS)
    return int((fired & non_sl).sum())


def build_exit_policy(reserve_df: pd.DataFrame):
    """Callable(strategy, i) -> выйти ли сейчас; фиксирует бары решения oracle.

    Бары решения берутся из frozen-расчёта на границах v7 (абсолютные
    позиционные индексы). В полном движке границы могут сдвинуться из-за
    cooldown/CB — это и есть knock-on, он замеряется отдельно.
    """
    decision_bars = frozenset(int(b) for b in reserve_df["decision_bar"]
                              if b >= 0)

    def policy(strategy, i):
        """Oracle exit_policy: выйти на баре ``i``, если он в decision_bars."""
        return i in decision_bars

    policy.decision_bars = decision_bars     # для тестов/диагностики
    return policy
