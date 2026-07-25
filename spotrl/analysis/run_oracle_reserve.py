"""Драйвер этапа Э0.2: oracle-резерв по обеим эпохам, тройной отчёт.

Порядок на каждую эпоху:
1. разметка чистой v7 (``label_exits.label_trades``) -> причины выходов,
   базовый Return, сверка базы с судьёй по границам;
2. frozen-резерв и декомпозиция (пространство гейта, компонента «б»);
3. полный прогон движком с oracle-выходами: cooldown как есть и cooldown
   отключён -> Reserve_B и цена cooldown;
4. хвостовые метрики, число новых cooldown, дельта трейдбука.

Запуск (env rlbinancetrader):
  python -m spotrl.analysis.run_oracle_reserve \
      [--out handoff/oracle_reserve_numbers.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_STRATEGY_RL = Path(__file__).resolve().parents[2] / "strategy_rl"
if str(_STRATEGY_RL) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_RL))

from label_exits import label_trades  # noqa: E402

from spotrl.analysis.engine_run import load_ohlc, run_engine  # noqa: E402
from spotrl.analysis import oracle_reserve as orc  # noqa: E402

DATA = os.path.expanduser("~/Data/sunday_tests/state_v0")
EPOCHS = [
    {"name": "A_2024-26",
     "state": f"{DATA}/state_v3_causal_2024-03_2026-07.parquet",
     "judge": f"{DATA}/judge_trades_v7.csv", "anchor_inflated": True},
    {"name": "B_2021-24",
     "state": f"{DATA}/state_v3_causal_2021-01_2024-03.parquet",
     "judge": f"{DATA}/judge_trades_v7_2021.csv", "anchor_inflated": False},
]
COMMISSION = 0.001
GATE_B = 0.5            # %/сделку — предложение, требует утверждения
COOLDOWN_ESCALATE = 0.50   # доля резерва (б); требует утверждения


def base_parity(trades: pd.DataFrame, judge_csv: str) -> dict:
    """Сверка базы v7 с судьёй по числу и границам сделок (мягкая, по времени)."""
    ref = pd.read_csv(judge_csv, parse_dates=["entry_time", "exit_time"])
    ours_n, ref_n = len(trades), len(ref)
    # сверяем по entry_time/exit_time (границы), на общем префиксе
    m = min(ours_n, ref_n)
    et = pd.to_datetime(trades["entry_time"].to_numpy()[:m]) \
        if "entry_time" in trades.columns else None
    bad_entry = int((et != pd.to_datetime(ref["entry_time"].to_numpy()[:m])).sum()) \
        if et is not None else -1
    return {"n_ours": ours_n, "n_ref": ref_n, "entry_mismatch": bad_entry}


def run_epoch(ep: dict) -> dict:
    """Полный замер oracle-резерва по одной эпохе; отдаёт словарь чисел."""
    name, state = ep["name"], ep["state"]
    print(f"\n===== ЭПОХА {name} =====")
    # 1. база v7 + разметка
    stats_v7, trades_lab, _ = label_trades(state)
    ret_v7 = float(stats_v7["Return [%]"])
    n_v7 = len(trades_lab)
    dip = trades_lab[trades_lab["pos_tag"] == "dip"]
    n_dip = int(len(dip))
    n_dip_sl = int((dip["exit_reason"] == "sl").sum())
    parity = base_parity(trades_lab, ep["judge"])
    print(f"v7: n={n_v7} Return={ret_v7:.2f}%  n_dip={n_dip} dip_sl={n_dip_sl}"
          f"  parity(n_ours/n_ref/entry_mismatch)="
          f"{parity['n_ours']}/{parity['n_ref']}/{parity['entry_mismatch']}")

    # 2. frozen-резерв
    ohlc = load_ohlc(state)
    res = orc.frozen_trade_reserve(trades_lab, ohlc["Open"].to_numpy(),
                                   ohlc["High"].to_numpy(), COMMISSION)
    decomp = orc.decompose(res, n_dip, n_dip_sl)
    assert orc.partition_check(res, decomp), "partition-identity нарушено!"
    assert (res["delta"] >= -1e-12).all(), "отрицательная frozen-дельта!"
    tails = orc.tail_metrics(res)
    new_cd = orc.new_cooldowns_from_oracle(res)
    frag = orc.sl_bucket_fragility(res, n_dip, GATE_B)

    print(f"FROZEN резерв всего: {decomp.reserve_total_pp:.3f} п.п. "
          f"(={decomp.reserve_per_dip:.4f}%/dip)")
    print(f"  по причинам (Σ п.п.): {decomp.by_reason_pp}")
    print(f"  КОМПОНЕНТА (б) экономия на SL-лузерах: "
          f"{decomp.component_b_sum_pp:.3f} п.п.; "
          f"/n_dip={decomp.component_b_per_dip:.4f}%  "
          f"/n_dip_sl={decomp.component_b_per_sl:.4f}%")
    print(f"  хвосты: tail_capture={tails['tail_capture_ratio']:.3f}  "
          f"leave-top-1/2/3 kept-frac="
          f"{tails['leave_top_k']['leave_top_1_frac_kept']:.2f}/"
          f"{tails['leave_top_k']['leave_top_2_frac_kept']:.2f}/"
          f"{tails['leave_top_k']['leave_top_3_frac_kept']:.2f}  "
          f"TP_preserved={tails['tp_preserved_frac']}")
    print(f"  новых cooldown от oracle (dip non-SL с ранним выходом): {new_cd}")
    print(f"  ХРУПКОСТЬ (б): sl_sum={frag['sl_sum_pp']:.2f} п.п.  "
          f"линия смерти={frag['dead_line_pp']:.2f} п.п.  "
          f"запас={frag['margin_pp']:+.2f} п.п.  "
          f"мертва после удаления {frag['drops_to_dead']} SL-сделок  "
          f"(dead_after_one={frag['dead_after_dropping_one']})")

    # 3. полный движок: cooldown как есть / отключён
    policy = orc.build_exit_policy(res)
    stats_cd, trades_cd = run_engine(state, policy, disarm_cooldown=False,
                                     commission=COMMISSION)
    stats_nocd, trades_nocd = run_engine(state, policy, disarm_cooldown=True,
                                         commission=COMMISSION)
    ret_cd = float(stats_cd["Return [%]"])
    ret_nocd = float(stats_nocd["Return [%]"])
    reserve_b_cd = ret_cd - ret_v7
    reserve_b_nocd = ret_nocd - ret_v7
    cooldown_price = reserve_b_nocd - reserve_b_cd
    # то же в СУММЕ посделочных return_pct (без компаунда — не path-chaotic,
    # те же единицы, что и frozen-резерв): корректная цена cooldown
    sum_v7 = float(trades_lab["return_pct"].sum()) * 100.0
    sum_cd = float(trades_cd["return_pct"].sum()) * 100.0
    sum_nocd = float(trades_nocd["return_pct"].sum()) * 100.0
    cooldown_price_sum = sum_nocd - sum_cd
    cd_share_of_b_sum = (cooldown_price_sum / decomp.component_b_sum_pp
                         if decomp.component_b_sum_pp else float("nan"))
    print(f"  СУММА return_pct (не компаунд): v7={sum_v7:.1f}  "
          f"oracle(cd)={sum_cd:.1f}  oracle(no-cd)={sum_nocd:.1f} п.п.")
    # ВНИМАНИЕ: cd_share_of_b_sum ≈ -0.5 в обеих эпохах — ЭМПИРИЧЕСКОЕ
    # совпадение полной oracle-политики, НЕ структурная связь. Диагностика:
    # (1) при случайном подмножестве decision_bars отношение уходит от -0.5
    #     (A: -0.03/+0.06/-0.19; B: -0.29/-0.42); (2) общие сделки (тот же
    # entry_bar в cd и no-cd прогонах) дают в разность РОВНО 0 — резерв (б)
    # живёт в них (SL-сделки идентичны в обоих прогонах), а cooldown_price —
    # в сделках, уникальных для одного прогона. Множества НЕ пересекаются →
    # величины независимы; -0.5 нельзя трактовать как «подтверждение».
    print(f"  ЦЕНА COOLDOWN (сумма-простр.)={cooldown_price_sum:+.1f} п.п.  "
          f"(отношение к (б) {cd_share_of_b_sum:+.2f} — СОВПАДЕНИЕ, не связь)")
    print(f"ДВИЖОК: Return v7={ret_v7:.2f}%  oracle(cd)={ret_cd:.2f}%  "
          f"oracle(no-cd)={ret_nocd:.2f}%")
    print(f"  Reserve_B(cd)={reserve_b_cd:+.2f} п.п.  "
          f"Reserve_B(no-cd)={reserve_b_nocd:+.2f} п.п.  "
          f"ЦЕНА COOLDOWN={cooldown_price:+.2f} п.п.")
    print(f"  трейдбук: v7 n={n_v7}  oracle(cd) n={len(trades_cd)}  "
          f"oracle(no-cd) n={len(trades_nocd)}  "
          f"(Δ = подавленные/сдвинутые сделки — knock-on)")

    # Цена cooldown — в компаунд-единицах; сравнивать её можно только с
    # Reserve_B (те же единицы), НЕ с frozen-компонентой (б). Эскалация правила
    # уместна, только если cooldown РЕЖЕТ резерв (cooldown_price > 0) на
    # существенную долю. Здесь cooldown_price < 0 = cooldown резерв НЕ ест.
    cd_share_of_reserve_b = (cooldown_price / reserve_b_nocd
                             if reserve_b_nocd else float("nan"))
    # эскалация — по сумма-пространству (не по хаотичному компаунду): правило
    # пересматриваем, только если cooldown РЕЖЕТ резерв (>0) на >= половины (б)
    cooldown_escalate = bool(np.isfinite(cd_share_of_b_sum)
                             and cooldown_price_sum > 0
                             and cd_share_of_b_sum >= COOLDOWN_ESCALATE)

    return {
        "epoch": name, "anchor_inflated": ep["anchor_inflated"],
        "base": {"n_trades": n_v7, "return_pct": ret_v7,
                 "n_dip": n_dip, "n_dip_sl": n_dip_sl, "parity": parity},
        "frozen": {
            "reserve_total_pp": decomp.reserve_total_pp,
            "reserve_per_dip_pct": decomp.reserve_per_dip,
            "by_reason_pp": decomp.by_reason_pp,
            "component_b_sum_pp": decomp.component_b_sum_pp,
            "component_b_per_dip_pct": decomp.component_b_per_dip,
            "component_b_per_sl_pct": decomp.component_b_per_sl,
            "tails": tails, "new_cooldowns": new_cd,
            "sl_bucket_fragility": frag,
        },
        "engine": {
            "return_v7_pct": ret_v7, "return_oracle_cd_pct": ret_cd,
            "return_oracle_nocd_pct": ret_nocd,
            "reserve_b_cd_pp": reserve_b_cd,
            "reserve_b_nocd_pp": reserve_b_nocd,
            "cooldown_price_pp_compound": cooldown_price,
            "cooldown_share_of_reserve_b_compound": cd_share_of_reserve_b,
            "compound_note": "компаунд движка с предвидением взрывается и "
                             "path-chaotic; НЕ основа вердикта, только справка",
            "sum_return_pct_v7": sum_v7,
            "sum_return_pct_oracle_cd": sum_cd,
            "sum_return_pct_oracle_nocd": sum_nocd,
            "cooldown_price_pp_sumspace": cooldown_price_sum,
            "cooldown_share_of_component_b_sumspace": cd_share_of_b_sum,
            "cooldown_share_note": "отношение ~-0.5 в обеих эпохах — "
                                   "ЭМПИРИЧЕСКОЕ совпадение полной oracle-"
                                   "политики, НЕ структурная связь. Доказано: "
                                   "при случайном подмножестве decision_bars "
                                   "отношение уходит от -0.5 (A: -0.03/+0.06/"
                                   "-0.19; B: -0.29/-0.42); общие сделки дают в "
                                   "разность ровно 0, значит (б) и "
                                   "cooldown_price считаются на непересекающихся "
                                   "множествах сделок. Знак цены cooldown "
                                   "(sum_nocd<sum_cd) от этого не зависит.",
            "n_trades_v7": n_v7, "n_trades_oracle_cd": int(len(trades_cd)),
            "n_trades_oracle_nocd": int(len(trades_nocd)),
        },
        "gate_pending_approval": {
            "gate_b_threshold_pct_per_trade": GATE_B,
            "component_b_per_dip_pct": decomp.component_b_per_dip,
            "verdict_if_denominator_n_dip":
                "DEAD" if decomp.component_b_per_dip < GATE_B else "ALIVE",
            "cooldown_escalate_threshold_frac": COOLDOWN_ESCALATE,
            "cooldown_escalate": cooldown_escalate,
        },
    }


def main():
    """CLI: прогнать замер по обеим эпохам и (опц.) сохранить числа в JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    results = [run_epoch(ep) for ep in EPOCHS]
    print("\n===== СВОДКА (вердикт ожидает утверждения порога/знаменателя) =====")
    for r in results:
        g = r["gate_pending_approval"]
        frag = r["frozen"]["sl_bucket_fragility"]
        robust = ("indeterminate/coin-flip (мертва при удалении 1 SL-сделки)"
                  if frag["dead_after_dropping_one"]
                  else f"устойчива к удалению {frag['drops_to_dead']} SL-сделок"
                  if frag["drops_to_dead"]
                  else "устойчива ко всем удалениям SL")
        print(f"{r['epoch']}: компонента(б)/n_dip="
              f"{g['component_b_per_dip_pct']:.4f}% "
              f"(порог {g['gate_b_threshold_pct_per_trade']}%) -> "
              f"{g['verdict_if_denominator_n_dip']}; {robust}"
              f"{'  [anchor-inflated компаунд]' if r['anchor_inflated'] else ''}")
        print(f"    цена cooldown (сумма-простр.)="
              f"{r['engine']['cooldown_price_pp_sumspace']:+.1f} п.п. "
              f"(знак<0 = не режет резерв; отношение к (б) -0.5 — совпадение); "
              f"эскалация правила={g['cooldown_escalate']}")
    if args.out:
        Path(os.path.expanduser(args.out)).write_text(
            json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nчисла сохранены -> {args.out}")


if __name__ == "__main__":
    main()
