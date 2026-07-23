"""Пост-разметка причин выходов + поминутные BC-метки (коррекции критики 0.3).

Что делает:
1. Прогоняет чистую v7 (хук выключен) по окну STATE.
2. Каждой сделке проставляет exit_reason: tp / legflip / signal / b2b /
   sl / forced_eod (приоритет как в родительском next(): TP → legflip →
   сигнальный; закрытие без метки решения = SL; выход на последнем баре
   без метки = принудительное закрытие конца данных).
3. forced_eod-сделке ставит bc_weight=0 (её цена выхода анти-каузальна —
   движок закрывает по open того же бара; в обучение не идёт).
4. Строит поминутные BC-метки действий: wait / enter / hold / exit
   (индексы из strategy_rl.actionspec). Бары SL-закрытий остаются hold —
   SL исполняет среда, это не решение политики.
5. Проверки: сумма причин == числу сделок; ожидания по окну (--expect-sl,
   --expect-forced) — из независимого источника (критика: 2024-26 = 11 SL
   + 1 forced; 2021-24 = 48 SL).

Run (env sunday-base-213-tests):
  python strategy_rl/label_exits.py <state.parquet> \
      --trades-out <trades_labeled.csv> --bc-out <bc_labels.parquet> \
      [--expect-sl N] [--expect-forced N]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from actionspec import WAIT, ENTER, HOLD, EXIT  # noqa: E402
from v7runner import run_v7, load_bt_df, P7  # noqa: E402


def label_trades(state_path: str):
    stats, trades, expert = run_v7(state_path, collect_expert=True)
    df, st = load_bt_df(state_path)
    n_bars = len(df)
    leg_dn = df["leg_dn"].to_numpy()
    close = df["Close"].to_numpy(float)
    rc = df["regime_code"].to_numpy(int)

    log = expert.set_index("bar", drop=False)
    reasons = []
    for t in trades.itertuples():
        d = int(t.exit_bar) - 1                     # бар решения о выходе
        sig_bar = int(t.entry_bar) - 1              # бар решения о входе
        tag = ""
        for b in range(int(t.entry_bar), min(int(t.exit_bar) + 1, n_bars)):
            row_tag = log.at[b, "pos_tag"] if b in log.index else ""
            if row_tag:
                tag = row_tag
                break
        tag = tag or "dip"
        mech = bool(log.at[d, "mech_exit"]) if d in log.index else False
        agent = bool(log.at[d, "agent_exit"]) if d in log.index else False
        exit_sig = bool(log.at[d, "exit_sig"]) if d in log.index else False

        if agent:
            reason = "agent"
        elif mech:
            tp_pct = P7["transition_tp"] if tag == "transition" else P7["take_profit_pct"]
            entered_on_up = not bool(leg_dn[sig_bar])
            if tp_pct > 0 and close[d] >= float(t.entry_price) * (1 + tp_pct):
                reason = "tp"
            elif tag == "dip" and P7["use_legflip_exit"] and entered_on_up and leg_dn[d]:
                reason = "legflip"
            elif tag == "transition" and exit_sig and rc[d] != 2:
                reason = "b2b"
            elif exit_sig:
                reason = "signal"
            else:
                reason = "mech_unresolved"          # не должно случаться
        elif int(t.exit_bar) >= n_bars - 1:
            reason = "forced_eod"
        else:
            reason = "sl"
        reasons.append((reason, tag))

    trades = trades.copy()
    trades["exit_reason"] = [r for r, _ in reasons]
    trades["pos_tag"] = [g for _, g in reasons]
    trades["bc_weight"] = np.where(trades["exit_reason"] == "forced_eod", 0.0, 1.0)

    # поминутные BC-метки действий
    bc = expert[["bar", "time", "in_pos_before", "pos_tag"]].copy()
    act = np.full(len(bc), WAIT, dtype=np.int8)
    act[bc["in_pos_before"].to_numpy()] = HOLD
    act[expert["entry_placed"].to_numpy()] = ENTER
    act[(expert["mech_exit"] | expert["agent_exit"]).to_numpy()] = EXIT
    bc["bc_action"] = act
    # причина выхода на выходных барах (для анализа; SL-бары остаются hold)
    reason_by_bar = {int(t.exit_bar) - 1: r
                     for t, (r, _) in zip(trades.itertuples(), reasons)
                     if r not in ("sl", "forced_eod")}
    bc["exit_reason"] = bc["bar"].map(reason_by_bar).fillna("")
    return stats, trades, bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--trades-out", required=True)
    ap.add_argument("--bc-out", required=True)
    ap.add_argument("--expect-sl", type=int, default=-1)
    ap.add_argument("--expect-forced", type=int, default=-1)
    args = ap.parse_args()

    stats, trades, bc = label_trades(os.path.expanduser(args.state))
    dist = trades["exit_reason"].value_counts().to_dict()
    print(f"trades: {len(trades)} | причины выходов: {dist}")
    assert sum(dist.values()) == len(trades), "сумма причин != числу сделок"
    assert "mech_unresolved" not in dist, f"неразобранные mech-выходы: {dist}"
    n_enter = int((bc['bc_action'] == ENTER).sum())
    n_exit = int((bc['bc_action'] == EXIT).sum())
    print(f"BC-метки: enter {n_enter}, exit {n_exit}, "
          f"hold {int((bc['bc_action'] == HOLD).sum()):,}, "
          f"wait {int((bc['bc_action'] == WAIT).sum()):,}")
    assert n_enter == len(trades), "enter-баров != числу сделок"
    n_decision_exits = int((trades["exit_reason"].isin(
        ("tp", "legflip", "signal", "b2b", "agent"))).sum())
    assert n_exit == n_decision_exits, \
        f"exit-баров {n_exit} != решённых выходов {n_decision_exits}"
    if args.expect_sl >= 0:
        assert dist.get("sl", 0) == args.expect_sl, \
            f"SL: {dist.get('sl', 0)} != ожидания {args.expect_sl}"
    if args.expect_forced >= 0:
        assert dist.get("forced_eod", 0) == args.expect_forced, \
            f"forced_eod: {dist.get('forced_eod', 0)} != {args.expect_forced}"
    trades.to_csv(os.path.expanduser(args.trades_out), index=False)
    bc.to_parquet(os.path.expanduser(args.bc_out))
    print("LABELS OK")


if __name__ == "__main__":
    main()
