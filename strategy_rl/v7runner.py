"""Прогон v7-зеркала с хуком (этап 0.3): данные, запуск, parity-проверка.

Загрузка данных и параметры — ИЗ КОДА SUNDAY (П3, никакой своей версии):
PARAMS импортируется из judge_labels_mirror (v7 = PARAMS с выключенным
bounce, как в их драйвере judge_labels_v7.py и from_sunday_reply_5).

Запуск (env sunday-base-213-tests):
  python strategy_rl/v7runner.py <state.parquet> \
      [--ref <judge_trades_v7.csv>] [--expert-out <log.parquet>] \
      [--trades-out <trades.csv>]

--ref включает parity-проверку: состав сделок бар-в-бар + допуски по
итоговым метрикам (зафиксированы ДО прогона: |ΔReturn| < 0.1 п.п.,
|ΔMaxDD| < 0.1 п.п., счёт сделок точный).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SUNDAY_ROOT = Path("/home/cubecloud/Python/projects/sunday")
if str(SUNDAY_ROOT) not in sys.path:
    sys.path.insert(0, str(SUNDAY_ROOT))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from backtesting import Backtest  # noqa: E402
from datawizard.causal_ta import rolling_mean  # noqa: E402
from tooling.audit.judge_labels_mirror import (PARAMS, CALIB_DIR, CALIB_KEY,  # noqa: E402
                                               BUY_UUID, SELL_UUID,
                                               ROLL_WARMUP_MIN)

from v7hook import AgentHookBT, EXPERT_LOG_COLS  # noqa: E402

# v7 = база v6-зеркала БЕЗ bounce (from_sunday_reply_5: остальное идентично)
P7 = dict(PARAMS, use_oracle_entry_filter=False)

# допуски parity (зафиксированы до прогона)
TOL_RETURN_PP = 0.1
TOL_MAXDD_PP = 0.1


def load_bt_df(state_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """parquet каузального STATE -> df движка (спецификация from_sunday_reply_5)."""
    st = pd.read_parquet(state_path)
    x = np.array([0.3, 0.5, 0.7])
    assert np.allclose(rolling_mean(x, 1), x, equal_nan=True), "rolling_mean(x,1) != x"
    df = pd.DataFrame(index=st.index)
    df["Open"] = st["open"].to_numpy()
    df["High"] = st["high"].to_numpy()
    df["Low"] = st["low"].to_numpy()
    df["Close"] = st["close"].to_numpy()
    df["Volume"] = st["volume"].to_numpy()
    df["raw_buy"] = st["q_buy"].to_numpy()      # уже roll33 -> roll_window=1
    df["raw_sell"] = st["q_sell"].to_numpy()    # уже roll59 -> roll_window=1
    df["regime_code"] = st["regime_code"].to_numpy().astype(int)
    df["leg_dn"] = st["leg_dn"].to_numpy().astype(bool)   # каузальный, НЕ пересчитываем
    df["tradeable"] = np.arange(len(df)) >= ROLL_WARMUP_MIN
    df = df.dropna(subset=["raw_buy", "raw_sell"])
    # коррекция критики №3: пока dropna ничего не выкидывает, позиционный
    # bar-индекс лога == строке STATE; если это сломается — джойны по bar
    # недействительны, нужно осознанное решение (джойн по времени)
    assert len(df) == len(st), \
        f"dropna удалил {len(st) - len(df)} строк — сверь джойны разметки!"
    return df, st


def load_calib() -> dict:
    tb = json.loads(open(f"{CALIB_DIR}/calib_{BUY_UUID}_{CALIB_KEY}.json").read())
    ts = json.loads(open(f"{CALIB_DIR}/calib_{SELL_UUID}_{CALIB_KEY}.json").read())
    return {"buy": tb["buy"], "sell": ts["sell"],
            "n_buy": tb["n_buy"], "n_sell": ts["n_sell"]}


def run_v7(state_path: str, exit_policy=None, collect_expert: bool = False):
    """-> (stats, trades_df, expert_df|None). exit_policy=None = чистая v7."""
    df, _ = load_bt_df(state_path)
    AgentHookBT.calib_table = load_calib()
    AgentHookBT.exit_policy = staticmethod(exit_policy) if exit_policy else None
    log: list | None = [] if collect_expert else None
    AgentHookBT.expert_log = log
    try:
        bt = Backtest(df, AgentHookBT, cash=10_000_000, commission=0.001,
                      trade_on_close=False, exclusive_orders=True)
        stats = bt.run(**P7)
    finally:
        AgentHookBT.exit_policy = None
        AgentHookBT.expert_log = None

    idx = df.index
    tr = stats._trades.copy().sort_values("EntryBar")
    trades = pd.DataFrame({
        "signal_time": idx[(tr["EntryBar"].astype(int) - 1).to_numpy()],
        "entry_time": idx[tr["EntryBar"].astype(int).to_numpy()],
        "exit_time": idx[np.minimum(tr["ExitBar"].astype(int).to_numpy(), len(idx) - 1)],
        "return_pct": tr["ReturnPct"].astype(float).to_numpy(),
        "duration_min": (tr["ExitBar"].astype(int) - tr["EntryBar"].astype(int)).to_numpy(),
        "entry_bar": tr["EntryBar"].astype(int).to_numpy(),
        "exit_bar": tr["ExitBar"].astype(int).to_numpy(),
        "entry_price": tr["EntryPrice"].astype(float).to_numpy(),
        "exit_price": tr["ExitPrice"].astype(float).to_numpy(),
        "size": tr["Size"].astype(float).to_numpy(),
    })
    expert = None
    if collect_expert:
        expert = pd.DataFrame(log, columns=EXPERT_LOG_COLS)
        expert["time"] = idx[expert["bar"].to_numpy()]
        # фиксация покрытия лога (коррекция критики №3): движок стартует
        # с бара 1, лог обязан покрыть все бары без дыр
        assert int(expert["bar"].min()) == 1 and len(expert) == len(df) - 1, \
            f"expert log coverage broken: {expert['bar'].min()}..{len(expert)} vs {len(df) - 1}"
    return stats, trades, expert


def parity_check(trades: pd.DataFrame, stats, ref_csv: str) -> list[str]:
    """Сравнение с эталоном sunday. Возвращает список расхождений (пусто = PASS)."""
    ref = pd.read_csv(ref_csv, parse_dates=["signal_time", "entry_time", "exit_time"])
    problems: list[str] = []
    if len(trades) != len(ref):
        problems.append(f"trades count: ours {len(trades)} != ref {len(ref)}")
    n = min(len(trades), len(ref))
    for col in ("signal_time", "entry_time", "exit_time"):
        ours = pd.to_datetime(trades[col].to_numpy()[:n])
        theirs = pd.to_datetime(ref[col].to_numpy()[:n])
        bad = int((ours != theirs).sum())
        if bad:
            first = int(np.argmax((ours != theirs).to_numpy()))
            problems.append(f"{col}: {bad}/{n} mismatch; first at #{first}: "
                            f"ours {ours[first]} vs ref {theirs[first]}")
    dret = float(np.abs(trades["return_pct"].to_numpy()[:n]
                        - ref["return_pct"].to_numpy()[:n]).max()) if n else 0.0
    if dret > 1e-9:
        problems.append(f"return_pct: max |diff| {dret:.2e}")
    for col in ("duration_min", "size"):
        if col in ref.columns:
            d = float(np.abs(trades[col].to_numpy()[:n]
                             - ref[col].to_numpy()[:n]).max()) if n else 0.0
            if d > 1e-6:
                problems.append(f"{col}: max |diff| {d:.2e}")
    # Движковые метрики (Return/MaxDD/WinRate) сверяются с ЧИСЛАМИ, которые
    # sunday получил своим прогоном (--expect "162,267.5,-30.1,53.7").
    # Наивный компаунд посделочных return_pct движок НЕ воспроизводит
    # (сайзинг 0.9999, целочисленные лоты) — таким сравнением не проверяем.
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--ref", default="")
    ap.add_argument("--expect", default="",
                    help="ожидание движковых метрик: 'n,ret,maxdd,winrate' от sunday")
    ap.add_argument("--expert-out", default="")
    ap.add_argument("--trades-out", default="")
    args = ap.parse_args()

    stats, trades, expert = run_v7(os.path.expanduser(args.state),
                                   collect_expert=bool(args.expert_out))
    print(f"trades: {len(trades)}  Return: {stats['Return [%]']:.1f}%  "
          f"MaxDD: {stats['Max. Drawdown [%]']:.1f}%  "
          f"WinRate: {stats['Win Rate [%]']:.1f}%")
    if args.trades_out:
        trades.to_csv(os.path.expanduser(args.trades_out), index=False)
    if args.expert_out and expert is not None:
        expert.to_parquet(os.path.expanduser(args.expert_out))
        acts = expert[["mech_exit", "agent_exit", "entry_placed"]].sum()
        print(f"expert log: {len(expert):,} bars; entries {int(acts.entry_placed)}, "
              f"mech exits {int(acts.mech_exit)}")
    problems: list[str] = []
    if args.ref:
        problems += parity_check(trades, stats, os.path.expanduser(args.ref))
    if args.expect:
        n_e, ret_e, dd_e, wr_e = (float(x) for x in args.expect.split(","))
        checks = [("trades", len(trades), n_e, 0.5),
                  ("Return", float(stats["Return [%]"]), ret_e, TOL_RETURN_PP),
                  ("MaxDD", float(stats["Max. Drawdown [%]"]), dd_e, TOL_MAXDD_PP),
                  ("WinRate", float(stats["Win Rate [%]"]), wr_e, 0.1)]
        problems += [f"{name}: ours {ours:.2f} vs expected {exp:.2f} (tol {tol})"
                     for name, ours, exp, tol in checks if abs(ours - exp) > tol]
    if args.ref or args.expect:
        if problems:
            print("PARITY FAIL:")
            for p in problems:
                print("  -", p)
            sys.exit(1)
        print("PARITY PASS")


if __name__ == "__main__":
    main()
