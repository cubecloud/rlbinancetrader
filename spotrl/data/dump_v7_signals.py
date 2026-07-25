"""Выгрузка побарных булевых сигналов v7 в артефакт (шаг 1 плана obs v2).

Зачем: `state_v0`/`state_v3` содержат непрерывные составляющие входа/выхода
(buy_margin, sell_margin, bounce_pct, leg_age_h) и каузальный `leg_dn`, но НЕ
булевы сигналы `entry_signal` / `trans_entry_signal` / `exit_sig`. Без них среда
не восстановит `pos_tag` (dip vs transition) на баре входа и не отделит штатный
сигнальный выход. Эти массивы уже посчитаны движком v7 (объект стратегии
`_entry` / `_trans_entry` / `_exit`), поэтому не пересчитываются заново (П3):
прогоняется эталон v7 через мост и читаются его массивы.

`leg_dn` берётся из КАУЗАЛЬНОГО state (передан в движок как есть, не
пересчитывается): требование train/serve-паритета — тот же источник leg_dn, что
видел эксперт (иначе селектор смысла действия разъедется, находка по гейту
выхода 2024).

Запуск (env sunday-base-213-tests, где есть backtesting):
  python -m spotrl.data.dump_v7_signals <state.parquet> --out <signals.parquet>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Каталог с уже проверенным мостом v7 (strategy_rl не изменяется этим пакетом).
_LEGACY = Path(__file__).resolve().parents[2] / "strategy_rl"

# Имена булевых колонок артефакта сигналов (стабильный контракт со средой).
SIGNAL_COLUMNS = ("entry_signal", "trans_entry_signal", "exit_sig", "leg_dn")


def dump_signals(state_path: str) -> pd.DataFrame:
    """Прогнать v7 на каузальном state и вернуть побарные булевы сигналы.

    Args:
        state_path: путь к parquet каузального state (тот же, что кормится v7).

    Returns:
        DataFrame c индексом времени state и колонками SIGNAL_COLUMNS
        (bool): `entry_signal`=self._entry, `trans_entry_signal`=self._trans_entry,
        `exit_sig`=self._exit, `leg_dn`=self._leg_dn (каузальный, из state).
    """
    if str(_LEGACY) not in sys.path:
        sys.path.insert(0, str(_LEGACY))
    from v7runner import run_v7  # noqa: E402  (ленивый импорт: требует sunday)

    stats, _trades, _expert = run_v7(state_path, collect_expert=False)
    strat = stats._strategy
    entry = np.asarray(strat._entry, dtype=bool)
    trans = np.asarray(strat._trans_entry, dtype=bool)
    exit_sig = np.asarray(strat._exit, dtype=bool)
    leg_dn = np.asarray(strat._leg_dn, dtype=bool)
    index = pd.read_parquet(state_path).index
    n = len(index)
    if not (len(entry) == len(trans) == len(exit_sig) == len(leg_dn) == n):
        raise ValueError(
            f"длины сигналов не совпадают с state: entry={len(entry)}, "
            f"trans={len(trans)}, exit={len(exit_sig)}, leg_dn={len(leg_dn)}, n={n}")
    return pd.DataFrame(
        {"entry_signal": entry, "trans_entry_signal": trans,
         "exit_sig": exit_sig, "leg_dn": leg_dn}, index=index)


def main() -> None:
    """CLI: прогнать v7 и сохранить артефакт сигналов рядом со state."""
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    frame = dump_signals(os.path.expanduser(args.state))
    frame.to_parquet(os.path.expanduser(args.out))
    counts = {c: int(frame[c].sum()) for c in SIGNAL_COLUMNS}
    print(f"signals: {len(frame):,} bars; True counts {counts}")


if __name__ == "__main__":
    main()
