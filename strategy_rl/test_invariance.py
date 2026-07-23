"""Invariance-тест хука (коррекция критики 0.3, п.8).

Политика «всегда держи» (always-False) обязана дать БИТ-В-БИТ тот же
результат, что и выключенный хук: сам факт вызова политики не должен
влиять ни на одну сделку. Ловит случайные побочные эффекты в хуке.

Run (env sunday-base-213-tests):
  python strategy_rl/test_invariance.py <state.parquet>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from v7runner import run_v7  # noqa: E402


def main():
    state = os.path.expanduser(sys.argv[1])
    _, base, _ = run_v7(state, exit_policy=None)
    _, held, _ = run_v7(state, exit_policy=lambda strategy, i: False)
    assert len(base) == len(held), f"trades: {len(base)} vs {len(held)}"
    for col in ("signal_time", "entry_time", "exit_time"):
        assert (base[col].to_numpy() == held[col].to_numpy()).all(), f"{col} differs"
    assert np.allclose(base["return_pct"], held["return_pct"], atol=0), "returns differ"
    assert np.allclose(base["size"], held["size"], atol=0), "sizes differ"
    print(f"INVARIANCE PASS: {len(base)} сделок бит-в-бит при always-hold политике")


if __name__ == "__main__":
    main()
