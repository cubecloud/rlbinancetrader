"""Мост к движку sunday: запуск v7 и parity-проверка (гейт Э0.1).

Что здесь: тонкая обёртка над уже проверенным `strategy_rl/v7runner.py`.
Своей копии механики v7 в пакете НЕТ и быть не должно (PLAN П3): единственный
источник истины — движок sunday, вызванный через этот мост.

Чего здесь НЕТ: ничего про обучение, наблюдение и награду.

Зависимость от sunday подключается ЛЕНИВО (внутри функции), чтобы импорт
пакета не требовал установленного окружения sunday.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

# Каталог со старым, уже проверенным мостом (не изменяется этим пакетом)
LEGACY_BRIDGE_DIR = Path(__file__).resolve().parents[2] / "strategy_rl"


@dataclass(frozen=True)
class ParityResult:
    """Результат сверки с эталонной книгой сделок sunday.

    Attributes:
        n_trades: число сделок нашего прогона.
        n_ref: число сделок эталона.
        max_return_diff_pp: максимальное расхождение return_pct, п.п.
        problems: список расхождений; пустой список = PASS.
    """

    n_trades: int
    n_ref: int
    max_return_diff_pp: float
    problems: List[str]

    @property
    def passed(self) -> bool:
        """Гейт пройден, если расхождений нет."""
        return not self.problems


def _legacy_module(name: str):
    """Импортировать модуль из strategy_rl, добавив каталог в sys.path."""
    if str(LEGACY_BRIDGE_DIR) not in sys.path:
        sys.path.insert(0, str(LEGACY_BRIDGE_DIR))
    return __import__(name)


def run_v7(state_path: str, exit_policy: Optional[Callable] = None,
           collect_expert: bool = False):
    """Прогнать эталонную стратегию v7 на каузальном state.

    Args:
        state_path: путь к parquet каузального state.
        exit_policy: политика раннего выхода `(strategy, bar) -> bool`.
        collect_expert: собирать ли побарный лог эксперта.

    Returns:
        Кортеж (stats, trades_df, expert_df|None) движка backtesting.
    """
    runner = _legacy_module("v7runner")
    return runner.run_v7(state_path, exit_policy=exit_policy,
                         collect_expert=collect_expert)


def check_parity(trades: pd.DataFrame, ref_csv: str) -> ParityResult:
    """Сверить книгу сделок с эталоном sunday (Т1).

    Гейт Э0.1: совпадение состава сделок по (entry_bar, exit_bar) и
    |ΔReturn| <= 0.1 п.п. (TOL_RETURN_PP из v7runner — допуск сайзинга).
    """
    runner = _legacy_module("v7runner")
    ref = pd.read_csv(ref_csv, parse_dates=["signal_time", "entry_time", "exit_time"])
    problems = runner.parity_check(trades, None, ref_csv)
    n = min(len(trades), len(ref))
    diff = float((trades["return_pct"].to_numpy()[:n]
                  - ref["return_pct"].to_numpy()[:n]).__abs__().max()) if n else 0.0
    return ParityResult(n_trades=len(trades), n_ref=len(ref),
                        max_return_diff_pp=diff, problems=list(problems))
