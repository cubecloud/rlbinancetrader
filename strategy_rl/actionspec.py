"""Полное пространство действий агента + маски свободы (решение 23.07).

Идея: сеть с первого дня имеет выходы на ВСЕ будущие решения; что агенту
пока нельзя — запрещается маской (MaskablePPO), а не вырезается из сети.
Разблокировка свободы = изменение FreedomConfig (одна строка конфига),
веса и архитектура не трогаются. Каждая разблокировка — отдельное pre-reg
решение (см. PLAN.md, беклог).

Соответствие старому binanceenv.discrete_4 (Buy/Sell/Hold/Wait) намеренно
сохранено по смыслу: ENTER~Buy, EXIT~Sell, HOLD~Hold, WAIT~Wait.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# индексы действий — СТАБИЛЬНЫ НАВСЕГДА (менять нельзя: сломает веса клона)
WAIT, ENTER, HOLD, EXIT = 0, 1, 2, 3
N_ACTIONS = 4
ACTION_NAMES = ("wait", "enter", "hold", "exit")


@dataclass(frozen=True)
class FreedomConfig:
    """Какие решения агенту разрешено принимать САМОМУ (иначе — как v7).

    Стартовая конфигурация RL-этапа (утверждено 23.07):
    только exit_dip=True; всё прочее делает механика v7.
    На этапе BC-копирования свобода не нужна вовсе — клон учится повторять
    v7 по полной разметке (все 4 действия), маски применяются позже.
    """
    exit_dip: bool = True          # ранний выход из dip-сделки
    exit_transition: bool = False  # выход из transition (заперт, беклог)
    entry_veto: bool = False       # право пропустить вход v7 (заперт)
    entry_own: bool = False        # право входить самому (заперт)


def action_mask(in_position: bool, pos_tag: str, freedom: FreedomConfig) -> np.ndarray:
    """Маска допустимых действий для текущего состояния.

    Семантика: WAIT/HOLD («делай как v7») допустимы всегда в своём
    состоянии; активные действия открываются свободами.
    """
    m = np.zeros(N_ACTIONS, dtype=bool)
    if in_position:
        m[HOLD] = True
        if pos_tag == "dip":
            m[EXIT] = freedom.exit_dip
        elif pos_tag == "transition":
            m[EXIT] = freedom.exit_transition
    else:
        m[WAIT] = True
        # ENTER: при entry_veto агент выбирает WAIT (=пропустить) на баре
        # входа v7; при entry_own — может инициировать вход сам.
        m[ENTER] = freedom.entry_veto or freedom.entry_own
    return m
