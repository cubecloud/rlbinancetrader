"""AgentHookBT — наследник зеркала v7 с хуком для RL-агента (этап 0.3).

Дизайн согласован с пользователем 23.07.2026 (см. PLAN.md, переписка):
- хук вызывается ПОСЛЕ родительского next(): агент может только ДОБАВИТЬ
  более ранний выход; все штатные выходы (сигнальный/legflip/TP/SL/CB)
  продолжают работать — «позже штатного нельзя» гарантировано структурой,
  а не проверкой;
- агентский выход НЕ помечается сигнальным → родительская логика (строка
  267 regimeb_bt_strategy.py) взводит cooldown, как после SL — решение
  пользователя «cooldown важен»;
- агент допущен ТОЛЬКО к dip-сделкам; transition-сделки запрещены
  (утверждено 23.07, совет sunday из from_sunday_reply_5 §2);
- exit_policy=None → код проходит строго по родительскому пути, поведение
  бит-в-бит равно зеркалу v7 (parity-режим). Никакой логики стратегии
  здесь НЕТ — только вызов политики и наблюдение за фактами.

Требует sunday в sys.path (см. strategy_rl.sundaybridge.ensure_sunday_first
или ручной sys.path.insert в скрипте-драйвере).
"""
from __future__ import annotations

from tooling.audit.regimeb_perfold15k.regimeb_bt_strategy import RegimeDipBuyerBT

# колонки поминутной экспертной разметки (BC-датасет, этап 2)
EXPERT_LOG_COLS = ["bar", "in_pos_before", "pos_tag", "mech_exit", "agent_exit",
                   "entry_placed", "exit_sig", "cooldown_active", "cb_active"]


class AgentHookBT(RegimeDipBuyerBT):
    # callable(strategy, i) -> True (выйти сейчас) / False (держать).
    # Вызывается только в открытой dip-позиции и только если штатный выход
    # на этом баре НЕ сработал.
    exit_policy = None
    # list-приёмник кортежей EXPERT_LOG_COLS (поминутная разметка для BC);
    # None = не логировать.
    expert_log = None

    def next(self):
        flag_before = self._await_signal_close
        pos_before = bool(self.position)
        tag_before = self._pos_tag if pos_before else ""

        super().next()

        # штатный выход на этом баре? (родитель ставит флаг при close по
        # сигналу/TP/legflip; SL исполняет движок вне next())
        parent_closed = self._await_signal_close and not flag_before

        agent_closed = False
        if (self.exit_policy is not None and self.position
                and not parent_closed and self._pos_tag == "dip"):
            i = len(self.data) - 1
            if self.exit_policy(self, i):
                # НЕ сигнальный close: _await_signal_close не трогаем,
                # родитель на следующем баре взведёт cooldown (строка 267)
                self.position.close()
                agent_closed = True

        if self.expert_log is not None:
            i = len(self.data) - 1
            entry_placed = (not pos_before) and len(self.orders) > 0
            self.expert_log.append((
                i, pos_before, tag_before,
                bool(parent_closed), bool(agent_closed), bool(entry_placed),
                bool(self._exit[i]),
                bool(i <= self._cooldown_until),
                bool(self._cb_triggered)))
