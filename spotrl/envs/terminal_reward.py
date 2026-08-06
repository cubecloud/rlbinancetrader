"""ТЕРМИНАЛЬНАЯ per-trade награда (amend4, принцип 1 пользователя).

Значение имеет ФИНАЛЬНАЯ эквити сделки, не поминутный шум. Wrapper копит
поминутный log-прирост эквити внутри позиции и отдаёт СУММУ одним скаляром на
баре ЗАКРЫТИЯ сделки (0 на всех прочих барах в позиции). Инвариант T7 сохранён:
Σ поминутного reward за сделку == терминальный скаляр == log(1+доходность).

Радиус (проверено §1 k4_root_check): circuit breaker читает КРИВУЮ эквити
(spot_env считает просадку по world_equity независимо от скаляра награды) —
перенос награды на закрытие CB НЕ трогает. Эквити-кривая обновляется
mark-to-market каждый бар как раньше; меняется только КОГДА прилетает reward.

Немедленный кредит выходу: агентский FLIP закрывает сделку → терминальный
reward прилетает НА ТОМ ЖЕ баре, где принято решение о выходе. Дисперсия
поминутного mark-to-market убрана с промежуточных баров.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from spotrl.spec.observation import ObservationSpec


class RewardToTradeClose(gym.Wrapper):
    """Переносит поминутный reward на бар закрытия сделки (сумма = T7-инвариант).

    Оборачивается ВОКРУГ драйвера (env → driver → RewardToTradeClose). obs не
    трогает (SeedScaler стоит снаружи). Флаги/книга среды не меняются.
    """

    _I_INPOS = tuple(ObservationSpec.v2().names).index("a_in_position")

    def __init__(self, env):
        """Обернуть драйвер; аккумулятор награды пуст, позиция flat."""
        super().__init__(env)
        self._acc = 0.0
        self._was_in_pos = False

    def reset(self, **kwargs):
        """Сброс аккумулятора терминальной награды."""
        self._acc = 0.0
        self._was_in_pos = False
        return self.env.reset(**kwargs)

    def step(self, action):
        """Копить reward в позиции; отдать сумму на баре закрытия сделки."""
        obs, r, term, trunc, info = self.env.step(action)
        in_pos_after = bool(np.asarray(obs).reshape(-1)[self._I_INPOS] > 0.5)
        self._acc += float(r)

        if self._was_in_pos and not in_pos_after:
            out_r = self._acc            # сделка закрылась — отдать накопленное
            self._acc = 0.0
        elif in_pos_after:
            out_r = 0.0                  # держим позицию — отложить кредит
        else:
            out_r = self._acc            # flat→flat: пропустить (≈0)
            self._acc = 0.0

        if trunc and self._acc != 0.0:   # усечение в позиции — не терять кредит
            out_r += self._acc
            self._acc = 0.0

        self._was_in_pos = in_pos_after
        return obs, out_r, term, trunc, info
