"""Спецификация пространства действий — версионированная (PLAN v0.5, 3.5).

Что здесь: описание действий и корзин параметров сделки как ДАННЫЕ.
Чего здесь НЕТ: вычислений, обращений к среде, масок действий (масок в v0.5
нет вовсе — свобода регулируется флагами исполнения среды и loss-masking'ом
по головам, см. envs/freedom.py и algo/head_masking.py).

Пространство: MultiDiscrete([2, n_SL, n_TP]).
  голова 0 — {STAY, FLIP}: единственное решение о позиции на каждом баре;
  голова 1 — корзина стоп-лосса, содержательна ТОЛЬКО на баре входа;
  голова 2 — корзина тейк-профита, там же.

Совместимость обеспечивается ВЕРСИЕЙ спецификации (`ActionSpec.version`),
которая попадает в манифест прогона, а не комментарием «индексы стабильны».
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np

# голова 0: индексы стабильны внутри одной версии спецификации
STAY, FLIP = 0, 1
POSITION_ACTION_NAMES: Tuple[str, ...] = ("stay", "flip")
N_POSITION_ACTIONS = 2

# Индексы голов в векторе действия MultiDiscrete
HEAD_POSITION, HEAD_SL, HEAD_TP = 0, 1, 2
HEAD_NAMES: Tuple[str, ...] = ("position", "sl", "tp")
N_HEADS = 3

# Границы корзин SL/TP — ЗАГЛУШКА до офлайн-перебора (волна 2, PLAN 3.5.4).
# Значения в долях цены входа; первый элемент = константа v7 (нужна для
# гейта копирования: клон обязан УМЕТЬ выразить действие эксперта точно).
DEFAULT_SL_BUCKETS: Tuple[float, ...] = (0.05,)
DEFAULT_TP_BUCKETS: Tuple[float, ...] = (0.10,)


@dataclass(frozen=True)
class ActionSpec:
    """Версионированная спецификация пространства действий.

    Attributes:
        version: версия спецификации, попадает в манифест прогона (1.5.1, п.4).
        sl_buckets: значения стоп-лосса в долях цены входа, по возрастанию.
        tp_buckets: значения тейк-профита в долях цены входа, по возрастанию.
        expert_sl_index: индекс корзины, равной константе эксперта v7.
        expert_tp_index: индекс корзины, равной константе эксперта v7.

    На шагах лестницы до L4п корзины вырождены (длина 1) — тогда
    MultiDiscrete([2, 1, 1]) поведенчески тождественно Discrete(2).
    """

    version: str = "v1"
    sl_buckets: Tuple[float, ...] = field(default=DEFAULT_SL_BUCKETS)
    tp_buckets: Tuple[float, ...] = field(default=DEFAULT_TP_BUCKETS)
    expert_sl_index: int = 0
    expert_tp_index: int = 0

    def __post_init__(self) -> None:
        """Валидация на границе (PLAN 1.5.1, п.2)."""
        for name, buckets in (("sl_buckets", self.sl_buckets),
                              ("tp_buckets", self.tp_buckets)):
            if len(buckets) < 1:
                raise ValueError(f"{name}: нужна хотя бы одна корзина")
            if any(b <= 0.0 for b in buckets):
                raise ValueError(f"{name}: значения должны быть > 0")
            if list(buckets) != sorted(buckets):
                raise ValueError(f"{name}: корзины должны идти по возрастанию")
        if not 0 <= self.expert_sl_index < len(self.sl_buckets):
            raise ValueError("expert_sl_index вне диапазона sl_buckets")
        if not 0 <= self.expert_tp_index < len(self.tp_buckets):
            raise ValueError("expert_tp_index вне диапазона tp_buckets")

    @property
    def nvec(self) -> Tuple[int, int, int]:
        """Размерности голов для gymnasium.spaces.MultiDiscrete."""
        return (N_POSITION_ACTIONS, len(self.sl_buckets), len(self.tp_buckets))

    def expert_action(self, position_action: int) -> np.ndarray:
        """Действие эксперта v7: решение о позиции + константные SL/TP.

        Нужна для BC: цель по головам SL/TP — индекс корзины, равной
        константе эксперта. Если константы нет в корзинах, копирование
        бар-в-бар невозможно в принципе (PLAN 3.5.3).
        """
        if position_action not in (STAY, FLIP):
            raise ValueError(f"position_action должен быть STAY/FLIP, дано {position_action}")
        return np.array([position_action, self.expert_sl_index, self.expert_tp_index],
                        dtype=np.int64)

    def decode(self, action: Sequence[int]) -> "DecodedAction":
        """Разбор вектора действия в осмысленные величины."""
        if len(action) != N_HEADS:
            raise ValueError(f"ожидалось {N_HEADS} голов, дано {len(action)}")
        pos, i_sl, i_tp = int(action[0]), int(action[1]), int(action[2])
        if pos not in (STAY, FLIP):
            raise ValueError(f"голова позиции: недопустимый индекс {pos}")
        return DecodedAction(flip=(pos == FLIP),
                             sl_frac=self.sl_buckets[i_sl],
                             tp_frac=self.tp_buckets[i_tp])

    def describe(self) -> dict:
        """Сериализуемое описание для манифеста прогона."""
        return {"action_spec_version": self.version,
                "nvec": list(self.nvec),
                "sl_buckets": list(self.sl_buckets),
                "tp_buckets": list(self.tp_buckets),
                "expert_sl_index": self.expert_sl_index,
                "expert_tp_index": self.expert_tp_index}


@dataclass(frozen=True)
class DecodedAction:
    """Разобранное действие агента на одном баре.

    Attributes:
        flip: переключить состояние позиции (вне позиции — войти, в позиции — выйти).
        sl_frac: доля цены входа для стоп-лосса (значима только на баре входа).
        tp_frac: доля цены входа для тейк-профита (значима только на баре входа).
    """

    flip: bool
    sl_frac: float
    tp_frac: float
