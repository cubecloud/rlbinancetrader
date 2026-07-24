"""Спецификация наблюдения: состав вектора и зарезервированные слоты.

Что здесь: имена признаков и их порядок как ДАННЫЕ (версионировано).
Чего здесь НЕТ: чтения данных и вычисления признаков — это features/.

Приём из PLAN v0.5, 4.5.2: в векторе заранее объявлены слоты-заглушки
`rule_slot_0..2` с константой 0.0. Тогда добавление нового жёсткого правила
мира не меняет размерность входа и не требует расширять первый слой сети.
Константный признак градиента не даёт и ни на что не влияет.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

# Признаки рынка (s_* — соглашение репозитория для колонок state-таблицы)
MARKET_FEATURES: Tuple[str, ...] = (
    "s_ret_1", "s_ret_5", "s_ret_60",
    "s_atr_rel", "s_q_buy", "s_q_sell", "s_regime_code", "s_leg_dn",
)

# Признаки состояния агента — считает ТОЛЬКО среда из своей книги сделок
# (PLAN П4): предвычисленные tb_-колонки использовать запрещено.
AGENT_FEATURES: Tuple[str, ...] = (
    "a_in_position", "a_unreal_pnl", "a_peak_unreal", "a_bars_in_trade",
    "a_dist_to_sl", "a_dist_to_tp",
)

# Признаки жёстких правил мира (наблюдаемость правил с памятью, PLAN 4.5.3)
WORLD_RULE_FEATURES: Tuple[str, ...] = (
    "w_data_age", "w_breaker_armed", "w_equity_drawdown",
)

# Зарезервированные слоты под будущие правила мира (PLAN 4.5.2, приём 1)
RESERVED_SLOTS: Tuple[str, ...] = ("rule_slot_0", "rule_slot_1", "rule_slot_2")
RESERVED_SLOT_VALUE = 0.0


@dataclass(frozen=True)
class ObservationSpec:
    """Версионированный состав вектора наблюдения.

    Attributes:
        version: версия состава, попадает в манифест прогона.
        market: имена рыночных признаков.
        agent: имена признаков состояния агента.
        world_rules: имена признаков жёстких правил мира.
        reserved: имена зарезервированных слотов-заглушек.
    """

    version: str = "v1"
    market: Tuple[str, ...] = field(default=MARKET_FEATURES)
    agent: Tuple[str, ...] = field(default=AGENT_FEATURES)
    world_rules: Tuple[str, ...] = field(default=WORLD_RULE_FEATURES)
    reserved: Tuple[str, ...] = field(default=RESERVED_SLOTS)

    def __post_init__(self) -> None:
        """Валидация: имена признаков уникальны."""
        names = self.names
        if len(set(names)) != len(names):
            raise ValueError("имена признаков наблюдения должны быть уникальны")

    @property
    def names(self) -> Tuple[str, ...]:
        """Полный порядок признаков в векторе наблюдения."""
        return self.market + self.agent + self.world_rules + self.reserved

    @property
    def size(self) -> int:
        """Размерность вектора наблюдения."""
        return len(self.names)

    def index_of(self, name: str) -> int:
        """Позиция признака в векторе; KeyError, если признака нет."""
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"нет такого признака наблюдения: {name}") from exc

    def describe(self) -> dict:
        """Сериализуемое описание для манифеста прогона."""
        return {"obs_spec_version": self.version,
                "size": self.size,
                "names": list(self.names),
                "reserved": list(self.reserved)}
