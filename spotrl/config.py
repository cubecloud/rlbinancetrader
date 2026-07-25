"""Конфигурация пакета: правила мира и параметры обучения как ДАННЫЕ.

Приём из PLAN v0.5, 4.5.2 (пункт 2): жёсткое правило мира — это параметр
конфигурации, а не новая ветка в коде среды. Каждое правило реализуется
одним механизмом «предикат на баре -> принудительный выход» (envs/worldrules.py),
включается флагом здесь и записывается в манифест прогона.

`world_version` обязателен: любое число сравнивается только с числом той же
версии мира (PLAN 4.5.2).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from spotrl.spec.actions import ActionSpec
from spotrl.spec.observation import ObservationSpec

# Комиссия за сторону — как в зеркале движка sunday (Backtest commission=0.001)
FEE_SIDE = 0.001


@dataclass(frozen=True)
class WorldConfig:
    """Жёсткие правила мира — параметры, а не код (PLAN 4.5.2, 4.5.3).

    Attributes:
        world_version: версия набора правил; входит в манифест прогона.
        stop_loss_frac: стоп-лосс на сделку, доля цены входа (0 = выключен).
        breaker_drawdown_frac: circuit breaker по просадке эквити счёта.
        breaker_cooldown_bars: сколько баров после срабатывания вход запрещён.
        cooldown_bars: post-exit cooldown v7 — сколько баров вход запрещён ПОСЛЕ
            НЕсигнального закрытия (SL или агентский выход). ДРУГОЙ механизм, чем
            circuit breaker. В v7 (P7) = 423. Разделитель гейта входа.
        close_at_end: принудительно закрывать позицию на конце данных, чтобы
            у каждой сделки был исход (PLAN 4.5.6).
        max_data_age_bars: правило «свежесть данных» — при большем возрасте
            последнего бара решения не принимаются (PLAN 4.5.3, 4-е правило).
        fee_side: комиссия на сторону, входит в награду (не «прикручена потом»).
    """

    world_version: str = "w1"
    stop_loss_frac: float = 0.05
    breaker_drawdown_frac: float = 0.50
    breaker_cooldown_bars: int = 1440
    cooldown_bars: int = 423
    close_at_end: bool = True
    max_data_age_bars: int = 5
    fee_side: float = FEE_SIDE

    def __post_init__(self) -> None:
        """Валидация на границе (PLAN 1.5.1, п.2)."""
        if not 0.0 <= self.stop_loss_frac < 1.0:
            raise ValueError("stop_loss_frac должен быть в [0, 1)")
        if not 0.0 < self.breaker_drawdown_frac <= 1.0:
            raise ValueError("breaker_drawdown_frac должен быть в (0, 1]")
        if self.breaker_cooldown_bars < 0:
            raise ValueError("breaker_cooldown_bars должен быть >= 0")
        if self.cooldown_bars < 0:
            raise ValueError("cooldown_bars должен быть >= 0")
        if self.max_data_age_bars < 1:
            raise ValueError("max_data_age_bars должен быть >= 1")
        if not 0.0 <= self.fee_side < 0.01:
            raise ValueError("fee_side вне разумного диапазона [0, 0.01)")

    def describe(self) -> dict:
        """Сериализуемое описание для манифеста прогона."""
        return asdict(self)


@dataclass(frozen=True)
class FreedomConfig:
    """Флаги ИСПОЛНЕНИЯ среды: какие решения агент принимает сам.

    Масок действий в v0.5 нет (PLAN, раздел 2). Если свобода выключена,
    среда исполняет решение эксперта v7 вместо агента, а соответствующая
    голова исключается из policy-лосса loss-masking'ом (algo/head_masking.py).

    Attributes:
        exit_own: агент решает выход из позиции.
        entry_own: агент решает вход в позицию.
        params_own: агент выбирает SL и TP на баре входа (шаг лестницы L4п).
    """

    exit_own: bool = True
    entry_own: bool = False
    params_own: bool = False

    def describe(self) -> dict:
        """Сериализуемое описание для манифеста прогона."""
        return asdict(self)


@dataclass(frozen=True)
class EnvConfig:
    """Параметры среды обучения.

    Attributes:
        world: жёсткие правила мира.
        freedom: флаги исполнения (свободы агента).
        action_spec: версионированное пространство действий.
        obs_spec: версионированный состав наблюдения.
        episode_len: длина эпизода в барах; по PLAN 5.2 должна быть не меньше
            максимальной длительности сделки эксперта (34 738 баров), иначе
            сделки обрываются усечением.
        gamma: коэффициент дисконтирования; ровно 1.0 — обязателен, не опция
            (PLAN 5.1, поминутная награда принята ПАКЕТОМ с gamma=1.0).
    """

    world: WorldConfig = field(default_factory=WorldConfig)
    freedom: FreedomConfig = field(default_factory=FreedomConfig)
    action_spec: ActionSpec = field(default_factory=ActionSpec)
    obs_spec: ObservationSpec = field(default_factory=ObservationSpec)
    episode_len: int = 34_738
    gamma: float = 1.0

    def __post_init__(self) -> None:
        """Валидация: gamma ровно 1.0, эпизод положительный."""
        if self.gamma != 1.0:
            raise ValueError("gamma обязана быть 1.0 (PLAN v0.5, раздел 5.1)")
        if self.episode_len < 1:
            raise ValueError("episode_len должен быть >= 1")

    def describe(self) -> dict:
        """Сериализуемое описание для манифеста прогона."""
        return {"world": self.world.describe(),
                "freedom": self.freedom.describe(),
                "action_spec": self.action_spec.describe(),
                "obs_spec": self.obs_spec.describe(),
                "episode_len": self.episode_len,
                "gamma": self.gamma}
