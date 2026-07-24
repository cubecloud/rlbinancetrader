"""Жёсткие правила мира: единый механизм «предикат на баре -> выход».

Приём из PLAN v0.5, 4.5.2 (пункт 2): правило — это ПАРАМЕТР конфигурации,
а не отдельная ветка в коде среды. Добавление правила = новый предикат в
этом списке плюс флаг в WorldConfig плюс запись в манифест, а не правка
шага среды.

Чего здесь НЕТ: решений агента и наград.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from spotrl.config import WorldConfig
from spotrl.envs.tradebook import TradeBook


@dataclass(frozen=True)
class ForcedExit:
    """Результат срабатывания правила мира.

    Attributes:
        reason: причина выхода ('sl', 'tp', 'end').
        price: цена, по которой правило закрывает позицию.
    """

    reason: str
    price: float


def check_forced_exit(book: TradeBook, world: WorldConfig, high: float, low: float,
                      close: float, is_last_bar: bool) -> Optional[ForcedExit]:
    """Проверить жёсткие правила на баре и вернуть принудительный выход.

    Порядок проверок задан механикой движка v7 (PLAN, 3.5.4):
    стоп-лосс — внутрибарно по Low (брокерский стоп),
    тейк-профит — по цене закрытия бара,
    закрытие на конце данных — чтобы у каждой сделки был исход (4.5.6).

    Returns:
        ForcedExit или None, если ни одно правило не сработало.
    """
    trade = book.open_trade
    if trade is None:
        return None
    if world.stop_loss_frac > 0.0:
        stop_price = trade.entry_price * (1.0 - trade.sl_frac)
        if low <= stop_price:
            return ForcedExit(reason="sl", price=stop_price)
    take_price = trade.entry_price * (1.0 + trade.tp_frac)
    if close >= take_price:
        return ForcedExit(reason="tp", price=close)
    if is_last_bar and world.close_at_end:
        return ForcedExit(reason="end", price=close)
    return None


def breaker_armed(equity_drawdown: float, world: WorldConfig) -> bool:
    """Сработал ли circuit breaker по просадке эквити счёта."""
    return equity_drawdown >= world.breaker_drawdown_frac
