"""Книга сделок агента: единственный источник контекста позиции (PLAN П4).

Что здесь: учёт открытой позиции и список закрытых сделок.
Чего здесь НЕТ: чтения предвычисленных `tb_*`-колонок из parquet — состояние
агента считается средой из СВОИХ действий, иначе оно лживо при отклонении
агента от опорной политики.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OpenTrade:
    """Открытая позиция.

    Attributes:
        entry_bar: индекс бара входа (исполнение по open этого бара).
        entry_price: цена входа.
        sl_frac: стоп-лосс в долях цены входа, выбран на баре входа.
        tp_frac: тейк-профит в долях цены входа, выбран на баре входа.
        peak_unreal: максимум незакрытой доходности за время сделки.
    """

    entry_bar: int
    entry_price: float
    sl_frac: float
    tp_frac: float
    peak_unreal: float = 0.0


@dataclass(frozen=True)
class ClosedTrade:
    """Закрытая сделка.

    Attributes:
        entry_bar: индекс бара входа.
        exit_bar: индекс бара выхода.
        entry_price: цена входа.
        exit_price: цена выхода.
        return_pct: чистая доходность сделки в процентах (с комиссией).
        exit_reason: причина выхода ('agent', 'sl', 'tp', 'rule', 'end').
    """

    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    return_pct: float
    exit_reason: str


@dataclass
class TradeBook:
    """Книга сделок одного эпизода.

    Attributes:
        fee_side: комиссия на сторону, входит в доходность сделки.
        open_trade: текущая открытая позиция (None вне позиции).
        closed: список закрытых сделок в порядке закрытия.
    """

    fee_side: float
    open_trade: Optional[OpenTrade] = None
    closed: List[ClosedTrade] = field(default_factory=list)

    @property
    def in_position(self) -> bool:
        """Есть ли открытая позиция."""
        return self.open_trade is not None

    def open(self, bar: int, price: float, sl_frac: float, tp_frac: float) -> None:
        """Открыть позицию; повторное открытие запрещено (спот, long/flat)."""
        if self.open_trade is not None:
            raise RuntimeError("позиция уже открыта")
        self.open_trade = OpenTrade(entry_bar=bar, entry_price=float(price),
                                    sl_frac=float(sl_frac), tp_frac=float(tp_frac))

    def close(self, bar: int, price: float, reason: str) -> ClosedTrade:
        """Закрыть позицию и записать сделку в список закрытых."""
        if self.open_trade is None:
            raise RuntimeError("позиция не открыта")
        trade = self.open_trade
        gross = float(price) / trade.entry_price
        net = gross * (1.0 - self.fee_side) ** 2 - 1.0
        closed = ClosedTrade(entry_bar=trade.entry_bar, exit_bar=int(bar),
                             entry_price=trade.entry_price, exit_price=float(price),
                             return_pct=net * 100.0, exit_reason=reason)
        self.closed.append(closed)
        self.open_trade = None
        return closed

    def mark(self, price: float) -> float:
        """Обновить пик незакрытой доходности и вернуть текущую доходность.

        Мутация живёт ЗДЕСЬ, а не в вычислении наблюдения (тест Т2).
        """
        if self.open_trade is None:
            return 0.0
        unreal = float(price) / self.open_trade.entry_price - 1.0
        if unreal > self.open_trade.peak_unreal:
            self.open_trade.peak_unreal = unreal
        return unreal
