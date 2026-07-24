"""Сборка вектора наблюдения — ЧИСТАЯ функция состояния (PLAN, тест Т2).

Что здесь: предвычисление рыночной части признаков по StateDataset и сборка
полного вектора наблюдения на баре t.

Чего здесь НЕТ: мутации состояния. Дефект старого кода (`policyio.py:57-59`,
`env.py:89` — обновление `peak_unreal` внутри вычисления наблюдения) здесь не
повторяется: пик передаётся снаружи готовым числом, наблюдение его не меняет.
Обращения к барам > t также запрещены (каузальность).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spotrl.data.state_dataset import StateDataset
from spotrl.spec.observation import ObservationSpec, RESERVED_SLOT_VALUE


@dataclass(frozen=True)
class AgentState:
    """Состояние агента на баре t — считает среда из своей книги сделок (П4).

    Attributes:
        in_position: есть открытая позиция.
        unreal_pnl: незакрытая доходность позиции в долях (0 вне позиции).
        peak_unreal: максимум unreal_pnl за время текущей сделки.
        bars_in_trade: сколько баров позиция открыта.
        dist_to_sl: расстояние до стопа в долях цены (0 вне позиции).
        dist_to_tp: расстояние до тейка в долях цены (0 вне позиции).
    """

    in_position: bool = False
    unreal_pnl: float = 0.0
    peak_unreal: float = 0.0
    bars_in_trade: int = 0
    dist_to_sl: float = 0.0
    dist_to_tp: float = 0.0


@dataclass(frozen=True)
class WorldState:
    """Наблюдаемая часть жёстких правил мира (PLAN 4.5.3 — правила с памятью).

    Attributes:
        data_age_bars: возраст последнего бара в барах (в backtest = 1).
        breaker_armed: circuit breaker сработал и вход запрещён.
        equity_drawdown: текущая просадка эквити от пика, доля.
    """

    data_age_bars: int = 1
    breaker_armed: bool = False
    equity_drawdown: float = 0.0


def precompute_market_features(state: StateDataset) -> np.ndarray:
    """Рыночная часть наблюдения для всех баров сразу, массив (n, k).

    Все признаки каузальны: значение на баре t использует только бары <= t.
    Порядок колонок соответствует ObservationSpec.market.
    """
    close = state.close
    log_close = np.log(close)
    ret_1 = np.diff(log_close, prepend=log_close[0])
    ret_5 = log_close - np.concatenate([np.full(5, log_close[0]), log_close[:-5]])
    ret_60 = log_close - np.concatenate([np.full(60, log_close[0]), log_close[:-60]])
    high, low = state.ohlcv[:, 1], state.ohlcv[:, 2]
    atr_rel = (high - low) / np.maximum(close, 1e-12)
    return np.column_stack([
        ret_1, ret_5, ret_60, atr_rel,
        state.signals[:, 0], state.signals[:, 1],
        state.regime_code.astype(np.float64),
        state.leg_dn.astype(np.float64),
    ]).astype(np.float32)


@dataclass(frozen=True)
class ObservationLayout:
    """Разложение спецификации наблюдения в числовые смещения.

    Спецификация хранит ИМЕНА признаков, и вычисление `spec.size` каждый раз
    заново склеивает четыре кортежа имён (замер: 0.28 мкс на обращение). В
    горячем шаге это лишняя работа: состав наблюдения за прогон не меняется,
    поэтому смещения считаются один раз здесь.

    Attributes:
        size: длина вектора наблюдения.
        n_market: число рыночных признаков (они идут первыми).
        agent_at: смещение блока состояния агента.
        world_at: смещение блока правил мира.
        reserved_at: смещение блока зарезервированных слотов.
    """

    size: int
    n_market: int
    agent_at: int
    world_at: int
    reserved_at: int

    @classmethod
    def from_spec(cls, spec: ObservationSpec) -> "ObservationLayout":
        """Построить разложение по версионированной спецификации."""
        n_market = len(spec.market)
        agent_at = n_market
        world_at = agent_at + len(spec.agent)
        reserved_at = world_at + len(spec.world_rules)
        return cls(size=spec.size, n_market=n_market, agent_at=agent_at,
                   world_at=world_at, reserved_at=reserved_at)


def write_observation(out: np.ndarray, market_row: np.ndarray, in_position: float,
                      unreal_pnl: float, peak_unreal: float, bars_in_trade: float,
                      dist_to_sl: float, dist_to_tp: float, data_age: float,
                      breaker: float, drawdown: float, layout: ObservationLayout,
                      write_reserved: bool = True) -> np.ndarray:
    """Записать наблюдение в готовый буфер `out` и вернуть его.

    Единственная реализация раскладки вектора: `build_observation` — обёртка
    над этой функцией. Функция ЧИСТАЯ по отношению к состоянию среды: она
    пишет только в `out` и ничего не мутирует; при равных аргументах результат
    побитово одинаков (гейт Э1.2б проверяется на ДВУХ разных буферах).

    Args:
        out: буфер float32 длины `layout.size` (владелец — вызывающая сторона).
        market_row: строка предвычисленных рыночных признаков.
        in_position: 1.0 в позиции, иначе 0.0.
        unreal_pnl: незакрытая доходность в долях.
        peak_unreal: пик незакрытой доходности за сделку.
        bars_in_trade: сколько баров позиция открыта.
        dist_to_sl: расстояние до стопа в долях цены.
        dist_to_tp: расстояние до тейка в долях цены.
        data_age: возраст последнего бара в барах.
        breaker: 1.0 если circuit breaker сработал.
        drawdown: просадка эквити от пика, доля.
        layout: числовые смещения блоков.
        write_reserved: писать ли слоты-заглушки. Их значение — константа
            `RESERVED_SLOT_VALUE`, поэтому в постоянном буфере среды они
            заполняются один раз при создании, а на шаге не переписываются
            (замер: 0.28 мкс на шаг). Для свежего буфера флаг обязан быть
            True, иначе слоты останутся неинициализированными.

    Returns:
        Тот же массив `out`.
    """
    out[:layout.n_market] = market_row
    j = layout.agent_at
    out[j] = in_position
    out[j + 1] = unreal_pnl
    out[j + 2] = peak_unreal
    out[j + 3] = bars_in_trade
    out[j + 4] = dist_to_sl
    out[j + 5] = dist_to_tp
    w = layout.world_at
    out[w] = data_age
    out[w + 1] = breaker
    out[w + 2] = drawdown
    if write_reserved:
        out[layout.reserved_at:] = RESERVED_SLOT_VALUE
    return out


def build_observation(market_row: np.ndarray, agent: AgentState, world: WorldState,
                      spec: ObservationSpec) -> np.ndarray:
    """Собрать вектор наблюдения на одном баре в НОВОМ массиве.

    Args:
        market_row: строка предвычисленных рыночных признаков (spec.market).
        agent: состояние агента, посчитанное средой.
        world: наблюдаемая часть правил мира.
        spec: версионированный состав наблюдения.

    Returns:
        Вектор float32 длины `spec.size`. Функция чистая: одинаковые входы
        дают побитово одинаковый выход, аргументы не изменяются.
    """
    if len(market_row) != len(spec.market):
        raise ValueError(f"market_row: ожидалось {len(spec.market)}, дано {len(market_row)}")
    return write_observation(
        np.empty(spec.size, dtype=np.float32), market_row,
        float(agent.in_position), agent.unreal_pnl, agent.peak_unreal,
        float(agent.bars_in_trade), agent.dist_to_sl, agent.dist_to_tp,
        float(world.data_age_bars), float(world.breaker_armed), world.equity_drawdown,
        ObservationLayout.from_spec(spec))
