"""Сборка вектора наблюдения — ЧИСТАЯ функция состояния (PLAN, тест Т2).

Что здесь: предвычисление рыночной части признаков по StateDataset и сборка
полного вектора наблюдения на баре t.

Чего здесь НЕТ: мутации состояния. Дефект старого кода (`policyio.py:57-59`,
`env.py:89` — обновление `peak_unreal` внутри вычисления наблюдения) здесь не
повторяется: пик передаётся снаружи готовым числом, наблюдение его не меняет.
Обращения к барам > t также запрещены (каузальность).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from spotrl.data.state_dataset import StateDataset
from spotrl.spec.observation import (AGENT_FEATURES_V2, ObservationSpec,
                                     RESERVED_SLOT_VALUE, SPEC_CONSTANTS_V2)


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
        entered_on_up: вход на растущей ноге (not leg_dn на баре решения);
            False вне позиции.
        pos_tag: тип открытой сделки — 'none' (вне позиции), 'dip' или
            'transition'; определён на баре входа.
        price_drawdown: откат от пика цены позиции = price/peak_price − 1 (≤0;
            0 вне позиции).

    Поля v2 (entered_on_up, pos_tag, price_drawdown) имеют дефолты и не влияют на
    сборку v1-наблюдения: `build_observation` для spec v1 читает только первые
    шесть полей.
    """

    in_position: bool = False
    unreal_pnl: float = 0.0
    peak_unreal: float = 0.0
    bars_in_trade: int = 0
    dist_to_sl: float = 0.0
    dist_to_tp: float = 0.0
    entered_on_up: bool = False
    pos_tag: str = "none"
    price_drawdown: float = 0.0


@dataclass(frozen=True)
class WorldState:
    """Наблюдаемая часть жёстких правил мира (PLAN 4.5.3 — правила с памятью).

    Attributes:
        data_age_bars: возраст последнего бара в барах (в backtest = 1).
        breaker_armed: circuit breaker сработал и вход запрещён.
        equity_drawdown: текущая просадка эквити от пика, доля (equity/пик − 1, ≤0).
        cb_active: circuit breaker v7 активен (halt по просадке, снятие по
            календарному дню); отдельный разделитель гейта входа.
        cb_cleared_today: CB был активен и снят в ТЕКУЩЕМ календарном дне.
        cooldown_active: активен post-exit cooldown (вход запрещён N баров после
            НЕсигнального закрытия — SL или агентского выхода).
        cooldown_remain: остаток cooldown в долях его длины (0..1).

    Поля v2 (cb_active, cb_cleared_today, cooldown_active, cooldown_remain) имеют
    дефолты и не участвуют в сборке v1-наблюдения.
    """

    data_age_bars: int = 1
    breaker_armed: bool = False
    equity_drawdown: float = 0.0
    cb_active: bool = False
    cb_cleared_today: bool = False
    cooldown_active: bool = False
    cooldown_remain: float = 0.0


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


def _delta_log(x: np.ndarray) -> np.ndarray:
    """Лог-доходность бар-к-бару Δlog(x); первый бар = 0 (warmup, БЕЗ bfill).

    Каузально: значение на баре t использует только x[t] и x[t-1]. Утечки нет —
    префикс [0:t] даёт тот же результат в точке t, что и полный ряд.
    """
    log_x = np.log(np.maximum(x, 1e-12))
    return np.diff(log_x, prepend=log_x[0])


def _causal_rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    """Скользящая медиана строго ПРОШЛОГО: out[t] = median(x[t-window:t]).

    Текущий бар ИСКЛЮЧЁН (`.shift(1)`), поэтому признак на баре t не видит x[t]
    и утечки нет (гейт утечки: префикс [0:t] == полный ряд в t). Прогрев
    (min_periods=1) считает медиану по доступному прошлому; на баре 0 прошлого
    нет → NaN (вызывающая сторона заменяет на нейтраль).
    """
    s = pd.Series(x, dtype="float64")
    return s.rolling(window=window, min_periods=1).median().shift(1).to_numpy()


def precompute_market_features_v2(
        state: StateDataset,
        constants: Optional[dict] = None) -> np.ndarray:
    """Рыночный блок наблюдения v2 для всех баров сразу, массив (n, 18) float32.

    Состав и порядок = `MARKET_FEATURES_V2` (spec): 6 цен (Δlog + внутрибарные),
    3 безразмерных торговых числа, 9 сигналов v7. Все признаки КАУЗАЛЬНЫ и
    само-нормированы (без замороженного скейлера). Скользящие окна — строго
    прошлое (`.shift(1)`), пороги clip — фиксированные из spec.

    Args:
        state: источник данных (нужны quote_volume/trades/taker_buy_base и
            непрерывные составляющие v7; их отсутствие деградирует к нейтрали).
        constants: константы нормировки из spec (окна, пороги clip). По
            умолчанию `SPEC_CONSTANTS_V2`.

    Returns:
        Массив (n, 18) float32 в порядке `ObservationSpec.v2().market`.
    """
    c = constants or SPEC_CONSTANTS_V2
    clip_rv = float(c["clip_rel_volume"])
    vol_neutral = float(c["vol_zero_neutral"])
    rv_win = int(c["rel_volume_median_window_bars"])
    as_win = int(c["avg_size_median_window_bars"])

    close = state.close
    high, low = state.ohlcv[:, 1], state.ohlcv[:, 2]
    base_vol = state.base_volume
    quote_vol = state.quote_volume
    trades = state.trades
    taker_base = state.taker_buy_base

    # --- Цены (6) ---
    ret_close = _delta_log(close)
    ret_high = _delta_log(high)
    ret_low = _delta_log(low)
    # vwap побарный = quote/base (истинная средневзвешенная цена бара); при
    # отсутствии денежного объёма/нулевом объёме — типичная цена (H+L+C)/3.
    typical = (high + low + close) / 3.0
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = quote_vol / base_vol
    bad_vwap = ~np.isfinite(vwap) | (base_vol <= 0.0)
    vwap = np.where(bad_vwap, typical, vwap)
    vwap_ret = _delta_log(vwap)
    hi_close = (high - close) / np.maximum(close, 1e-12)
    close_lo = (close - low) / np.maximum(close, 1e-12)

    # --- Торговые числа (3), безразмерные ---
    # 1) доля агрессивных покупок; base_volume=0/NaN → нейтраль 0.5 (НЕ clip)
    with np.errstate(divide="ignore", invalid="ignore"):
        aggr = taker_base / base_vol
    aggr = np.where(np.isfinite(aggr) & (base_vol > 0.0), aggr, vol_neutral)
    # 2) средний размер сделки = log(base/trades) − log(median_past(base/trades))
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_size = base_vol / trades
    valid_size = np.isfinite(raw_size) & (trades > 0.0) & (base_vol > 0.0)
    log_size = np.where(valid_size, np.log(np.where(valid_size, raw_size, 1.0)), np.nan)
    med_size = _causal_rolling_median(np.where(valid_size, raw_size, np.nan), as_win)
    avg_trade_size = log_size - np.log(np.maximum(med_size, 1e-12))
    avg_trade_size = np.where(np.isfinite(avg_trade_size), avg_trade_size, 0.0)
    # 3) относительный объём = log(quote / median_past_24h(quote)), clip ±3
    med_q = _causal_rolling_median(quote_vol, rv_win)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_vol = np.log(quote_vol / med_q)
    rel_vol = np.where(np.isfinite(rel_vol), rel_vol, 0.0)
    rel_vol = np.clip(rel_vol, -clip_rv, clip_rv)

    # --- Сигналы v7 (9) ---
    return np.column_stack([
        ret_close, ret_high, ret_low, vwap_ret, hi_close, close_lo,
        aggr, avg_trade_size, rel_vol,
        state.entry_signal.astype(np.float64),
        state.trans_entry_signal.astype(np.float64),
        state.buy_margin, state.sell_margin, state.bounce_pct, state.leg_age,
        state.exit_sig.astype(np.float64),
        state.leg_dn.astype(np.float64),
        state.regime_code.astype(np.float64),
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
        drawdown: просадка эквити от пика, доля (equity/пик − 1, ≤0).
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

    Диспетчеризует по версии spec: v1 (6 полей агента) — прежний путь; v2 (14
    полей агента) — расширенная упаковка `write_observation_v2`. Функция чистая.

    Args:
        market_row: строка предвычисленных рыночных признаков (spec.market).
        agent: состояние агента, посчитанное средой.
        world: наблюдаемая часть правил мира.
        spec: версионированный состав наблюдения.

    Returns:
        Вектор float32 длины `spec.size`. Одинаковые входы → побитово одинаковый
        выход, аргументы не изменяются.
    """
    if len(market_row) != len(spec.market):
        raise ValueError(f"market_row: ожидалось {len(spec.market)}, дано {len(market_row)}")
    if len(spec.agent) == len(AGENT_FEATURES_V2):
        return write_observation_v2(
            np.empty(spec.size, dtype=np.float32), market_row, agent, world,
            ObservationLayout.from_spec(spec), spec.constants)
    return write_observation(
        np.empty(spec.size, dtype=np.float32), market_row,
        float(agent.in_position), agent.unreal_pnl, agent.peak_unreal,
        float(agent.bars_in_trade), agent.dist_to_sl, agent.dist_to_tp,
        float(world.data_age_bars), float(world.breaker_armed), world.equity_drawdown,
        ObservationLayout.from_spec(spec))


def write_observation_v2(out: np.ndarray, market_row: np.ndarray,
                         agent: AgentState, world: WorldState,
                         layout: ObservationLayout,
                         constants: Optional[dict] = None) -> np.ndarray:
    """Упаковать наблюдение v2 (18 рынок + 14 агент + 3 мир + 3 резерв) в `out`.

    Порядок полей агента = `AGENT_FEATURES_V2`; нормировки по
    observation_design: чистый PnL clip ±0.5, one-hot pos_tag (порядок
    `POS_TAG_ORDER_V2`), tanh(bars/median_hold). Функция ЧИСТАЯ: пишет только в
    `out`, состояние агента не мутирует.

    ВАЖНО (решение дизайна, impl_state_machine 207-218): слот просадки эквити,
    который видит агент (`w_equity_drawdown`), в режиме копирования = просадка
    МИРОВОЙ CB-эквити (`world.equity_drawdown` заполняется средой мировой
    просадкой), НЕ наградной. Свою открытую прибыль агент видит через
    `a_unreal_pnl`/`a_peak_unreal`. Здесь упаковщик кладёт то, что передала
    среда в `world.equity_drawdown` — обязанность подать мировую просадку лежит
    на `world_state()` при v2-упаковке.

    Args:
        out: буфер float32 длины `layout.size`.
        market_row: строка рыночных признаков v2 (18).
        agent: состояние агента (все 14 полей v2).
        world: наблюдаемая часть правил мира v2.
        layout: числовые смещения блоков.
        constants: константы нормировки spec (median_hold, clip PnL, порядок
            one-hot). По умолчанию `SPEC_CONSTANTS_V2`.

    Returns:
        Тот же массив `out`.
    """
    return write_observation_v2_flat(
        out, market_row, float(agent.in_position), agent.unreal_pnl,
        agent.peak_unreal, agent.price_drawdown, agent.dist_to_sl, agent.dist_to_tp,
        float(agent.entered_on_up), agent.pos_tag, float(world.cb_active),
        float(world.cb_cleared_today), world.cooldown_remain,
        float(agent.bars_in_trade), float(world.data_age_bars),
        float(world.breaker_armed), world.equity_drawdown, layout, constants, True)


def write_observation_v2_flat(
        out: np.ndarray, market_row: np.ndarray, in_position: float,
        unreal_pnl: float, peak_unreal: float, price_drawdown: float,
        dist_to_sl: float, dist_to_tp: float, entered_on_up: float, pos_tag: str,
        cb_active: float, cb_cleared_today: float, cooldown_remain: float,
        bars_in_trade: float, data_age: float, breaker: float,
        equity_drawdown: float, layout: ObservationLayout,
        constants: Optional[dict] = None,
        write_reserved: bool = True) -> np.ndarray:
    """Плоский (без объектов) писарь наблюдения v2 — ЕДИНСТВЕННАЯ арифметика v2.

    `write_observation_v2` (объектная обёртка) и горячий путь `step` оба зовут
    ровно это ядро, поэтому наблюдение из `step` побитово равно `observe()`
    независимо от выбранных операций (гейт эквивалентности 100k шагов). Функция
    ЧИСТАЯ: пишет только в `out`. Порядок полей = `AGENT_FEATURES_V2` +
    `WORLD_RULE_FEATURES_V2` + резерв. Нормировки: чистый PnL clip ±clip_pnl
    (ручной clamp, без np — не создаём объектов на бар), one-hot pos_tag по
    `pos_tag_order`, tanh(bars/median_hold), cooldown_remain clamp ≤1.

    ВАЖНО (реш. дизайна): `equity_drawdown` — просадка МИРОВОЙ CB-эквити (по
    которой блокируются входы), а не наградной; передаётся вызывающей стороной.
    """
    c = constants or SPEC_CONSTANTS_V2
    clip_pnl = float(c["clip_unreal_pnl"])
    median_hold = float(c["median_hold_bars"])
    pos_order = c["pos_tag_order"]
    cd_len = float(c["cooldown_len_bars"])

    out[:layout.n_market] = market_row
    j = layout.agent_at
    out[j] = in_position
    out[j + 1] = (clip_pnl if unreal_pnl > clip_pnl
                  else -clip_pnl if unreal_pnl < -clip_pnl else unreal_pnl)
    out[j + 2] = (clip_pnl if peak_unreal > clip_pnl
                  else -clip_pnl if peak_unreal < -clip_pnl else peak_unreal)
    out[j + 3] = price_drawdown
    out[j + 4] = dist_to_sl
    out[j + 5] = dist_to_tp
    out[j + 6] = entered_on_up
    # one-hot pos_tag в порядке POS_TAG_ORDER_V2 (none, dip, transition)
    out[j + 7] = 1.0 if pos_tag == pos_order[0] else 0.0
    out[j + 8] = 1.0 if pos_tag == pos_order[1] else 0.0
    out[j + 9] = 1.0 if pos_tag == pos_order[2] else 0.0
    out[j + 10] = cb_active
    out[j + 11] = cb_cleared_today
    out[j + 12] = (1.0 if cooldown_remain > 1.0 else cooldown_remain) if cd_len > 0 else 0.0
    out[j + 13] = math.tanh(bars_in_trade / median_hold)
    w = layout.world_at
    out[w] = data_age
    out[w + 1] = breaker
    out[w + 2] = equity_drawdown
    if write_reserved:
        out[layout.reserved_at:] = RESERVED_SLOT_VALUE
    return out
