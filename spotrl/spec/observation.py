"""Спецификация наблюдения: состав вектора и зарезервированные слоты.

Что здесь: имена признаков и их порядок как ДАННЫЕ (версионировано).
Чего здесь НЕТ: чтения данных и вычисления признаков — это features/.

Приём из PLAN v0.5, 4.5.2: в векторе заранее объявлены слоты-заглушки
`rule_slot_0..2` с константой 0.0. Тогда добавление нового жёсткого правила
мира не меняет размерность входа и не требует расширять первый слой сети.
Константный признак градиента не даёт и ни на что не влияет.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Tuple

# Признаки рынка (s_* — соглашение репозитория для колонок state-таблицы)
MARKET_FEATURES: Tuple[str, ...] = (
    "s_ret_1", "s_ret_5", "s_ret_60",
    "s_atr_rel", "s_q_buy", "s_q_sell", "s_regime_code", "s_leg_dn",
)

# ---------------------------------------------------------------------------
# СОСТАВ v2 (observation_design_2026-07-25.md) — версионированные ДАННЫЕ.
# Порядок колонок и порядок one-hot стабильны В ПРЕДЕЛАХ версии spec; хэш
# spec фиксирует весь этот состав вместе с константами нормировки.
# ---------------------------------------------------------------------------

# Блок РЫНОК. Цены — доходности как prepare_ret_obs, но БЕЗ утечки bfill:
# x1..x3/vwap_ret = Δlog(...) (истинная лог-доходность, а не pct_change по
# лог-цене), x4/x5 — внутрибарные сырые пропорции. Первый бар = 0.
MARKET_PRICE_FEATURES_V2: Tuple[str, ...] = (
    "m_ret_close",    # Δlog(close)
    "m_ret_high",     # Δlog(high)
    "m_ret_low",      # Δlog(low)
    "m_vwap_ret",     # Δlog(vwap), vwap — скользящий, строго .shift(1)
    "m_hi_close",     # (high − close) / close
    "m_close_lo",     # (close − low) / close
)

# Торговые числа — 3 безразмерных, снимают многолетний рост объёмов.
MARKET_FLOW_FEATURES_V2: Tuple[str, ...] = (
    "m_aggr_buy_frac",   # taker_buy_base / base_volume; volume=0 → 0.5
    "m_avg_trade_size",  # log(base_volume/trades) − log(rolling_median); trades=0 → 0
    "m_rel_volume",      # log(quote_volume / rolling_median_24h), .shift(1), clip ±3
)

# Сигналы v7 и их непрерывные составляющие (для копирования, гейт немарковости).
# Каузальность: прогнозы голов (predict_time=now) — можно; реальное будущее — нет.
MARKET_V7_FEATURES_V2: Tuple[str, ...] = (
    "m_entry_signal",        # булев self._entry
    "m_trans_entry_signal",  # булев self._trans_entry
    "m_buy_margin",          # непрерывная составляющая входа (q_buy − thr_buy)
    "m_sell_margin",         # непрерывная составляющая выхода (q_sell − thr_sell)
    "m_bounce_pct",          # bounce эксперта
    "m_leg_age",             # возраст ноги (leg_age_h), непрерывная
    "m_exit_flag",           # _exit[i] — штатный выход на баре (булев)
    "m_leg_dn",              # leg_dn ТЕКУЩЕГО бара — определяющий разделитель
                             # гейта выхода 1.2а (популяция = leg_dn=True dip-бары);
                             # entered_on_up (константа входа) его НЕ заменяет
    "m_regime_code",         # режим рынка
)

MARKET_FEATURES_V2: Tuple[str, ...] = (
    MARKET_PRICE_FEATURES_V2 + MARKET_FLOW_FEATURES_V2 + MARKET_V7_FEATURES_V2)

# Блок СОСТОЯНИЕ АГЕНТА (среда считает из СВОЕЙ книги, П4). Одна величина не
# ограничена сверху (дни в позиции) — только она проходит через tanh; остальное
# уже само-нормировано. pos_tag — one-hot, порядок категорий ЗАФИКСИРОВАН.
POS_TAG_ORDER_V2: Tuple[str, ...] = ("none", "dip", "transition")

AGENT_FEATURES_V2: Tuple[str, ...] = (
    "a_in_position",      # булев
    "a_unreal_pnl",       # ЧИСТЫЙ нереализованный PnL, clip ±0.5
    "a_peak_unreal",      # ЧИСТЫЙ пик нереализованного PnL
    "a_price_drawdown",   # price/peak_price − 1 (≤0), откат от пика позиции
    "a_dist_to_sl",       # ВАЛОВОЕ расстояние до ценового уровня стопа
    "a_dist_to_tp",       # ВАЛОВОЕ расстояние до ценового уровня тейка
    "a_entered_on_up",    # булев: вход при not leg_dn на баре решения
    "a_pos_tag_none",     # one-hot pos_tag (порядок POS_TAG_ORDER_V2)
    "a_pos_tag_dip",
    "a_pos_tag_transition",
    "a_cb_active",        # булев: circuit breaker активен (вход запрещён)
    "a_cb_cleared_today",  # булев: CB снят в текущем календарном дне
    "a_cooldown_remain",  # остаток cooldown 0..1 (remaining / breaker_cooldown_bars)
    "a_days_in_trade",    # tanh(bars_in_trade / median_hold)
)

WORLD_RULE_FEATURES_V2: Tuple[str, ...] = (
    "w_data_age", "w_breaker_armed", "w_equity_drawdown",
)

# Константы нормировки — часть СПЕЦИФИКАЦИИ (входят в хэш spec), НЕ μ/σ.
# ПРОВЕРЕНО по доку/spec/движку v7 — можно фиксировать хэшем:
SPEC_CONSTANTS_V2_VERIFIED: Dict[str, object] = {
    "median_hold_bars": 2442,        # tanh(days/median_hold), медиана длит. сделки (spec)
    "rel_volume_median_window_bars": 24 * 60,  # rolling_median_24h — зафикс. дизайн-доком
    "clip_rel_volume": 3.0,          # относительный объём clip ±3 (док)
    "clip_unreal_pnl": 0.5,          # чистый PnL clip ±0.5 (док)
    "vol_zero_neutral": 0.5,         # доля агрессивных покупок при volume=0 (док)
    "pos_tag_order": list(POS_TAG_ORDER_V2),
    "rolling_shift": 1,              # все скользящие окна строго .shift(1) (док)
    "price_warmup_fill": 0.0,        # прогрев доходностей — ноль, НЕ bfill (док)
    "cooldown_len_bars": 423,        # длина post-exit cooldown v7 (P7 cooldown_bars);
                                     # подтверждена машиной среды и гейтом env↔expert
                                     # (0 расхождений cooldown на обеих эпохах)
}

# ПРЕДВАРИТЕЛЬНО (НЕ подтверждено кодом среды) — потому версия помечена "-draft".
# ВАЖНО: v7 имеет ДВА разных механизма: post-SL entry cooldown
# (regimeb_bt_strategy.py:102-109, `cooldown_bars`, по умолчанию 376, срабатывает
# ТОЛЬКО после SL) и circuit breaker `cb_active` (снятие по календарному дню).
# Их НЕЛЬЗЯ сливать: гейт входа использует cooldown_active и cb_active как ОТДЕЛЬНЫЕ
# разделители. Раньше здесь ошибочно стояло cooldown_len=1440 (= breaker cooldown) —
# это неверно. Длину делителя a_cooldown_remain и окна vwap/avg_size фиксируем
# ТОЛЬКО когда среда реализует соответствующую машину состояния (раздел 2.1 отчёта).
SPEC_CONSTANTS_V2_PROVISIONAL: Dict[str, object] = {
    "vwap_window_bars": None,        # окно скользящего vwap — не зафикс. доком
    "avg_size_median_window_bars": None,  # окно медианы среднего размера сделки — не зафикс.
}

# Полный набор констант спецификации (проверенные + предварительные).
SPEC_CONSTANTS_V2: Dict[str, object] = {
    **SPEC_CONSTANTS_V2_VERIFIED, **SPEC_CONSTANTS_V2_PROVISIONAL}

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
    constants: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Валидация: имена признаков уникальны."""
        names = self.names
        if len(set(names)) != len(names):
            raise ValueError("имена признаков наблюдения должны быть уникальны")

    @classmethod
    def v2(cls) -> "ObservationSpec":
        """Состав v2 по observation_design_2026-07-25.md (версионированные данные).

        Порядок блоков: рынок (цены, торговые числа, сигналы v7) → состояние
        агента → правила мира → зарезервированные слоты. Константы нормировки
        (median_hold, окна, пороги clip, порядок one-hot) — часть спецификации
        и входят в хэш.

        Версия помечена "v2-draft": СОСТАВ и порядок колонок зафиксированы, но
        часть констант нормировки предварительна (длина post-SL cooldown, окна
        vwap/avg_size — их фиксирует только реализация машины состояния среды).
        Пока они None; авторитетным "v2" хэш станет после их фиксации.
        """
        return cls(version="v2-draft", market=MARKET_FEATURES_V2,
                   agent=AGENT_FEATURES_V2, world_rules=WORLD_RULE_FEATURES_V2,
                   reserved=RESERVED_SLOTS, constants=dict(SPEC_CONSTANTS_V2))

    def spec_hash(self) -> str:
        """sha256 всей спецификации (состав + порядок + константы нормировки).

        Хэш фиксирует СПЕЦИФИКАЦИЮ, а не μ/σ: длины окон, константы, список и
        порядок колонок, порядок one-hot. Стабилен при равном составе.
        """
        payload = json.dumps(
            {"version": self.version, "names": list(self.names),
             "constants": self.constants},
            sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

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
                "reserved": list(self.reserved),
                "constants": self.constants,
                "spec_hash": self.spec_hash()}
