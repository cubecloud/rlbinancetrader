# Реализация наблюдения v2 — части 2 и 3 (2026-07-25)

Источник истины: `handoff/observation_design_2026-07-25.md` и разделы 2.2/2.3
`handoff/impl_obs_v2_2026-07-25.md`. Правились только `spotrl/{spec,features,
data,tests}/`. Коммитов нет. Часть 1 (машина скрытого состояния) и паритет 6/6
НЕ тронуты и остаются зелёными.

Быстрый набор: **129 passed, 4 skipped, 6 deselected, ~8.7 с** (бюджет ≤60 с).
Паритет 6/6 (slow) перепроверен после правок слоя данных: **2 passed, 25.5 с**.
Покрытие: builder **100%**, spot_env **99%**, observation **98%**, sources 94%,
state_dataset 90%, synthetic 100% — все ≥85%.

## 1. Рыночный строитель v2 + нормировка (СДЕЛАНО)

`spotrl/features/builder.py::precompute_market_features_v2(state, constants)` →
массив (n, 18) float32 в порядке `ObservationSpec.v2().market`. Все признаки
каузальны и само-нормированы, БЕЗ замороженного скейлера; скользящие окна строго
`.shift(1)` (текущий бар исключён).

Состав и формы нормировки:

| # | признак | форма | нормировка / край |
|---|---------|-------|-------------------|
| 0 | m_ret_close | Δlog(close) | первый бар=0, БЕЗ bfill |
| 1 | m_ret_high | Δlog(high) | -//- |
| 2 | m_ret_low | Δlog(low) | -//- |
| 3 | m_vwap_ret | Δlog(vwap), vwap=quote/base **побарный** | vol=0 → (H+L+C)/3; без окна, без утечки |
| 4 | m_hi_close | (high−close)/close | сырое внутрибарное |
| 5 | m_close_lo | (close−low)/close | сырое внутрибарное |
| 6 | m_aggr_buy_frac | taker_buy_base/base_volume | ∈[0,1]; base=0/NaN → 0.5 (НЕ clip) |
| 7 | m_avg_trade_size | log(base/trades) − log(median_прошл(base/trades)) | окно 1440, trades=0 → 0 |
| 8 | m_rel_volume | log(quote/median_прошл_24h(quote)) | окно 1440, `.shift(1)`, clip ±3 |
| 9..17 | сигналы v7 | entry/trans (булев), buy/sell_margin, bounce_pct, leg_age, exit_flag (булев), leg_dn (булев), regime_code | как есть из state |

Проверено численно на реальном state_v3_causal_2024 (6000 баров):
rel_volume std 0.79 (зажат ±3), avg_trade_size std 0.38, aggr ∈[0.03,0.98],
vwap_ret std 7e-4. Всё конечно (0 NaN).

### Развилки, где НЕ выдумывал число (важно для ревью)
- **vwap** двусмыслен: дизайн-док (стр. 33) трактует его как обычный Δlog рядом
  с close; `binanceenv/observations.py::prepare_vwap` — скользящий с ПАРАМЕТРОМ
  окна (не фикс. числом). Взят **побарный VWAP = quote_asset_volume/base_volume**
  (истинная средневзвешенная цена бара): каузален, без окна → без утечки и без
  выдуманной константы. Зафиксировано в spec как `vwap_mode=
  "per_bar_quote_over_base"` (входит в хэш).
- **окно среднего размера сделки** доком не зафиксировано → взято 1440 (=окно
  относительного объёма) как ВЫБОР РЕАЛИЗАЦИИ; лежит в spec-константах (входит в
  хэш — чтобы смена не осталась незамеченной). Версия остаётся **`v2-draft`**:
  промотировать эти два в VERIFIED нельзя, источник их дословно не фиксирует.
  Правки в `spotrl/spec/observation.py`: `SPEC_CONSTANTS_V2_PROVISIONAL` теперь
  `{vwap_mode, avg_size_median_window_bars=1440}` вместо двух `None`.

### Прокидка колонок (СДЕЛАНО)
`StateDataset` получил ОПЦИОНАЛЬНЫЕ поля: `quote_volume` (из
quote_asset_volume), `trades`, `taker_buy_base`, `buy_margin`, `sell_margin`,
`bounce_pct`, `leg_age` (из leg_age_h); `base_volume`=volume — свойство.
Отсутствие → нейтраль (объёмы NaN → деградация к дока-нейтралям 0.5/0/0;
составляющие v7 → 0). `from_frame`, `attach_signals`, `WindowData.to_state_dataset`
их прокидывают. `make_synthetic_frame` расширен теми же колонками с реальной
вариацией — иначе гейт утечки мерил бы константы. `taker_buy_quote` не берётся.

## 2. Гейт УТЕЧКИ — ГЛАВНАЯ новая проверка (СДЕЛАНО, с числами)

`spotrl/tests/test_market_v2.py::test_market_v2_leak_gate_rolling_features_prefix_equals_full`:
для ~250 сэмплированных точек среза t строим StateDataset ТОЛЬКО из прошлого
[0:t] и сверяем ПОСЛЕДНЮЮ строку рыночного вектора с полным рядом в t — побитово.

- **синтетика (4000 баров, 250 срезов): 0 расхождений**;
- **реальные данные state_v3_causal_2024 (6000 баров, 150 срезов): 0 расхождений**
  (прогнано вручную, число в отчёте не вхолостую).

Зубастость гейта защищена отдельным тестом
`test_market_v2_leak_gate_covers_rolling_columns` (std rel_volume/avg_trade_size
> 1e-4 — точки среза реально попадают в непустые окна, а не в нейтрали).

Плюс блочная каузальность (`test_market_v2_causal_prefix_equals_full`) и
деградация к нейтрали при отсутствии торговых колонок
(`test_market_v2_missing_trade_columns_degrade_to_neutral`).

## 3. Парность источников v2 (СДЕЛАНО для ядра; real-PG — как db-тест)

`spotrl/tests/test_source_parity_v2.py`: рыночное ПОД-ядро v2 (первые 9 —
6 цен + 3 торговых числа) из `ParquetSource` и `PgSource` на одинаковых
extended-колонках == **побитово, 0 расхождений** (быстрый тест, fetcher
инъектирован). Непрерывные составляющие v7 и булевы сигналы в сырой базе
отсутствуют (они только в снимке) — поэтому сверяется ядро, не полный
v2-market-блок (ТЗ это допускает). Реальный поход в PG —
`test_builder_v2_source_parity_real_pg` (`@pytest.mark.db`, честный skip без
базы; НЕ прогонялся здесь — базы в этом заходе не касался).

## 4. Упаковка полного вектора v2 (СДЕЛАНО аддитивно; переключение среды — НЕТ)

`build_observation` теперь диспетчеризует по версии spec; для v2 —
`write_observation_v2` (18 рынок + 14 агент + 3 мир + 3 резерв = 38). Нормировки
по доку: чистый PnL clip ±0.5, one-hot pos_tag (порядок POS_TAG_ORDER_V2),
tanh(bars/2442), cooldown_remain, слот `w_equity_drawdown` = то, что среда
подала в `world.equity_drawdown`. Тесты `test_pack_v2.py` (3): состав/чистота,
one-hot+tanh+clip+слот, pos_tag='none' вне позиции. Геттеры
`agent_state()/world_state()` уже несут все 14 полей (часть 1) — упаковщик их
только читает.

### ПЕРЕКЛЮЧЕНИЕ `EnvConfig.obs_spec` на v2 — НЕ сделано (осознанно)
Это единственная часть, которая реально валит защищённые гейты (T2, T7,
эквивалентность 100k, паритет 6/6), и её нельзя переплетать с готовой работой.
Дефолт остаётся **v1**. Что осталось для переключения (ordered, root cause →
лист):
1. `world_state()` при v2 обязан подать в слот `equity_drawdown` **МИРОВУЮ**
   CB-просадку (`_world_cash/_cb_peak − 1`), а не наградную (сейчас
   `spot_env.py:407` отдаёт наградную). Решение дизайна: impl_state_machine
   207-218 / ТЗ. Свою прибыль агент видит через a_unreal_pnl.
2. Горячий путь `step()` (`spot_env.py:372-376`) жёстко пишет 6 полей агента +
   3 мира через `write_observation`; для v2 нужен параллельный горячий писарь на
   14+расширенный-мир (эталон — геттеры; приёмочный тест — гейт эквивалентности
   100k, он ловит расхождение step vs геттеры).
3. Адаптация env-тестов, индексирующих v1-раскладку (test_t2
   `w_equity_drawdown`, test_t7). Упаковщик и геттеры под это готовы.

## 5. Границы достоверности (честно)
- Гейт утечки, парность ядра источников, упаковка v2 — ПОДТВЕРЖДЕНЫ тестами и
  числами (0 расхождений синтетика+реал для утечки).
- Среда на v2 НЕ переключена → v2-вектор в обучении пока не используется; это
  осознанный изолированный остаток (см. 4).
- `vwap_mode` и окно avg_size — выбор реализации, не догма источника; версия
  `v2-draft` это фиксирует. real-PG парность не прогонялась (нет базы в заходе).
- **Чистота T2 на v2** закрыта только на уровне `build_observation`/
  `write_observation_v2` (test_pack_v2: два вызова побитово равны, агент не
  мутируется), НЕ через `env._obs()` — среда всё ещё v1. Полный env-путь T2-v2
  появится с переключением среды (см. 4).
- Регресс НЕ сломан: сдвиг rng-потока в `make_synthetic_frame` (новые колонки
  рисуются в потоке) проверен полным slow-набором — **6 passed** (t1 parity v7,
  oracle reserve, паритет 6/6); ни один запинованный гейт не покраснел.

## 6. Файлы
- ИЗМЕНЕНЫ: `spotrl/features/builder.py` (+`precompute_market_features_v2`,
  `write_observation_v2`, диспетч `build_observation`), `spotrl/spec/observation.py`
  (константы vwap_mode/avg_size), `spotrl/data/state_dataset.py` (+7 опц. колонок,
  `base_volume`), `spotrl/data/synthetic.py` (extended-колонки), `spotrl/data/sources.py`
  (`to_state_dataset` прокидывает торговые колонки).
- НОВЫЕ ТЕСТЫ: `spotrl/tests/test_market_v2.py` (утечка+каузальность+нейтраль),
  `spotrl/tests/test_source_parity_v2.py`, `spotrl/tests/test_pack_v2.py`.
</content>
