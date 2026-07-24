# Слой данных: parquet + PostgreSQL за одним интерфейсом (2026-07-24)

От: rlbinancetrader-сессия (реализация). Код только в `spotrl/`. Коммитов нет.

## 0. Редакция 2026-07-25 (обобщение интерфейса под выбор колонок)

Уточнение пользователя, подтверждённое запросом к живой базе: база содержит все
нужные СЫРЫЕ колонки, их надо просто запрашивать. С `use_extended_cols=True`
датафетчер отдаёт 9 колонок: OHLCV + `quote_asset_volume`, `trades`,
`taker_buy_base`, `taker_buy_quote` (поток ордеров). Признаки
`q_buy/q_sell/regime_code/leg_dn` — ПРОИЗВОДНЫЕ стратегии v7, в сырой базе их нет
и не должно быть; они нужны только на этапе копирования v7 и приходят из снимка.

Что изменено против первой редакции (разделы 2-3 ниже описывают старую форму
signals/regime_code/leg_dn — оставлены для истории; действующий код такой):

1. **Набор колонок — параметр источника, не хардкод.** `PgSource(columns=None)`
   по умолчанию берёт extended-набор (`use_extended_cols=True`); можно задать
   явный `columns=(...)`. Производные v7 из обязательных для PG убраны.
2. **`WindowData` обобщён**: вместо жёстких полей — именованная матрица
   `columns: tuple[str,...]` + `data: (n,k) float64`, плюс срез-свойство
   `ohlcv`, доступ `column(name)`/`select(names)`/`has(name)`. `ParquetSource`
   по умолчанию отдаёт полный state-набор снимка (OHLCV + v7), но принимает
   `columns=` (например extended). `to_state_dataset()` — только когда есть v7.
3. **Гейт `bitwise_gate(a, b)`** сравнивает ПЕРЕСЕЧЕНИЕ колонок по имени; в
   `columns_compared` — что реально сверено, в `columns_only_*` — что уникально.
4. **Манифест** пишет список выбранных колонок и их источник.

### Результат ГЛАВНОГО ГЕЙТА (редакция 2026-07-25, прогнано на этой машине)

Снимок (`columns=extended`) vs живая база (extended), окно 24 ч
`2024-03-10 05:00 .. 2024-03-11 05:00`:

```
GATE common_bars=1440
 cols=[open, high, low, close, volume, quote_asset_volume, trades,
       taker_buy_base, taker_buy_quote]
 diff=0  max_abs=0.0
 per_col={все 9 колонок: 0}
 only_parquet=1  only_pg=0  first=None
```

**Все 9 сырых колонок (OHLCV + поток ордеров) совпадают ПОБИТОВО** — ноль
различий, max|a−b|=0.0. Order-flow из базы прочитан и не содержит NaN
(проверяется в тесте). Единственное расхождение — 1 граничный бар (эффект
`last_full_bar`), вынесен в `only_parquet=1`, значений не касается. Пересечение
теперь богаче OHLCV: производные v7 в пересечение не входят (их в PG нет) — это
корректно, а не пробел.

Тесты: 13 быстрых (2.4-3 с), покрытие `sources.py` **90%** (порог 85%), db-гейт и
slow на реальном снимке — зелёные. Ниже — исходная (устаревшая по форме полей)
записка; механика чтения базы, ключей и края в ней актуальна.

## 1. Что нашёл про доступ к базе (разведка, первичные источники)

- Чтение базы в репозитории идёт через `dbbinance`:
  `datawizard/dataprocessor.py:168-179` вызывает
  `self.fetcher.resample_to_timeframe(...)`, fetcher создаётся через
  `dbbinance.fetcher.getfetcher.get_datafetcher()` (read-only, PG-креды из
  `ConfigPostgreSQL`, binance-ключи — заглушки `"dummy"`).
- `resample_to_timeframe` (`datafetcher.py:1255`) — это back-compat алиас,
  делегирует в `pg_resample_to_timeframe` (`:1432`): ресемпл на стороне SQL.
  Каузальность «последнего ЗАКРЫТОГО бара» обеспечивает `last_full_bar=True`
  (по умолчанию) → `_drop_partial_tail` (`:1152`) отбрасывает хвостовой бин,
  у которого `label + interval > end`. Это и есть NaT-guard живого края.
- ЭТАЛОН параметров чтения OHLCV — не dataprocessor, а БИЛДЕР СНИМКА
  `sunday/tooling/audit/state_v0_builder.py:79-85`:
  `pg_resample_to_timeframe(table_name="spot_data_btcusdt_1m",
  to_timeframe="1m", origin="start", open_time_index=True,
  use_cols=Constants.binance_extended_cols, use_dtypes=...)`, затем
  `df[~df.index.duplicated(keep="last")].sort_index()` и индекс
  `tz_convert("UTC").tz_localize(None)`. Именно эти параметры воспроизводит
  `PgSource`, иначе побитовое равенство OHLCV невозможно.
- Своё подключение НЕ изобретал: `PgSource` ходит через готовый
  `get_datafetcher()`.

### Находка про интерактивный SALT-ввод (требование 6) — подтверждено эмпирически
Импорт `dbbinance.fetcher.*` тянет `dbbinance.config.configpostgresql`, который
на уровне модуля создаёт `secureapikey.Secure()` и печатает `Enter the SALT
phrase: ...`. При закрытом stdin получает EOF и продолжает (fallback на env),
но в живом tty без ввода — БЛОКИРУЕТСЯ (`input()`/`getpass` не бросают, а
ждут). Проверка: `timeout 15 python -c "import
dbbinance.fetcher.getfetcher" </dev/null` → печатает промпт, но `exit=0`.
Реальное подключение здесь падает `psycopg2 ... no password supplied` — то есть
в этом headless-окружении **PG недоступна** (кредов нет).

Вывод для дизайна: импорт слоя данных НЕ должен трогать `dbbinance`. В
`PgSource` импорт ленивый — только внутри `_get_fetcher`/`_use_cols`, и на пути
с инъектированным fetcher (тесты) `dbbinance` не импортируется вовсе. Путь
parquet вообще не знает про `dbbinance`.

## 2. Что реализовал (`spotrl/data/sources.py`)

- **Единый интерфейс** `DataSource.load_window(start, end) -> WindowData`.
  `WindowData` — плотный результат: `index` (tz-naive UTC), `ohlcv` (n,5)
  float64 — всегда; `signals`/`regime_code`/`leg_dn` — либо массивы, либо
  `None`; `source`, `meta`. Методы: `has_features()`, `ohlcv_hash()`,
  `to_state_dataset()` (падает, если признаков нет).
- **ParquetSource** — обёртка над `state_dataset.load_state`; срез окна по
  индексу (границы включительно), отдаёт OHLCV И признаки (снимок собран
  билдером). Не импортирует `dbbinance`.
- **PgSource** — база через `dbbinance`. `load_window` повторяет параметры
  билдера (таблица, `to_timeframe`, `origin="start"`, дедуп, tz-naive UTC),
  отдаёт ТОЛЬКО OHLCV (`signals=None`). `last_full_bar=True` по умолчанию.
  `load_latest_closed(now, lookback_bars)` — онлайн-дочитывание последних
  ЗАКРЫТЫХ баров с assert'ом `last_label + freq <= now` (неполный текущий бар
  не попадает). Импорт `dbbinance` ленивый; поддержана инъекция `fetcher` для
  тестов без базы.
### 2.1 Загрузка ключей PG (правка по решению пользователя)
`PgSource._load_secrets()` перед импортом `dbbinance` грузит `PSGSQL_KEY.env` и
`PSGSQLKEYS.env` из каталога sunday (по умолчанию `~/Python/projects/sunday`,
переопределяется `env_dir=` или `$SPOTRL_PG_ENV_DIR` — абсолют не захардкожен).
После этого `secureapikey` расшифровывает доступ БЕЗ интерактивного SALT-ввода.
Значения ключей нигде не логируются и в проект не копируются — только читаются
из sunday. Загрузка ленивая: только в `_get_fetcher`, parquet-путь её не
касается. Файлы .env в git не коммитятся.

- **Гейт** `ohlcv_bitwise_gate(a, b) -> GateResult`: сравнивает OHLCV на
  ПЕРЕСЕЧЕНИИ меток (рассинхрон края не маскирует расхождение значений);
  элемент-в-элемент. Диагностика: `diff_elements`, `max_abs_diff`,
  `per_column`, `first_mismatch=(row, col, a, b)`, `n_only_a`/`n_only_b`
  (граничные бары, уникальные для источника). `passed` = есть общие бары и ноль
  различий OHLCV на пересечении; граничный бар `last_full_bar` гейт не валит.
  Флаг `index_equal` доступен отдельно как строгая сверка.
- **Манифест**: `source_manifest(win)` — версия dbbinance
  (`importlib.metadata.version('dbbinance-storage')`, у пакета нет
  `__version__`), ВЕРСИЯ БИЛДЕРА признаков (`feature_builder_version="state_v0"`),
  параметры чтения, границы окна, `sha256` OHLCV,
  список признаков, отсутствующих в сыром PG.

## 3. Результат гейта эквивалентности (НАСТОЯЩЕЕ ЧИСЛО, прогнано на этой машине)

Ключи PG загружены `PgSource` из `sunday/*.env` (см. раздел 2.1), БД ответила,
гейт прогнан НА САМОМ ДЕЛЕ (parquet-снимок vs живая база), окно 24 часа
`2024-03-10 05:00 .. 2024-03-11 05:00`:

```
GATE common=1440 diff=0 max_abs=0.0
     per_col={'open':0,'high':0,'low':0,'close':0,'volume':0}
     only_parquet=1 only_pg=0 first=None
```

Проверено также на окнах 7 дней (common=10080) и 1 месяц (common=44640) —
везде `diff=0, max_abs=0.0`.

- **OHLCV совпадает ПОБИТОВО**: ноль различающихся элементов из 5 колонок ×
  1440 (и ×10080, ×44640) баров, максимум |a−b| = 0.0. Это прямой тест
  train/serve-эквивалентности источников — пройден.
- **Единственное расхождение — 1 граничный бар**: parquet содержит бар на
  правой границе окна (метка `05:00` следующих суток, 1441-й), а `PgSource` его
  отбрасывает через `last_full_bar=True` (бар «закрыт» только в момент
  `05:01 > end`). Это НЕ ошибка данных, а корректная семантика «последнего
  закрытого бара» (П7). Гейт считает по ПЕРЕСЕЧЕНИЮ меток и выносит граничный
  бар в `only_parquet=1` как диагностику, а не как расхождение значений.
- **Структурное число** (что вообще сравнимо): из 9 обязательных колонок
  `state_dataset.REQUIRED_COLUMNS` — **5 (open/high/low/close/volume)
  воспроизводимы из сырого PG побитово**, **4 (q_buy, q_sell, regime_code,
  leg_dn) в сырой базе ОТСУТСТВУЮТ** — их считает сигнальный тракт стратегии
  (`state_v0_builder.py`: головы моделей, `regime_3voter`, зигзаг-нога 1h).
- **Что совпадает побитово**: OHLCV (при условии, что `PgSource` использует те
  же параметры чтения, что и билдер — они зафиксированы в коде и в манифесте).
  Логика гейта проверена юнит-тестами на общих массивах: идентичный OHLCV →
  `diff=0`; сдвиг на 1e-9 в одной ячейке → `diff=1`, указана колонка;
  рассинхрон окна → `index_equal=False`.
- **Что НЕ совпадает и почему**: 4 сигнальных признака — их в сырой базе нет.
  Как закрыть: материализовать их тем же билдером
  (`state_v0_builder.regime_3voter`, головы, зигзаг) поверх OHLCV из `PgSource`
  — тогда онлайн-путь даст полный `StateDataset`. Это архитектурная развилка,
  вынесена пользователю (раздел 5).

## 4. Тесты (`spotrl/tests/test_data_sources.py`)

Быстрый набор — 9 тестов, вся сюита 2.4 с (порог ≤60 с соблюдён), не ходят в
базу и не импортируют `dbbinance`:
- ParquetSource: формы, признаки, `to_state_dataset`, включительные границы,
  манифест (хэш + версия + список отсутствующих признаков);
- изоляция импорта (требование 6): parquet работает при заблокированном
  `dbbinance`/`secureapikey` (monkeypatch `sys.modules` + `__import__`),
  конструктор `PgSource()` не импортирует базу;
- PgSource с инъектированным fetcher: только OHLCV, tz-naive, `to_state_dataset`
  падает; живой край — неполный бар 05:09 при `now=05:09:30` отброшен, последний
  закрытый 05:08;
- гейт: 0-diff, ловля 1-битового расхождения, рассинхрон индекса.

Реальный PG-гейт — `@pytest.mark.db` (`test_pg_vs_parquet_bitwise_gate`):
прогнан на этой машине, `common=1440 diff=0 max_abs=0.0` (см. раздел 3);
при отсутствии базы/ключей — честный skip. Slow-тест на РЕАЛЬНОМ снимке
(`test_real_parquet_snapshot_loads`) прогнан: 1441 бар, признаки, float64,
tz-naive монотонный индекс, `to_state_dataset()` собирается. Маркеры `slow`/`db`
в `spotrl/pytest.ini`, `addopts` исключает их из быстрого набора.

Покрытие `spotrl/data/sources.py` — **89%** (`pytest --cov`), выше порога 85%
из PLAN 1.5.2. Непокрытое — ветки только-реального-PG и обработчики ошибок
(они закрываются db-тестом отдельно).

## 5. Что требует решения пользователя / правок ЧУЖИХ файлов (списком, не внесено)

1. **Онлайн-путь paper/live**: `PgSource` даёт только OHLCV. Для среды нужны
   q_buy/q_sell/regime_code/leg_dn. Развилка: (а) сделать
   `sunday/tooling/audit/state_v0_builder.py` импортируемым модулем (сейчас
   это скрипт с `main()`) и вызывать его билдер поверх OHLCV из `PgSource`;
   (б) оставить `PgSource` как OHLCV-only и материализовать признаки отдельным
   шагом. Рекомендация: (а) — единый билдер = train/serve-parity признаков.
2. **`dbbinance`**: добавить `__version__` (сейчас только через
   `importlib.metadata`) — уже отмечено в
   `handoff/for_sunday_dbbinance_1_0_10`.
3. **`secureapikey`/`configpostgresql`**: интерактивный SALT-ввод на импорте
   опасен для headless. Не правил (чужой код). Обходим так: `PgSource` грузит
   ключи из sunday/*.env ДО импорта dbbinance (раздел 2.1), плюс ленивый импорт.
   Для чужой машины путь к .env задаётся `env_dir=`/`$SPOTRL_PG_ENV_DIR`.
4. **conftest `REF_STATE`** указывает на `state_v3_causal_2024-03_2026-07.parquet`,
   которого на диске НЕТ — есть `state_v0_2024-03_2026-07.parquet`. Мои тесты
   используют реально существующий файл. Путь в conftest стоит согласовать
   (иначе slow/db-тесты скипаются «нет файла», а не «нет базы»).
