# Часть 1 плана obs v2 — машина скрытого состояния мира в среде (2026-07-25)

Источник истины: `handoff/observation_design_2026-07-25.md` и раздел 2.1
`handoff/impl_obs_v2_2026-07-25.md`. Правились только `spotrl/{spec,features,
envs,data,tests}/`. Коммитов нет. Части 2 (рыночный builder/нормировка) и 3
(утечка/парность источников) в этот заход НЕ делались.

## ГЛАВНЫЙ РЕЗУЛЬТАТ — паритет env↔expert-log (сильнее гейтов 1.2а)

Драйвер `spotrl/tests/parity_env_expert.py` прогоняет СРЕДУ потоком экспертных
действий v7 (повторяет каждый вход/выход) и побарно сверяет
`agent_state()/world_state()` с разделителями из expert-лога. Прогнан на ОБЕИХ
эпохах (env rlbinancetrader, читает готовые артефакты, sunday не нужен):

| Эпоха | сверено баров | in_position | pos_tag | cooldown | entered_on_up | cb |
|-------|---------------|-------------|---------|----------|---------------|--------|
| 2021  | 1 675 738     | **0**       | **0**   | **0**    | **0**         | 7 343 |
| 2024  | 1 240 978     | **0**       | **0**   | **0**    | **0**         | **0** |

Вывод: пять эквити-НЕзависимых разделителей (в позиции, тип сделки dip/transition,
активность cooldown, entered_on_up) среда воспроизводит **бит-в-бит, 0
расхождений** на 1.24M и 1.68M баров. Главный гейт пройден для этих ПЯТИ
разделителей; cb вынесен отдельно (см. ниже). Это доказывает, что СРЕДА кладёт в
наблюдение те же числа, что видел эксперт (гейты 1.2а этого не доказывали —
они мерили разделимость решения эксперта, а не воспроизводимость средой).
Примечание: `entered_on_up` в драйвере — проверка-избыточность (сверяет env с
`not leg_dn[entry_bar−1]` из того же массива), а не независимый разделитель; её
ноль обеспечен тем, что `in_position=0` фиксирует бары входа.

### cb — 2024: полный паритет (0); 2021: 7343 (остаток — базис эквити, часть 3)
После починки логического зазора (см. ниже) cb на **2024 совпал бит-в-бит (0)**,
а на 2021 упал 13 009 → 7 343. cb расходится по ДВУМ причинам, и это важно для
части 3:
1. **Логический зазор (ИСПРАВЛЕН).** v7 достигает проверки cb (regimeb:318)
   только после ДВУХ ранних выходов в next(): `if self.position: return` (:281)
   и `if i <= self._cooldown_until: return` (:306). То есть v7 НЕ оценивает cb
   на барах cooldown. Среда раньше оценивала cb на КАЖДОМ плоском баре, включая
   423-барное окно cooldown после SL, где реализованная эквити просажена и halt
   взводился на ~423 бара раньше v7. Теперь cb гейтится `not in_cooldown`
   (spot_env). Это убрало весь избыток cb 2024 и часть 2021.
2. **Базис эквити (остаётся, часть 3).** Среда: компаунд от 1.0, полная позиция,
   ДВУСТОРОННЯЯ комиссия. v7: портфель 10M, sizing 0.9999, целые лоты,
   ОДНОсторонняя комиссия. cb — пороговое (нелинейное) срабатывание, детерминиро-
   ванный сдвиг комиссии из дизайн-дока его НЕ спасает. Оставшиеся 7343 бара
   2021 — там, где просадка cb-эквити среды пересекает 15% на других барах, чем
   портфель v7. ВАЖНО для части 3: одной общей модели эквити НЕДОСТАТОЧНО без
   пункта 1 — нужно и гейтить cb по `not cooldown_active` (и адаптивный cooldown,
   в P7 он выключен). Логика cb (взвод → halt → снятие по календарному дню с
   переякориванием пика, regimeb:232-256) проверена юнит-тестом.

## Что построено

### Шаг 1 — булевы сигналы v7 в снимке/StateDataset
- `spotrl/data/dump_v7_signals.py` (НОВЫЙ): прогоняет v7 через мост и выгружает
  побарные `entry_signal`(=`_entry`), `trans_entry_signal`(=`_trans_entry`),
  `exit_sig`(=`_exit`), `leg_dn` (каузальный, из state). Запуск в env
  sunday-base-213-tests. Артефакты созданы:
  `/home/cubecloud/Data/rlbinancetrader/v7signals_{2021,2024}.parquet`.
- `spotrl/data/state_dataset.py`: StateDataset получил поля `entry_signal`,
  `trans_entry_signal`, `exit_sig` (по умолчанию массивы False длины n);
  `from_frame` читает их из колонок, если есть; функция `attach_signals`
  подкладывает артефакт с проверкой совпадения индекса (мастер-инвариант «один
  прогон» — тот же источник leg_dn/сигналов в train и serve).

### Шаг 2 — машина состояния в step()/TradeBook (мутация только в step/book)
- `spotrl/envs/tradebook.py`: `OpenTrade` получил `peak_price`, `entered_on_up`,
  `pos_tag`; `TradeBook.open` их принимает; `mark()` ведёт `peak_price`.
- `spotrl/config.py`: `WorldConfig.cooldown_bars = 423` (post-exit cooldown v7,
  ОТДЕЛЬНО от `breaker_cooldown_bars`; в отчёте ранее ошибочно фигурировало 376
  — это класс-дефолт; фактический P7 = **423**).
- `spotrl/envs/spot_env.py`: в `step()` ведутся
  - cooldown: НЕсигнальное закрытие (SL или агентский выход) взводит
    `_cooldown_until = close_bar + 423`; сигнальное — нет (директива
    `_exit_is_signal`, ставится драйвером паритета на штатные выходы v7);
  - circuit breaker: взвод по просадке cb-эквити ≥ порога на ПЛОСКОМ баре,
    снятие по календарному дню (`_dates`), «снят сегодня» (`_cb_cleared_date`);
  - `entered_on_up = not leg_dn[t]` на баре РЕШЕНИЯ (t = entry_bar−1, сверено с
    regimeb:335 и label_exits:69);
  - `pos_tag` = dip при `entry_signal[t]`, иначе transition при `trans_entry[t]`;
  - `peak_price` и откат от пика `price/peak_price − 1`.
  Геттеры `agent_state()/world_state()` только ЧИТАЮТ машину (чистые, Т2).

### Шаг 3 — расширение состояний (частично; полная упаковка v2-вектора — осталось)
- `AgentState` (+`entered_on_up`,`pos_tag`,`price_drawdown`) и `WorldState`
  (+`cb_active`,`cb_cleared_today`,`cooldown_active`,`cooldown_remain`) в
  `spotrl/features/builder.py` расширены с дефолтами — v1-сборка не тронута.
- `spotrl/spec/observation.py`: `cooldown_len_bars=423` переведён из
  PROVISIONAL в VERIFIED (подтверждён машиной и гейтом: 0 расхождений cooldown).

## Тесты и покрытие
- Новые: `spotrl/tests/test_state_machine.py` (8 тестов: cooldown после SL,
  сигнальный/агентский выход, dip/transition, entered_on_up на баре решения,
  откат от пика, календарный сброс cb, чистота геттеров); 2 теста сигналов
  StateDataset в `test_data_and_features.py`.
- Регресс главного гейта: `spotrl/tests/test_parity_env_expert.py` (помечен
  `@pytest.mark.slow`, вне быстрого набора; скип без артефактов) — фиксирует
  5 разделителей = 0 на обеих эпохах и cb=0 на 2024. Прогон: **3 passed, 35.6 c**.
- Быстрый набор: **111 passed, 4 skipped, 8 deselected**, ~4 c (бюджет ≤60 c).
- Покрытие spec/features/envs = **98%** (порог ≥85%): spot_env 99%,
  builder 100%, tradebook 100%, observation 98%.
- Т2 (чистота) и все прежние гейты Т1–Т8, Э1.1/Э1.2 зелёные.

## Что осталось (честно)
1. **Полная упаковка v2-вектора (38) и переключение `EnvConfig.obs_spec` на v2.**
   Сделан НЕ до конца намеренно: главный гейт сверяет `agent_state()/world_state()`,
   а не упакованный Box-вектор, и не требует упаковки. Переключение дефолта на v2
   сломало бы горячий путь `step()` (жёстко пишет 6 v1-полей агента) и прежние
   env-тесты — это отдельный этап: новый writer на 14 полей агента + нормировки
   (tanh(bars/2442), clip PnL ±0.5, one-hot pos_tag) + адаптация тестов. Машина и
   геттеры под это готовы; дефолт пока v1.
2. **Числовой паритет cb на 2021** (7343 бара; 2024 уже = 0) — требует общей
   модели эквити с v7 ПЛЮС гейта cb по `not cooldown_active` (см. раздел cb),
   относится к части 3.
3. Части 2 (рыночный builder/нормировка) и 3 (утечка/парность источников) —
   как в плане, не в этот заход.

## Как воспроизвести паритет
```
# 1) артефакты сигналов (env sunday-base-213-tests) — уже созданы:
python -m spotrl.data.dump_v7_signals <state.parquet> --out <v7signals.parquet>
# 2) гейт паритета (env rlbinancetrader):
python -m spotrl.tests.parity_env_expert
```

## Файлы
- НОВЫЕ: `spotrl/data/dump_v7_signals.py`, `spotrl/tests/parity_env_expert.py`,
  `spotrl/tests/test_parity_env_expert.py`, `spotrl/tests/test_state_machine.py`.
- ИЗМЕНЕНЫ: `spotrl/data/state_dataset.py`, `spotrl/envs/tradebook.py`,
  `spotrl/envs/spot_env.py`, `spotrl/config.py`, `spotrl/features/builder.py`,
  `spotrl/spec/observation.py`, `spotrl/tests/test_data_and_features.py`.
- READ-ONLY (для чисел): `strategy_rl/{v7hook,v7runner,label_exits}.py`,
  `sunday .../regimeb_bt_strategy.py`, артефакты
  `/home/cubecloud/Data/rlbinancetrader/`.
```
