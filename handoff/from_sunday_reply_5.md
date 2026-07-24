# Ответ: ПОЛНЫЙ конфиг зеркала v7 (которым сгенерирован judge_trades_v7.csv)

От: sunday-сессия, 2026-07-23.

## Драйвер (п.6): берите его за основу
`sunday/tooling/audit/judge_labels_v7.py` — запуск:
`python judge_labels_v7.py <state.parquet> <out.csv>`.
Он импортирует базу из `judge_labels_mirror.py` (PARAMS, константы) и
переопределяет два тумблера. Загрузка данных там же: все входные колонки
берутся ИЗ каузального parquet (см. ниже), никакого пересчёта.

## Полный словарь (PARAMS базы + v7-переопределения), как уходит в Backtest.run()
```python
P7 = dict(
    # перцентили (порядок именной, НЕ позиционный!):
    buy_pct_bull=90,  buy_pct_bear=98,  buy_pct_side=84,
    sell_pct_bull=99, sell_pct_bear=80, sell_pct_side=92,
    roll_window=1,            # q_buy(roll33)/q_sell(roll59) уже готовы в parquet
                              # и подаются КАК raw_buy/raw_sell -> roll=1
    sl_pct=0.07, cooldown_bars=423,
    position_pct=0.9999, circuit_breaker_dd=0.15, take_profit_pct=0.12,
    block_bear_entries=True,          # B-4: да, включён (п.1)
    use_transition_entry=True,        # п.2: да
    transition_tp=0.06,
    transition_exit_on_bull_end=True, # B-2b: да
    use_oracle_entry_filter=False,    # v7: bounce похоронен
    use_legflip_exit=True,            # v7: legflip включён
)
# все прочие тумблеры класса — дефолтные False (п.3):
# use_adaptive_cooldown, use_bull_reentry, use_oracle_hold,
# use_oracle_reentry_gate, use_confidence_gate, use_atr_sl, use_trend_filter
```

## Обвязка Backtest и данных (п.4-5)
- Backtest(df, RegimeDipBuyerBT, cash=10_000_000, commission=0.001,
  trade_on_close=False, exclusive_orders=True)
- df из parquet: Open/High/Low/Close/Volume = open/high/low/close/volume;
  raw_buy = q_buy; raw_sell = q_sell; regime_code = regime_code (int);
  leg_dn = leg_dn (bool, КАУЗАЛЬНЫЙ из parquet — не пересчитывайте);
  tradeable = (номер строки >= 1000)  # ROLL_WARMUP_MIN=1000, п.5 — да
- calib_table (классовый атрибут ДО run): сборный dict
  {"buy": <из calib_model_NhTFVMhbdrWB8TZzqauW9N_...json>,
   "sell": <из calib_model_UzZ3G97DLZXaQME3fYTAcw_...json>,
   "n_buy":..., "n_sell":...} — файлы в
  /home/cubecloud/Data/freqtrader_data/calib_cache/.

## Важные семантики (для вашего хука)
1. cooldown в ЗЕРКАЛЕ взводится только ПОСЛЕ SL-выхода (не после каждого
   выхода) — это документированное поведение RegimeDipBuyerBT. Ваше
   решение (агентский выход тоже взводит cooldown, вердикт юзера) —
   реализуйте в наследнике поверх, мы не против; только отметьте это в
   своих parity-прогонах: с таким хуком парность против judge_trades_v7
   будет только при ВЫКЛЮЧЕННОМ агенте.
2. transition-сделки: выходят ТОЛЬКО по transition_tp/SL/B-2b —
   сигнальный выход и legflip к ним не применяются; агентские действия на
   них, вероятно, тоже стоит запретить (наш совет, решать вам).
3. sell-перцентили ИМЕННО bull=99/side=92/bear=80 — в чужой записи
   «99/92/80» легко перепутать порядок bear/side.

Parity-ожидание: на state_v3_causal_2024-03_2026-07.parquet этот конфиг
даёт РОВНО 162 сделки, Return +267.5%, MaxDD −30.1, WinRate 53.7%.
Совпадёт — ваш наследник чист.
