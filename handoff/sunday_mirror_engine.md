# Зеркальный движок (симулятор для наград) и его parity

## Что это
Зеркало боевой стратегии v6 (RegimeDipBuyerC2B3, сейчас торгует в paper) на
backtesting.py. Ключевое свойство: при заморозке пакета v6 (2026-07-19)
доказана парность с freqtrade БИТ-В-БИТ на 3 фолдах (совпадение сделок и P&L).
Это кандидат на РОЛЬ СРЕДЫ/СИМУЛЯТОРА для RL-наград: можно проигрывать
альтернативные действия на истории.

## Файлы (в репо sunday, /home/cubecloud/Python/projects/sunday)
- `tooling/audit/regimeb_perfold15k/regimeb_bt_strategy.py` — класс RegimeDipBuyerBT
  со всеми тумблерами пакета: block_bear_entries (B-4), use_oracle_entry_filter
  (bounce через колонку oracle_entry_ok), use_legflip_exit (колонка leg_dn),
  use_transition_entry + transition_exit_on_bull_end (B-2/B-2b).
- `tooling/audit/judge_labels_mirror.py` — рабочий пример полного прогона:
  сборка входного df ИЗ STATE-parquet (готовые q_buy/q_sell подаются как raw_*
  с roll_window=1), сборная calib-таблица из двух frozen JSON, PARAMS пакета v6.
- calib JSON: `/home/cubecloud/Data/freqtrader_data/calib_cache/`
  `calib_model_NhTFVMhbdrWB8TZzqauW9N_202101011200_202403100500.json` (buy)
  `calib_model_UzZ3G97DLZXaQME3fYTAcw_202101011200_202403100500.json` (sell)

## Параметры v6 (= боевой json, sha256 в манифесте prereg_c2b3_paper_2026-07-19.md)
buy: bull p90 / side p84 / bear p98; sell: bull p99 / side p92 / bear p80;
SL −7%; TP 12% (dip) / 6% (transition); cooldown 423 мин; CB 15%; стейк 100%;
входы: dip (сигнал+bounce≥1%+не bear) и transition; выходы: sell-сигнал/SL/TP/
legflip/B-2b.

## Контроль тождественности (2026-07-21, по фолдовым границам манифеста)
Наш конвейер: +84.0/+47.2/+13.4/+27.7/+37.5 против зеркального эталона
заморозки +79.6/+43.1/+12.1/+25.3/+35.2. Систематическое +1.5-4 п.п. — причина
известна: нога зигзага/сглаживания считаются на НЕПРЕРЫВНОЙ полосе, а эталон
считал по-фолдово с 45д префиксом (у online-зигзага длинная память). Для
сравнений вариантов МЕЖДУ СОБОЙ на одной полосе — некритично; для строгого
бит-в-бит с эталоном — стройте state по-фолдово с 45д префиксами.

## Скорость
Полный прогон 1.24 млн минутных баров — минуты на CPU. Полный цикл
«датасет → метки → модель → walk-forward» укладывался в часы.
