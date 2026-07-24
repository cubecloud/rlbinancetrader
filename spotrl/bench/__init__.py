"""bench/ — замеры скорости среды и гейт эквивалентности прогона.

Что здесь: инструменты, которые запускаются РУКАМИ и печатают числа —
скорость шага (сырая среда и векторизаторы) и побитовое сравнение прогона
на 100 000 шагов до и после правок горячего пути.

Чего здесь НЕТ: логики среды, наград и признаков. Модуль только вызывает
публичный API `spotrl` и меряет; в тракт обучения он не входит.
"""
from __future__ import annotations

from spotrl.bench.equivalence import collect_runs, compare, scenarios
from spotrl.bench.speed import measure_raw_env, measure_vec_env

__all__ = ["collect_runs", "compare", "scenarios",
           "measure_raw_env", "measure_vec_env"]
