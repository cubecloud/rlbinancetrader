"""Заглушки проверок, которые невозможно написать до появления кода.

Каждый skip обязан нести причину (PLAN 1.5.3: ни одного skip без причины).
"""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="нужен наследник PPO со своей дистрибуцией (PLAN 3.7.3) — "
                         "модуль algo/ppo_headmask.py ещё не написан")
def test_ppo_head_masked_update():
    """Апдейт PPO с per-head loss-masking не меняет веса замороженных голов."""


@pytest.mark.skip(reason="замер пропускной способности — отдельный бенчмарк-скрипт "
                         "(гейты 754k шаг/с сырой среды и 30.5k шаг/с обучения, PLAN 1.5.2)")
def test_env_throughput():
    """Сырая среда держит не менее 754 000 шаг/с."""


@pytest.mark.skip(reason="BC-датасет ещё не собран: нужна разметка эксперта v7 "
                         "по барам (label_exits + разметка входов, PLAN волна 3)")
def test_bc_dataset_action_coverage():
    """В BC-датасете представлены оба значения головы позиции."""


@pytest.mark.skip(reason="корзины SL/TP задаются офлайн-перебором (волна 2); "
                         "до него границы корзин — заглушка")
def test_sl_tp_buckets_cover_expert_constants():
    """Константы SL/TP эксперта попадают в корзины точно, а не приближённо."""
