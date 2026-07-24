"""Общие фикстуры: синтетический ряд и среда без внешних данных."""
from __future__ import annotations

import pytest

from spotrl.config import ActionSpec, EnvConfig, FreedomConfig, WorldConfig
from spotrl.data.synthetic import make_synthetic_state
from spotrl.envs.spot_env import SpotFlipEnv

# Путь к эталонным данным sunday: если их нет, медленные тесты пропускаются
REF_STATE = "~/Data/sunday_tests/state_v0/state_v3_causal_2024-03_2026-07.parquet"
REF_TRADES = "~/Data/sunday_tests/state_v0/judge_trades_v7.csv"


@pytest.fixture()
def state():
    """Синтетический StateDataset из 5 000 баров (детерминированный)."""
    return make_synthetic_state(n_bars=5_000, seed=0)


@pytest.fixture()
def env_config():
    """Конфигурация среды с полной свободой агента и корзинами длины 2."""
    return EnvConfig(world=WorldConfig(stop_loss_frac=0.05),
                     freedom=FreedomConfig(exit_own=True, entry_own=True,
                                           params_own=True),
                     action_spec=ActionSpec(sl_buckets=(0.03, 0.05),
                                            tp_buckets=(0.10, 0.20),
                                            expert_sl_index=1,
                                            expert_tp_index=0),
                     episode_len=2_000)


@pytest.fixture()
def env(state, env_config):
    """Готовая среда на синтетическом ряде."""
    return SpotFlipEnv(state, env_config)
