"""Проверки спецификаций, скейлера и манифеста прогона."""
from __future__ import annotations

import numpy as np
import pytest

from spotrl.config import EnvConfig, WorldConfig
from spotrl.features.scaler import ObsScaler
from spotrl.runs.manifest import RunManifest, compare_world_version
from spotrl.spec.actions import ActionSpec, FLIP, STAY
from spotrl.spec.observation import ObservationSpec


def test_expert_action_is_representable():
    """Константа эксперта обязана быть выразима индексом корзины (PLAN 3.5.3)."""
    spec = ActionSpec(sl_buckets=(0.03, 0.05), tp_buckets=(0.10,),
                      expert_sl_index=1, expert_tp_index=0)
    action = spec.expert_action(FLIP)
    assert list(action) == [FLIP, 1, 0]
    assert spec.decode(action).sl_frac == 0.05


def test_action_spec_validation():
    """Некорректные корзины отвергаются на границе."""
    with pytest.raises(ValueError):
        ActionSpec(sl_buckets=(0.05, 0.03))
    with pytest.raises(ValueError):
        ActionSpec(sl_buckets=(), tp_buckets=(0.1,))
    with pytest.raises(ValueError):
        ActionSpec(expert_sl_index=5)


def test_decode_rejects_bad_head():
    """Недопустимый индекс головы позиции — ошибка, а не молчание."""
    spec = ActionSpec()
    with pytest.raises(ValueError):
        spec.decode([2, 0, 0])
    with pytest.raises(ValueError):
        spec.decode([STAY, 0])


def test_observation_spec_has_reserved_slots():
    """В наблюдении есть три зарезервированных слота (PLAN 4.5.2)."""
    spec = ObservationSpec()
    assert len(spec.reserved) == 3
    assert spec.size == len(spec.names)
    assert spec.index_of("rule_slot_0") >= 0
    with pytest.raises(KeyError):
        spec.index_of("no_such_feature")


def test_scaler_hash_and_roundtrip(tmp_path):
    """Скейлер сериализуется с хэшем и проверяет его при загрузке."""
    data = np.random.default_rng(0).normal(size=(500, 8))
    scaler = ObsScaler.fit(data)
    path = tmp_path / "scaler.json"
    digest = scaler.save(path)
    loaded = ObsScaler.load(path)
    assert loaded.sha256 == digest
    assert np.allclose(loaded.transform(data[:5]), scaler.transform(data[:5]))


def test_world_config_validation():
    """Правила мира валидируются на границе (PLAN 1.5.1, п.2)."""
    with pytest.raises(ValueError):
        WorldConfig(stop_loss_frac=1.5)
    with pytest.raises(ValueError):
        WorldConfig(max_data_age_bars=0)


def test_manifest_carries_world_version(tmp_path):
    """world_version попадает в манифест и блокирует сравнение разных миров."""
    left = RunManifest(run_id="a", seed=1, env_config=EnvConfig())
    right = RunManifest(run_id="b", seed=1,
                        env_config=EnvConfig(world=WorldConfig(world_version="w2")))
    assert compare_world_version(left, left) is None
    assert "world_version" in compare_world_version(left, right)
    payload = left.to_dict()
    assert payload["env"]["world"]["world_version"] == "w1"
    assert left.save(tmp_path / "manifest.json").exists()
