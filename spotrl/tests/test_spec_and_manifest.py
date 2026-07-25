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


def test_observation_spec_v2_composition():
    """Состав v2: блоки в порядке рынок→агент→мир→резерв, размер и хэш стабильны."""
    spec = ObservationSpec.v2()
    assert spec.version == "v2"            # финализирована: все константы зафиксированы
    assert len(spec.market) == 18          # 6 цен + 3 торговых числа + 9 сигналов v7
    assert len(spec.agent) == 14
    assert len(spec.world_rules) == 3
    assert len(spec.reserved) == 3
    assert spec.size == 38
    assert spec.constants["pos_tag_order"] == ["none", "dip", "transition"]
    # финализированные (бывшие provisional) константы зафиксированы и в хэше
    assert spec.constants["vwap_mode"] == "per_bar_quote_over_base"
    assert spec.constants["avg_size_median_window_bars"] == 24 * 60
    # текущий leg_dn — определяющий разделитель гейта выхода — присутствует
    assert spec.index_of("m_leg_dn") >= 0
    names = spec.names
    assert (names.index("a_pos_tag_none") < names.index("a_pos_tag_dip")
            < names.index("a_pos_tag_transition"))
    assert spec.index_of("a_unreal_pnl") >= 0
    assert spec.index_of("a_days_in_trade") >= 0
    assert spec.spec_hash() == ObservationSpec.v2().spec_hash()
    assert spec.spec_hash() != ObservationSpec().spec_hash()


def test_observation_spec_v2_constants_in_manifest():
    """Манифест v2 несёт константы нормировки и хэш спецификации (не μ/σ)."""
    d = ObservationSpec.v2().describe()
    assert d["obs_spec_version"] == "v2"
    assert d["constants"]["median_hold_bars"] == 2442
    assert d["constants"]["clip_rel_volume"] == 3.0
    assert d["constants"]["rolling_shift"] == 1
    assert len(d["spec_hash"]) == 64


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
    # cb_equity_mode принимает только два режима (леса копирования); опечатка
    # не должна тихо уйти в ветку копирования.
    with pytest.raises(ValueError):
        WorldConfig(cb_equity_mode="rewrad")
    assert WorldConfig(cb_equity_mode="reward").cb_equity_mode == "reward"
    assert WorldConfig().describe()["cb_equity_mode"] == "v7_ledger"


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
