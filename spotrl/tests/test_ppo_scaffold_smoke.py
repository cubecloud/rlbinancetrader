"""SMOKE-тест каркаса PPO: связка семя reg_cd_vw2 + маска dip + KL-якорь
стартует и прогоняет learn(100) на dummy-env без падения. НЕ обучение.

Требует сохранённого семени reg_cd_vw2 (bias-init критика). Если его нет —
skip (семя создаётся CLI spotrl.bc.value_biasinit).
"""
from __future__ import annotations

from pathlib import Path

import pytest

_SEED = "/home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy_reg_cd_vw2"
_MF = _SEED + ".manifest.json"


@pytest.mark.skipif(not Path(_SEED + ".zip").exists(),
                    reason="нет семени reg_cd_vw2 (создаётся value_biasinit)")
def test_scaffold_smoke_starts_and_runs():
    """Каркас стартует, learn(100) не падает, gamma==1, клон заморожен."""
    from spotrl.algo.ppo_scaffold import smoke_scaffold
    r = smoke_scaffold(_SEED, _MF, kl_coef=0.1, ent_coef=0.01, n_steps=100)
    assert r["started"] is True
    assert r["gamma"] == 1.0            # обязателен (EnvConfig.gamma)
    assert r["clone_frozen"] is True    # KL-якорь заморожен


@pytest.mark.skipif(not Path(_SEED + ".zip").exists(),
                    reason="нет семени reg_cd_vw2")
def test_scaffold_clone_is_frozen_deepcopy():
    """Клон-якорь — глубокая копия, ни один параметр не требует градиента."""
    from spotrl.algo.ppo_scaffold import build_scaffold, _dummy_vec_env
    env = _dummy_vec_env()
    model, clone = build_scaffold(env, _SEED, _MF, kl_coef=0.1, ent_coef=0.01,
                                  gamma=1.0, n_steps=64, batch_size=64, n_epochs=1)
    assert all(not p.requires_grad for p in clone.parameters())
    # клон отдельный объект (не тот же тензор, что у обучаемой политики).
    assert clone is not model.policy


def test_seed_scaler_matches_apply_scaler():
    """Обёртка семени = apply_scaler(obs, mu, sd) бар-в-бар (train/serve parity)."""
    import numpy as np
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv
    from spotrl.algo.ppo_scaffold import make_seed_scaler_vecenv
    from spotrl.bc.train_clone import apply_scaler

    mu = np.array([0.5, 0.4, 0.0], np.float32)
    sd = np.array([0.5, 0.49, 1.0], np.float32)
    raw = (np.random.randn(3) * sd + mu).astype(np.float32)

    class _E(gym.Env):
        """Среда с фиксированным сырым наблюдением."""

        def __init__(self):
            """Пространства под скейлер длины 3."""
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (3,),
                                                    np.float32)
            self.action_space = gym.spaces.MultiDiscrete([2, 1, 1])

        def reset(self, *, seed=None, options=None):
            """Вернуть сырое наблюдение."""
            return raw.copy(), {}

        def step(self, action):
            """Шаг-заглушка."""
            return raw.copy(), 0.0, False, True, {}

    w = make_seed_scaler_vecenv(DummyVecEnv([lambda: _E()]), mu, sd)
    obs = w.reset()
    assert np.allclose(obs[0], apply_scaler(raw, mu, sd))


def test_scaffold_rejects_gamma_not_one():
    """gamma != 1.0 отвергается на границе (тихий убийца кредита на холдах)."""
    from spotrl.algo.ppo_scaffold import build_scaffold, _dummy_vec_env
    env = _dummy_vec_env()
    with pytest.raises(ValueError):
        build_scaffold(env, _SEED, _MF, kl_coef=0.1, ent_coef=0.01, gamma=0.99)
