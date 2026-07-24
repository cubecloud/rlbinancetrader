"""Т6 — воспроизводимость по сиду: одинаковый сид, одинаковая книга сделок."""
from __future__ import annotations

import numpy as np

from spotrl.data.synthetic import make_synthetic_state
from spotrl.envs.spot_env import SpotFlipEnv


def _rollout(env, seed: int, n_steps: int = 300):
    """Прогон случайной политики с фиксированным сидом; книга + награды."""
    env.reset(seed=seed)
    env.action_space.seed(seed)
    rewards = []
    for _ in range(n_steps):
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        rewards.append(reward)
        if terminated or truncated:
            break
    return list(env.book.closed), np.asarray(rewards)


def test_same_seed_same_trades(env_config):
    """Два прогона одним сидом дают побитово одинаковые сделки и награды."""
    state = make_synthetic_state(n_bars=5_000, seed=0)
    trades_a, rewards_a = _rollout(SpotFlipEnv(state, env_config), seed=42)
    trades_b, rewards_b = _rollout(SpotFlipEnv(state, env_config), seed=42)
    assert trades_a == trades_b
    assert np.array_equal(rewards_a, rewards_b)


def test_different_seed_changes_rollout(env_config):
    """Разные сиды дают разные стартовые бары — тест на «живость» сида."""
    state = make_synthetic_state(n_bars=5_000, seed=0)
    env_a, env_b = SpotFlipEnv(state, env_config), SpotFlipEnv(state, env_config)
    _, info_a = env_a.reset(seed=1)
    _, info_b = env_b.reset(seed=2)
    assert info_a["start_bar"] != info_b["start_bar"]
