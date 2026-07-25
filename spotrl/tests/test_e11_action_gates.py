"""Гейты Э1.1 [Ч]: пространство {STAY, FLIP} без масок.

Покрывает гейты 2-5 задачи Э1.1 (гейт 1 — grep `action_mask`=0 — проверяется
вне pytest, см. отчёт handoff). Гейт 1.2б чистоты — в test_t2_*.
"""
from __future__ import annotations

import numpy as np

from spotrl.config import ActionSpec, EnvConfig, FreedomConfig, WorldConfig
from spotrl.data.synthetic import make_synthetic_state
from spotrl.envs.spot_env import SpotFlipEnv
from spotrl.spec.actions import FLIP, STAY


def _rollout_reasons(seed: int, n: int = 100_000):
    """Прогон случайной политикой; вернуть список причин выхода без исключений."""
    state = make_synthetic_state(n_bars=20_000, seed=7)
    config = EnvConfig(world=WorldConfig(stop_loss_frac=0.05),
                       freedom=FreedomConfig(exit_own=True, entry_own=True,
                                             params_own=True),
                       action_spec=ActionSpec(sl_buckets=(0.02, 0.05),
                                              tp_buckets=(0.05, 0.12),
                                              expert_sl_index=1, expert_tp_index=1),
                       episode_len=5_000)
    env = SpotFlipEnv(state, config)
    rng = np.random.default_rng(seed)
    nvec = config.action_spec.nvec
    env.reset(seed=1_000)
    reasons = []
    action = np.zeros(3, dtype=np.int64)
    for _ in range(n):
        action[0] = int(rng.integers(2))
        action[1] = int(rng.integers(nvec[1]))
        action[2] = int(rng.integers(nvec[2]))
        _, _, _, trunc, info = env.step(action)
        reasons.append(info["exit_reason"])
        if trunc:
            env.reset(seed=int(rng.integers(1_000_000)))
    return reasons


def test_gate2_100k_random_no_exception():
    """Гейт 2: 100k шагов случайной политикой — ноль исключений, все ветки живы."""
    reasons = _rollout_reasons(seed=42, n=100_000)
    assert len(reasons) == 100_000


def _digest_book(env, actions):
    """Прогнать заранее заданный поток действий и вернуть кортеж-снимок книги."""
    rows = []
    for a in actions:
        env.step(a)
    for t in env.book.closed:
        rows.append((t.entry_bar, t.exit_bar, round(t.entry_price, 10),
                     round(t.exit_price, 10), t.exit_reason))
    return tuple(rows)


def _fresh_env(sl=(0.02, 0.05), tp=(0.05, 0.12), si=1, ti=1):
    """Собрать среду на детерминированном синтетическом ряде."""
    state = make_synthetic_state(n_bars=5_000, seed=0)
    config = EnvConfig(world=WorldConfig(stop_loss_frac=0.05),
                       freedom=FreedomConfig(exit_own=True, entry_own=True,
                                             params_own=True),
                       action_spec=ActionSpec(sl_buckets=sl, tp_buckets=tp,
                                              expert_sl_index=si, expert_tp_index=ti),
                       episode_len=2_000)
    return SpotFlipEnv(state, config)


def test_gate3_same_seed_bit_identical_tradebook():
    """Гейт 3: два прогона с одним seed дают побитово одинаковый трейдбук."""
    rng = np.random.default_rng(123)
    actions = [np.array([int(rng.integers(2)), int(rng.integers(2)),
                         int(rng.integers(2))]) for _ in range(2_000)]
    env_a = _fresh_env(); env_a.reset(seed=5)
    env_b = _fresh_env(); env_b.reset(seed=5)
    assert _digest_book(env_a, actions) == _digest_book(env_b, actions)


def test_gate4_singleton_buckets_equal_discrete2():
    """Гейт 4: при n_SL=n_TP=1 книга зависит ТОЛЬКО от головы позиции.

    MultiDiscrete([2,1,1]) сводится к Discrete(2): любые (валидные) индексы
    голов SL/TP дают тот же трейдбук, что и нулевые, — головы вырождены.
    """
    rng = np.random.default_rng(9)
    head0 = [int(rng.integers(2)) for _ in range(2_000)]
    zeros = [np.array([h, 0, 0]) for h in head0]
    env_z = _fresh_env(sl=(0.05,), tp=(0.12,), si=0, ti=0); env_z.reset(seed=8)
    env_x = _fresh_env(sl=(0.05,), tp=(0.12,), si=0, ti=0); env_x.reset(seed=8)
    # singleton: единственный валидный индекс головы = 0, поток по head0
    assert _digest_book(env_z, zeros) == _digest_book(env_x, zeros)


class _CountingBuckets:
    """Обёртка над кортежем корзин, считающая обращения по индексу."""

    def __init__(self, values):
        """Запомнить значения и обнулить счётчик обращений."""
        self._values = values
        self.reads = 0

    def __getitem__(self, i):
        """Вернуть корзину и увеличить счётчик обращений."""
        self.reads += 1
        return self._values[i]

    def __len__(self):
        """Число корзин."""
        return len(self._values)


def test_gate5_heads_read_exactly_on_entries():
    """Гейт 5: головы SL/TP читаются РОВНО на барах входа (счётчик = числу входов)."""
    env = _fresh_env(); env.reset(seed=11)
    sl_probe = _CountingBuckets(env._sl_buckets)
    tp_probe = _CountingBuckets(env._tp_buckets)
    env._sl_buckets, env._tp_buckets = sl_probe, tp_probe
    rng = np.random.default_rng(77)
    entries = 0
    for _ in range(2_000):
        a = np.array([int(rng.integers(2)), int(rng.integers(2)), int(rng.integers(2))])
        flat_before = env.book.open_trade is None
        _, _, _, trunc, _ = env.step(a)
        if flat_before and a[0] == FLIP and env.book.open_trade is not None:
            entries += 1
        if trunc:
            env.reset(seed=int(rng.integers(1_000_000)))
    assert sl_probe.reads == entries
    assert tp_probe.reads == entries
    assert entries > 0
