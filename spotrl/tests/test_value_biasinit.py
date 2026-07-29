"""Быстрые тесты bias-init критика (замена недостижимого EV-порога).

Тяжёлый прогон (mean(G) из кэша reward + сохранение reg_cd_vw2) — через CLI
`spotrl.bc.value_biasinit`, в быстрый набор НЕ входит.
"""
from __future__ import annotations

import numpy as np

from spotrl.bc.train_clone import build_policy, head0_logits
from spotrl.bc.value_biasinit import bias_init_critic
from spotrl.bc.value_warmup import explained_variance, _predict_values


def test_bias_init_constant_output_and_scale():
    """weight=0, bias=mean → выход критика константа = mean; EV_all ровно 0."""
    _, policy = build_policy(6, seed=0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(512, 6)).astype(np.float32)
    G = rng.normal(loc=0.0044, scale=0.03, size=512).astype(np.float32)
    g_mean = float(G.mean())

    bi = bias_init_critic(policy, g_mean)
    assert bi["weight_absmax_after"] == 0.0
    assert abs(bi["bias_after"] - g_mean) < 1e-7

    v = _predict_values(policy, X)
    # выход одинаков на всех барах и равен mean(G).
    assert float(np.abs(v - g_mean).max()) < 1e-6
    # EV константы-среднего = 1 - Var(G-mean)/Var(G) = 0 ровно.
    assert abs(explained_variance(G, v)) < 1e-5


def test_bias_init_freezes_policy_logits():
    """bias-init трогает только критика: логиты головы 0 бит-в-бит неизменны."""
    _, policy = build_policy(6, seed=0)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(256, 6)).astype(np.float32)
    d_before = head0_logits(policy, X).copy()
    bias_init_critic(policy, 0.0044)
    d_after = head0_logits(policy, X)
    assert np.array_equal(d_before, d_after)


def test_bias_init_weight_not_frozen_by_zero():
    """weight=0 НЕ замораживает критика: ненулевой градиент MSE на первом шаге."""
    import torch as th
    _, policy = build_policy(6, seed=0)
    bias_init_critic(policy, 0.0044)
    x = th.randn(32, 6)
    feats = policy.extract_features(x)
    f = feats[0] if isinstance(feats, tuple) else feats
    _, lat = policy.mlp_extractor(f)
    v = policy.value_net(lat).squeeze(-1)
    loss = ((v - th.full((32,), 0.05)) ** 2).mean()
    loss.backward()
    assert float(policy.value_net.weight.grad.abs().max()) > 0.0
