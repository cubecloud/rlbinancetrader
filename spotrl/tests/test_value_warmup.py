"""Быстрые тесты value-warmup: заморозка головы политики + метрика EV.

Тяжёлый прогон (сбор return-to-go средой + обучение критика на saved reg_cd) —
через CLI `spotrl.bc.value_warmup`, в быстрый набор НЕ входит.
"""
from __future__ import annotations

import numpy as np

from spotrl.bc.train_clone import build_policy, head0_logits
from spotrl.bc.value_warmup import (explained_variance, value_params,
                                    train_value, trade_close_return_to_go,
                                    split_train_val_trades, _trade_segments)


def test_explained_variance_bounds():
    """EV=1 при точном предсказании, =0 при константе-среднем, <0 при худшем."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert explained_variance(y, y) == 1.0
    assert abs(explained_variance(y, np.full(4, y.mean()))) < 1e-9
    assert explained_variance(y, y[::-1]) < 0.0


def test_value_params_exclude_policy_head():
    """value_params = только критик; ни action_net, ни pi-ветки в нём нет."""
    _, policy = build_policy(6, seed=0)
    vp = set(id(p) for p in value_params(policy))
    for p in policy.action_net.parameters():
        assert id(p) not in vp
    for p in policy.mlp_extractor.policy_net.parameters():
        assert id(p) not in vp
    # критик целиком присутствует.
    for p in list(policy.value_net.parameters()) + \
            list(policy.mlp_extractor.value_net.parameters()):
        assert id(p) in vp


def test_train_value_freezes_policy_logits():
    """Обучение критика НЕ меняет логиты головы 0 (бит-в-бит), критик — двигает."""
    _, policy = build_policy(6, seed=0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(256, 6)).astype(np.float32)
    G = rng.normal(size=256).astype(np.float32)
    d_before = head0_logits(policy, X).copy()
    v_before = [p.detach().clone().numpy()
                for p in policy.value_net.parameters()]
    train_value(policy, X, G, seed=0, n_epochs=3)
    d_after = head0_logits(policy, X)
    assert np.array_equal(d_before, d_after)           # голова заморожена
    v_after = [p.detach().numpy() for p in policy.value_net.parameters()]
    assert not np.array_equal(v_before[0], v_after[0])  # критик изменился


def test_trade_close_return_to_go_resets_per_segment():
    """Trade-close = обратный cumsum ВНУТРИ in-position сегмента; flat → 0."""
    r = np.array([1.0, 2.0, 3.0, 4.0, 5.0], np.float32)
    ip = np.array([1, 1, 0, 1, 1], np.float32)
    g = trade_close_return_to_go(r, ip)
    assert g.tolist() == [3.0, 2.0, 0.0, 9.0, 5.0]


def test_trade_segments_and_split_are_trade_level():
    """Сегменты нумеруют сделки; валидация держит ЦЕЛЫЕ сделки (не бары)."""
    ip = np.array([0, 1, 1, 0, 1, 0, 1, 1], np.float32)  # сделки 0,1,2
    seg = _trade_segments(ip)
    assert seg.tolist() == [-1, 0, 0, -1, 1, -1, 2, 2]
    tr, va = split_train_val_trades(ip, every=2)          # валид. = сделки 0,2
    assert set(va.tolist()) == {1, 2, 6, 7}
    assert set(va).isdisjoint(set(tr))
    # ни один val-бар не флэт; целые сделки в одну сторону.
    assert all(ip[i] > 0.5 for i in va)
