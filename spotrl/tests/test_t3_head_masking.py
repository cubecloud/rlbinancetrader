"""Т3 — нулевой градиент на замороженных головах.

Проверка на torch: логиты замороженной головы получают ровно нулевой
градиент, логиты рабочей головы — ненулевой. Это заменяет маски действий
(PLAN v0.5, 3.7).
"""
from __future__ import annotations

import numpy as np
import pytest

from spotrl.algo.head_masking import head_loss_mask
from spotrl.spec.actions import HEAD_POSITION, HEAD_SL, HEAD_TP

torch = pytest.importorskip("torch")


def test_head_loss_mask_shape_and_values():
    """Голова позиции активна всегда, SL/TP — только на барах входа."""
    entry = np.array([True, False, False, True])
    mask = head_loss_mask(entry, params_own=True)
    assert mask.shape == (4, 3)
    assert np.all(mask[:, HEAD_POSITION] == 1.0)
    assert list(mask[:, HEAD_SL]) == [1.0, 0.0, 0.0, 1.0]
    assert list(mask[:, HEAD_TP]) == [1.0, 0.0, 0.0, 1.0]


def test_head_loss_mask_params_locked():
    """При запертой свободе SL/TP головы не участвуют в лоссе никогда."""
    mask = head_loss_mask(np.array([True, True]), params_own=False)
    assert np.all(mask[:, HEAD_SL] == 0.0) and np.all(mask[:, HEAD_TP] == 0.0)


def test_frozen_head_gets_zero_gradient():
    """Градиент по логитам замороженной головы ровно нулевой."""
    batch = 8
    logits_pos = torch.zeros(batch, 2, requires_grad=True)
    logits_sl = torch.zeros(batch, 4, requires_grad=True)
    actions_pos = torch.zeros(batch, dtype=torch.long)
    actions_sl = torch.full((batch,), 2, dtype=torch.long)
    mask = torch.as_tensor(head_loss_mask(np.zeros(batch, dtype=bool),
                                          params_own=False))
    lp_pos = torch.distributions.Categorical(logits=logits_pos).log_prob(actions_pos)
    lp_sl = torch.distributions.Categorical(logits=logits_sl).log_prob(actions_sl)
    loss = -(lp_pos * mask[:, HEAD_POSITION] + lp_sl * mask[:, HEAD_SL]).mean()
    loss.backward()
    assert torch.count_nonzero(logits_sl.grad).item() == 0
    assert torch.count_nonzero(logits_pos.grad).item() > 0
