"""Численная часть loss-masking: свёртка log-prob и энтропии по головам."""
from __future__ import annotations

import numpy as np
import pytest

from spotrl.algo.head_masking import head_loss_mask, masked_entropy, masked_log_prob


def test_masked_log_prob_drops_frozen_heads():
    """Замаскированная голова вносит в сумму ровно ноль."""
    per_head = np.array([[-0.5, -1.0, -2.0], [-0.1, -0.2, -0.3]])
    mask = head_loss_mask(np.array([True, False]), params_own=True)
    assert np.allclose(masked_log_prob(per_head, mask), [-3.5, -0.1])


def test_masked_entropy_uses_same_reduction():
    """Энтропия сворачивается той же маской, что и log-prob."""
    per_head = np.ones((3, 3))
    mask = head_loss_mask(np.zeros(3, dtype=bool), params_own=True)
    assert np.allclose(masked_entropy(per_head, mask), [1.0, 1.0, 1.0])


def test_shape_mismatch_is_an_error():
    """Несовпадение форм — ошибка, а не молчаливый broadcast."""
    with pytest.raises(ValueError):
        masked_log_prob(np.ones((2, 3)), np.ones((3, 3)))
