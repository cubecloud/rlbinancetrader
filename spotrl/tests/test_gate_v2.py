"""Тесты гейта 4.2-v2 (eval_gate_v2.new_gate): прибор различает сигнал и шум."""
import numpy as np

from spotrl.analysis.eval_gate_v2 import new_gate


def test_sparse_positive_effect_passes():
    """Разреженный, но систематически положительный эффект (как oracle) проходит,
    хотя >90% дельт — нули (старый медианный гейт бы провалил)."""
    d = np.zeros(245)
    d[:20] = np.linspace(1.0, 8.0, 20)      # 20 сильно положительных
    g = new_gate(d, "sparse_pos")
    assert g["A_portfolio"]["pass"], g["A_portfolio"]
    assert g["B_acted_quality"]["pass"], g["B_acted_quality"]
    assert g["EFFECT_PRESENT"]
    assert g["acted_share"] < 0.1            # именно разреженный


def test_all_zero_fails():
    """Политика == v7 (все нули) → эффекта нет."""
    g = new_gate(np.zeros(100), "zeros")
    assert not g["A_portfolio"]["pass"]
    assert not g["EFFECT_PRESENT"]


def test_random_noise_fails():
    """Симметричный шум вокруг нуля (как random) не проходит ни A, ни B."""
    rng = np.random.default_rng(0)
    d = np.zeros(245)
    d[:200] = rng.normal(0.0, 2.0, 200)      # действует часто, около нуля
    g = new_gate(d, "noise")
    assert not g["EFFECT_PRESENT"]


def test_sparse_negative_fails():
    """Разреженные отрицательные отклонения (как r20) → провал по A и B."""
    d = np.zeros(245)
    d[:22] = -np.linspace(0.5, 3.0, 22)
    g = new_gate(d, "sparse_neg")
    assert not g["A_portfolio"]["pass"]
    assert not g["B_acted_quality"]["pass"]
    assert not g["EFFECT_PRESENT"]
