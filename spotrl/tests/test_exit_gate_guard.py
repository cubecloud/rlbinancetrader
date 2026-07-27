"""Быстрые тесты стража гейта выхода (ШАГ-1): чистота априорной метки M.

M = {exit_reason=='tp' И m_exit_flag==0} обязана быть ЧИСТОЙ функцией v7-меток,
без обращения к логитам/политике клона (джерримендеринг-ассерт). Тяжёлый прогон
(saved reg_cd + кросс-эпоховые клоны) — через CLI, в быстрый набор НЕ входит.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from spotrl.bc.exit_gate_guard import mechanical_tp_mask


def _mini() -> pd.DataFrame:
    """tp-механический, tp-сигнальный, legflip, b2b, не-выход."""
    n = 5
    df = pd.DataFrame({
        "exit_reason": ["tp", "tp", "legflip", "b2b", ""],
        "is_flip_exit": np.array([1, 1, 1, 1, 0], bool),
        "m_exit_flag": np.array([0, 1, 0, 0, 0], np.float32),
    })
    return df


def test_M_is_only_mechanical_tp():
    """M ловит РОВНО tp с m_exit_flag==0; сигнальный tp (mf==1) исключён."""
    m = mechanical_tp_mask(_mini())
    assert m.tolist() == [True, False, False, False, False]


def test_signal_tp_stays_out_of_M():
    """tp с m_exit_flag==1 (signal-confirmed) НЕ попадает в M → остаётся must-copy."""
    df = _mini()
    m = mechanical_tp_mask(df)
    sig_tp = (df["exit_reason"].to_numpy() == "tp") \
        & (df["m_exit_flag"].to_numpy() > 0.5) \
        & df["is_flip_exit"].to_numpy().astype(bool)
    assert (m & sig_tp).sum() == 0
    assert sig_tp.sum() == 1


def test_M_pure_function_no_clone_logits():
    """Джерримендеринг-ассерт: сигнатура — только df; в исходнике нет logit/policy."""
    sig = inspect.signature(mechanical_tp_mask)
    assert list(sig.parameters) == ["df"]
    src = inspect.getsource(mechanical_tp_mask)
    for forbidden in ("logit", "head0", "policy", "action_net", "mlp_extractor",
                      "score", "PPO", "d["):
        assert forbidden not in src, f"M обращается к логитам клона: {forbidden}"


def test_M_ignores_injected_score_column():
    """Маска идентична при любой подмешанной колонке логитов (она её не видит)."""
    df = _mini()
    m1 = mechanical_tp_mask(df)
    df2 = df.copy()
    df2["d_clone"] = np.random.default_rng(0).normal(size=len(df))
    m2 = mechanical_tp_mask(df2)
    assert np.array_equal(m1, m2)
