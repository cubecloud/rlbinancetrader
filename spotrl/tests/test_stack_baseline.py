"""Тест-инвариант доказанного стека (rl_stack_baseline, Этап 0 аудита).

Проверяет, что ДЕФОЛТНАЯ сборка RL-раннера = доказанный стек (без ручной
сборки из пяти amend), и что джиттер стартов эпизода работает: сид-зависимость
(разные сиды → разные траектории), воспроизводимость (тот же сид → та же),
паритет (полный проход детерминирован со старта 0).
"""
import importlib
import os
from pathlib import Path

import numpy as np
import pytest

DATA = "/home/cubecloud/Data/rlbinancetrader"


def _fresh_k4_run(monkeypatch, **env):
    """Импортировать k4_run с чистым окружением (дефолты, не наследие env)."""
    for k in ("FF_SEED_MODEL", "FF_TERMINAL", "FF_PSL", "FF_ALPHA", "FF_LR",
              "FF_WD", "FF_EPLEN", "FF_MINHOLD", "FF_BDIP", "FF_ENT"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import spotrl.algo.k4_run as k4
    return importlib.reload(k4)


def test_default_build_is_proven_stack(monkeypatch):
    """Дефолты раннера = доказанный стек (baseline-документ)."""
    k4 = _fresh_k4_run(monkeypatch)
    assert k4.SEED_MODEL.endswith("bc_clone_v7_policy_reg_cd_vwk4cls")
    assert k4.TERMINAL is True                  # терминальная награда
    assert k4.PSL is True                       # p_SL-приор включён
    assert k4.PSL_ALPHA == 0.5                  # alpha=1.0 давал NaN
    assert k4.LR == pytest.approx(1e-5)         # низкий LR
    assert k4.WEIGHT_DECAY == pytest.approx(1e-4)
    assert k4.BETA == pytest.approx(0.369)      # раздельный якорь
    assert k4.BDIP == pytest.approx(0.10)
    assert k4.MIN_HOLD_DAYS == pytest.approx(0.0245)   # ~60 баров
    assert k4.EPISODE_LEN > 0                   # джиттер стартов для RL включён
    # артефакты доказанного стека существуют
    assert Path(k4.K3_PATH).exists(), "K3-классификатор p_SL не найден"
    assert Path(k4.SEED_MODEL + ".zip").exists() or \
        Path(k4.SEED_MODEL).exists() or \
        Path(k4.SEED_MODEL + ".manifest.json").exists()
    # прибор (гейт с калибровкой) доступен
    from spotrl.analysis.eval_gate_v2 import new_gate, oracle_deltas, random_deltas
    g = new_gate(np.zeros(10), "smoke")
    assert g["EFFECT_PRESENT"] is False


def _tiny_env(episode_len=None, allowed=None, n=5000):
    """Крошечная синтетическая среда SpotFlipEnv для тестов джиттера."""
    from spotrl.data.synthetic import make_synthetic_state
    from spotrl.envs.spot_env import SpotFlipEnv
    from spotrl.config import EnvConfig, WorldConfig, FreedomConfig, ActionSpec
    state = make_synthetic_state(n_bars=n, seed=0)
    cfg = EnvConfig(world=WorldConfig(), freedom=FreedomConfig(),
                    action_spec=ActionSpec(),
                    episode_len=(episode_len or n + 10))
    env = SpotFlipEnv(state, cfg)
    if allowed is not None:
        env.allowed_starts = allowed
    return env


def test_jitter_seed_dependence_and_reproducibility():
    """(б) разные сиды → разные старты; (в) тот же сид → тот же старт."""
    allowed = np.arange(0, 4000, 7)
    starts = {}
    for seed in (0, 1):
        env = _tiny_env(episode_len=256, allowed=allowed)
        _obs, info = env.reset(seed=seed)
        starts[seed] = info["start_bar"]
    assert starts[0] != starts[1], "сиды 0 и 1 дали одинаковый старт"
    env = _tiny_env(episode_len=256, allowed=allowed)
    _obs, info = env.reset(seed=0)
    assert info["start_bar"] == starts[0], "тот же сид дал другой старт"
    # старт принадлежит разрешённому множеству (flat-бары)
    assert starts[0] in set(allowed.tolist())


def test_full_pass_start_is_deterministic_zero():
    """(а) режим полного прохода (episode_len>=n) детерминирован: старт 0.

    Это режим копирования/гейтов — паритет 272/272 держится на нём (замер
    паритета в handoff; здесь — инвариант механизма старта)."""
    for seed in (0, 1, 7):
        env = _tiny_env(episode_len=None)          # n+10 → полный проход
        _obs, info = env.reset(seed=seed)
        assert info["start_bar"] == 0
