"""Гейт эквивалентности: побитовое сравнение прогона среды до и после правок.

Смысл: любая оптимизация горячего шага обязана оставить наблюдения, награды,
флаги и трейдбук ПОБИТОВО прежними. Порог — ноль различающихся элементов,
допусков нет.

Запуск::

    python -m spotrl.bench.equivalence record ref.npz   # до правок
    python -m spotrl.bench.equivalence check  ref.npz   # после правок

Компактный вид (хранится в репозитории, 3 КБ вместо 33 МБ)::

    python -m spotrl.bench.equivalence record-digest ref.json
    python -m spotrl.bench.equivalence check-digest  ref.json

Сценарии подобраны так, чтобы прогон задевал ВСЕ ветки шага: вход и выход по
решению агента, стоп-лосс, тейк-профит, закрытие на конце данных, усечение
эпизода. Гистограмма причин выхода печатается — если ветка не встретилась ни
разу, гейт для неё ничего не доказывает.
"""
from __future__ import annotations

import hashlib
import json
import sys
from typing import Dict, Iterable, Tuple

import numpy as np

from spotrl.config import EnvConfig, FreedomConfig, WorldConfig
from spotrl.data.synthetic import make_synthetic_state
from spotrl.envs.spot_env import SpotFlipEnv
from spotrl.spec.actions import ActionSpec

N_STEPS = 100_000
N_BARS = 20_000
DATA_SEED = 7
ACTION_SEED = 12_345
FIRST_EPISODE_SEED = 1_000


def scenarios() -> Dict[str, Tuple[EnvConfig, float]]:
    """Сценарии прогона: конфигурация среды и вероятность действия FLIP.

    Returns:
        Словарь «имя -> (конфигурация, p_flip)».
    """
    base = EnvConfig(freedom=FreedomConfig(exit_own=True, entry_own=True,
                                           params_own=False),
                     episode_len=5_000)
    full = EnvConfig(world=WorldConfig(stop_loss_frac=0.05),
                     freedom=FreedomConfig(exit_own=True, entry_own=True,
                                           params_own=True),
                     action_spec=ActionSpec(sl_buckets=(0.01, 0.02, 0.05),
                                            tp_buckets=(0.02, 0.05, 0.10),
                                            expert_sl_index=2, expert_tp_index=2),
                     episode_len=5_000)
    tail = EnvConfig(freedom=FreedomConfig(exit_own=True, entry_own=True),
                     episode_len=25_000)
    return {"base": (base, 0.05), "full": (full, 0.02), "tail": (tail, 0.01)}


def rollout(config: EnvConfig, p_flip: float, n_steps: int = N_STEPS) -> dict:
    """Прогон среды с жёстко заданными сидами эпизодов и потоком действий.

    Args:
        config: конфигурация среды.
        p_flip: вероятность действия FLIP на баре.
        n_steps: длина прогона в шагах.

    Returns:
        Словарь массивов: наблюдения, награды, флаги, причины выхода и
        трейдбук (числовая и текстовая части).
    """
    env = SpotFlipEnv(make_synthetic_state(n_bars=N_BARS, seed=DATA_SEED), config)
    rng = np.random.default_rng(ACTION_SEED)
    nvec = config.action_spec.nvec
    seeds = iter(range(FIRST_EPISODE_SEED, FIRST_EPISODE_SEED + n_steps))

    env.reset(seed=next(seeds))
    obs_log = np.empty((n_steps, config.obs_spec.size), dtype=np.float32)
    rewards = np.empty(n_steps, dtype=np.float64)
    terminated = np.empty(n_steps, dtype=bool)
    truncated = np.empty(n_steps, dtype=bool)
    reasons, trades = [], []
    action = np.zeros(3, dtype=np.int64)
    for i in range(n_steps):
        action[0] = 1 if rng.random() < p_flip else 0
        action[1] = rng.integers(nvec[1])
        action[2] = rng.integers(nvec[2])
        obs, reward, term, trunc, info = env.step(action)
        obs_log[i] = obs                      # копия: буфер среды переиспользуется
        rewards[i], terminated[i], truncated[i] = reward, term, trunc
        reasons.append(info["exit_reason"])
        if trunc:
            trades.extend(_trade_rows(env.book.closed))
            env.reset(seed=next(seeds))
    trades.extend(_trade_rows(env.book.closed))
    num, txt = _split_trades(trades)
    return {"obs": obs_log, "rew": rewards, "term": terminated, "trunc": truncated,
            "reasons": np.array(reasons), "trades_num": num, "trades_txt": txt}


def _trade_rows(closed: Iterable) -> list:
    """Разложить закрытые сделки в кортежи для сравнения."""
    return [(t.entry_bar, t.exit_bar, t.entry_price, t.exit_price,
             t.return_pct, t.exit_reason) for t in closed]


def _split_trades(trades: list) -> Tuple[np.ndarray, np.ndarray]:
    """Разделить трейдбук на числовой массив и массив причин выхода."""
    if not trades:
        return np.zeros((0, 5), dtype=np.float64), np.array([], dtype="<U8")
    num = np.array([row[:5] for row in trades], dtype=np.float64)
    return num, np.array([row[5] for row in trades], dtype="<U8")


def collect_runs(n_steps: int = N_STEPS, verbose: bool = True) -> Dict[str, np.ndarray]:
    """Прогнать все сценарии и собрать плоский словарь массивов."""
    out: Dict[str, np.ndarray] = {}
    for name, (config, p_flip) in scenarios().items():
        run = rollout(config, p_flip, n_steps=n_steps)
        for key, value in run.items():
            out[f"{name}_{key}"] = value
        if verbose:
            uniq, counts = np.unique(run["reasons"], return_counts=True)
            print(f"[{name}] сделок {len(run['trades_num'])}, "
                  f"усечений {int(run['trunc'].sum())}, "
                  f"причины выхода {dict(zip(uniq.tolist(), counts.tolist()))}")
    return out


def compare(reference: Dict[str, np.ndarray],
            current: Dict[str, np.ndarray], verbose: bool = True) -> int:
    """Сравнить два прогона побитово и вернуть число несовпавших массивов."""
    bad = 0
    for key in sorted(current):
        left, right = reference[key], current[key]
        same = left.shape == right.shape and np.array_equal(left, right)
        if not same:
            bad += 1
            diff = "форма" if left.shape != right.shape else int((left != right).sum())
            print(f"РАСХОЖДЕНИЕ {key}: {diff}")
        elif verbose:
            print(f"ok {key}: 0 различий из {left.size}")
    return bad


def digest(data: Dict[str, np.ndarray]) -> Dict[str, str]:
    """sha256 каждого массива прогона — компактный эталон вместо 33 МБ .npz.

    Побитовое равенство массивов эквивалентно равенству хэшей их байтов,
    поэтому для гейта достаточно хранить словарь хэшей.
    """
    out = {}
    for key in sorted(data):
        array = data[key]
        if array.dtype.kind in ("U", "S"):
            # Приводим строки к общей ширине: иначе хэш зависел бы от того,
            # какая длина строки встретилась в прогоне, а не от значений.
            array = array.astype("<U16")
        out[key] = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    return out


def compare_digest(reference: Dict[str, str], current: Dict[str, str],
                   verbose: bool = True) -> int:
    """Сравнить хэши двух прогонов и вернуть число несовпавших массивов."""
    bad = 0
    for key in sorted(current):
        if reference.get(key) != current[key]:
            bad += 1
            print(f"РАСХОЖДЕНИЕ {key}: sha256 не совпал")
        elif verbose:
            print(f"ok {key}: sha256 совпал")
    return bad


def main(argv: list) -> int:
    """Точка входа: `record|check <файл.npz>` или `record-digest|check-digest <файл.json>`."""
    modes = ("record", "check", "record-digest", "check-digest")
    if len(argv) != 3 or argv[1] not in modes:
        print(__doc__)
        return 2
    mode, path = argv[1], argv[2]
    data = collect_runs()
    if mode == "record-digest":
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(digest(data), handle, indent=1, sort_keys=True)
        print("эталонные хэши записаны:", path)
        return 0
    if mode == "check-digest":
        with open(path, encoding="utf-8") as handle:
            bad = compare_digest(json.load(handle), digest(data))
        print("ГЕЙТ ЭКВИВАЛЕНТНОСТИ:", "PASS" if bad == 0 else f"FAIL ({bad} массивов)")
        return 1 if bad else 0
    if mode == "record":
        np.savez(path, **data)
        print("эталон записан:", path)
        return 0
    with np.load(path) as ref:
        bad = compare({k: ref[k] for k in ref.files}, data)
    print("ГЕЙТ ЭКВИВАЛЕНТНОСТИ:", "PASS" if bad == 0 else f"FAIL ({bad} массивов)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
