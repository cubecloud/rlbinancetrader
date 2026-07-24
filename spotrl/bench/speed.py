"""Замер скорости среды: сырой шаг, векторизаторы, профиль горячего пути.

Запуск::

    python -m spotrl.bench.speed                 # сырая среда
    python -m spotrl.bench.speed --vec           # + DummyVecEnv / SubprocVecEnv
    python -m spotrl.bench.speed --profile       # + cProfile горячего цикла

Числа зависят от машины, поэтому в отчёт всегда идёт связка «наша среда /
эталонная мини-среда 0.4 на ТОМ ЖЕ железе», а не абсолютная цифра.
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import time
import numpy as np

from spotrl.config import EnvConfig, FreedomConfig
from spotrl.data.synthetic import make_synthetic_state
from spotrl.envs.spot_env import SpotFlipEnv

DEFAULT_STEPS = 200_000
DEFAULT_BARS = 100_000


def make_env(n_bars: int = DEFAULT_BARS, episode_len: int = 20_000) -> SpotFlipEnv:
    """Среда с включёнными свободами входа и выхода на синтетическом ряде."""
    config = EnvConfig(freedom=FreedomConfig(exit_own=True, entry_own=True),
                       episode_len=episode_len)
    return SpotFlipEnv(make_synthetic_state(n_bars=n_bars, seed=7), config)


def make_actions(n_steps: int, p_flip: float = 0.02, seed: int = 3) -> np.ndarray:
    """Поток действий, сгенерированный заранее — вне замеряемого цикла."""
    rng = np.random.default_rng(seed)
    actions = np.zeros((n_steps, 3), dtype=np.int64)
    actions[:, 0] = (rng.random(n_steps) < p_flip).astype(np.int64)
    return actions


def _loop(env: SpotFlipEnv, actions: np.ndarray) -> float:
    """Прогон и время в секундах (сбросы эпизода входят в замер)."""
    env.reset(seed=1)
    started = time.perf_counter()
    for i in range(len(actions)):
        _, _, _, truncated, _ = env.step(actions[i])
        if truncated:
            env.reset(seed=i)
    return time.perf_counter() - started


def measure_raw_env(n_steps: int = DEFAULT_STEPS, repeats: int = 3) -> float:
    """Скорость сырой среды в шагах в секунду (лучший из повторов).

    Прогрев обязателен: первый проход платит за прогрев кэшей и ленивые
    импорты numpy, и без него замер занижен примерно на треть.
    """
    env, actions = make_env(), make_actions(n_steps)
    _loop(env, actions[:min(20_000, n_steps)])
    return n_steps / min(_loop(env, actions) for _ in range(repeats))


def measure_vec_env(backend: str = "dummy", n_envs: int = 8,
                    n_steps: int = 100_000, n_bars: int = 50_000) -> float:
    """Скорость среды в векторизаторе SB3, шагов в секунду суммарно.

    Args:
        backend: 'dummy' или 'subproc'.
        n_envs: число сред.
        n_steps: суммарное число шагов по всем средам.
        n_bars: длина ряда на одну среду.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    vec_cls = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}[backend]

    def factory() -> SpotFlipEnv:
        """Фабрика одной среды для векторизатора."""
        return make_env(n_bars=n_bars)

    vec = vec_cls([factory for _ in range(n_envs)])
    try:
        vec.reset()
        actions = np.zeros((n_envs, 3), dtype=np.int64)
        iters = n_steps // n_envs
        for _ in range(min(200, iters)):
            vec.step(actions)
        started = time.perf_counter()
        for _ in range(iters):
            vec.step(actions)
        elapsed = time.perf_counter() - started
    finally:
        vec.close()
    return iters * n_envs / elapsed


def profile_step(n_steps: int = DEFAULT_STEPS, top: int = 15) -> None:
    """Напечатать cProfile горячего цикла (числа завышены накладными профайлера)."""
    env, actions = make_env(), make_actions(n_steps)
    _loop(env, actions[:20_000])
    profiler = cProfile.Profile()
    profiler.enable()
    _loop(env, actions)
    profiler.disable()
    pstats.Stats(profiler).sort_stats("tottime").print_stats(top)


def main() -> None:
    """Разобрать аргументы и напечатать замеры."""
    parser = argparse.ArgumentParser(description="скорость шага среды spotrl")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--vec", action="store_true", help="замерить векторизаторы SB3")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    raw = measure_raw_env(args.steps)
    print(f"сырая среда:        {raw:>12,.0f} шаг/с  ({1e6 / raw:.2f} мкс/шаг)")
    if args.vec:
        for backend in ("dummy", "subproc"):
            speed = measure_vec_env(backend, n_envs=args.n_envs)
            print(f"{backend:>8} x{args.n_envs}: {speed:>12,.0f} шаг/с")
    if args.profile:
        profile_step(args.steps)


if __name__ == "__main__":
    main()
