"""Этап 0.4: бенчмарк ML-стеков на РЕАЛЬНЫХ данных (окно из StateCache).

Меряем то, что определяет скорость обучения (П5):
  1) сырой step() среды (одиночный цикл python);
  2) векторизация: SB3 DummyVecEnv / SubprocVecEnv, pufferlib Serial/MP;
  3) обучение PPO: SB3 (torch) vs SBX (jax) — env-steps/сек на identичной
     мини-среде и identичных данных.

Запуск (env rlbench311, GPU не нужен):
  python strategy_rl/bench_stacks.py <manifest_key> [--steps 200000]
Результат: печать таблицы + strategy_rl/bench_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

CACHE_DIR = os.environ.get("RLBENCH_CACHE",
                           os.path.expanduser("~/Data/rlbinancetrader/state_cache"))
RESULTS = os.environ.get("RLBENCH_RESULTS",
                         os.path.join(os.path.dirname(__file__), "bench_results.json"))
N_ENVS = 8


def load_features(key: str) -> np.ndarray:
    import pandas as pd
    with open(os.path.join(CACHE_DIR, f"{key}.manifest.json")) as f:
        man = json.load(f)
    df = pd.read_parquet(man["parquet"])
    feat = df[[c for c in df.columns if not c.startswith("tb_")]] \
        .select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(feat)


def make_env_cls(data: np.ndarray):
    """Мини-среда «тайминг выхода» поверх предвычисленных массивов.

    Только для замера пропускной способности: логика позиций упрощена,
    но шаг делает ту же работу, что будет в бою: чтение строки массива,
    учёт позиции, награда по закрытию.
    """
    import gymnasium as gym

    close = data[:, 3].copy() + 1e-9  # четвёртая колонка исходного набора — close

    class MiniExitEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(data.shape[1] + 2,),
                dtype=np.float32)
            self.action_space = gym.spaces.Discrete(2)  # 0 hold / 1 exit
            self._extra = np.zeros(2, dtype=np.float32)

        def _obs(self):
            self._extra[0] = 1.0 if self.in_pos else 0.0
            self._extra[1] = (close[self.t] / self.entry - 1.0) if self.in_pos else 0.0
            return np.concatenate([data[self.t], self._extra])

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.t = int(self.np_random.integers(0, len(close) - 10_000))
            self.start = self.t
            self.in_pos = False
            self.entry = 1.0
            return self._obs(), {}

        def step(self, action):
            self.t += 1
            reward = 0.0
            if not self.in_pos and self.t % 240 == 0:      # «вход» каждые 4ч
                self.in_pos, self.entry = True, close[self.t]
            elif self.in_pos and (action == 1 or self.t % 240 == 239):
                reward = float(np.log(close[self.t] / self.entry))
                self.in_pos = False
            done = self.t - self.start >= 9_999
            return self._obs(), reward, done, False, {}

    return MiniExitEnv


def bench_raw(env_cls, n_steps: int) -> float:
    env = env_cls()
    env.reset(seed=0)
    a = 0
    t0 = time.perf_counter()
    for i in range(n_steps):
        _, _, done, _, _ = env.step(a)
        a = i & 1
        if done:
            env.reset()
    return n_steps / (time.perf_counter() - t0)


def bench_sb3_vec(env_cls, n_steps: int, backend: str) -> float:
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    vec_cls = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}[backend]
    vec = vec_cls([env_cls for _ in range(N_ENVS)])
    vec.reset()
    acts = np.zeros(N_ENVS, dtype=np.int64)
    iters = n_steps // N_ENVS
    t0 = time.perf_counter()
    for _ in range(iters):
        vec.step(acts)
    dt = time.perf_counter() - t0
    vec.close()
    return iters * N_ENVS / dt


def bench_puffer_vec(env_cls, n_steps: int, backend: str) -> float:
    import pufferlib
    import pufferlib.emulation
    import pufferlib.vector as pv

    def creator(buf=None):
        # pufferlib.vector передаёт shared-memory buf в env_creator
        return pufferlib.emulation.GymnasiumPufferEnv(env_creator=env_cls, buf=buf)

    vec_backend = {"serial": pv.Serial, "mp": pv.Multiprocessing}[backend]
    vec = pv.make(creator, num_envs=N_ENVS, backend=vec_backend)
    vec.reset()
    acts = np.zeros(N_ENVS, dtype=np.int64)
    iters = n_steps // N_ENVS
    t0 = time.perf_counter()
    for _ in range(iters):
        vec.step(acts)
    dt = time.perf_counter() - t0
    vec.close()
    return iters * N_ENVS / dt


def bench_ppo(env_cls, total: int, lib: str) -> float:
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec = DummyVecEnv([env_cls for _ in range(N_ENVS)])
    kw = dict(n_steps=256, batch_size=512, n_epochs=4, verbose=0)
    if lib == "sb3":
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", vec, device="cpu", **kw)
    elif lib == "sbx":
        from sbx import PPO
        model = PPO("MlpPolicy", vec, **kw)
    else:
        raise ValueError(lib)
    t0 = time.perf_counter()
    model.learn(total_timesteps=total, progress_bar=False)
    dt = time.perf_counter() - t0
    vec.close()
    return total / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--ppo-steps", type=int, default=100_000)
    ap.add_argument("--only", default="",
                    help="запятая-список тестов (напр. raw_single,ppo_sb3)")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))

    data = load_features(args.key)
    print(f"data: {data.shape} float32 ({data.nbytes / 1e6:.0f}MB)")
    env_cls = make_env_cls(data)

    res = {"key": args.key, "data_shape": list(data.shape), "n_envs": N_ENVS}
    tests = [
        ("raw_single", lambda: bench_raw(env_cls, args.steps)),
        ("sb3_dummy", lambda: bench_sb3_vec(env_cls, args.steps, "dummy")),
        ("sb3_subproc", lambda: bench_sb3_vec(env_cls, args.steps, "subproc")),
        ("puffer_serial", lambda: bench_puffer_vec(env_cls, args.steps, "serial")),
        ("puffer_mp", lambda: bench_puffer_vec(env_cls, args.steps, "mp")),
        ("ppo_sb3", lambda: bench_ppo(env_cls, args.ppo_steps, "sb3")),
        ("ppo_sbx", lambda: bench_ppo(env_cls, args.ppo_steps, "sbx")),
    ]
    for name, fn in tests:
        if only and name not in only:
            continue
        try:
            sps = fn()
            res[name] = round(sps)
            print(f"{name:>14}: {sps:>12,.0f} steps/s")
        except Exception as e:  # noqa: BLE001 — бенчмарк должен дойти до конца
            res[name] = f"FAIL: {type(e).__name__}: {e}"
            print(f"{name:>14}: FAIL — {e}")

    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=1)
    print("saved:", RESULTS)


if __name__ == "__main__":
    main()
