"""Парная per-trade оценка раннего выхода на dip: RL vs v7 и RL vs «случайный выход».

Гейт 4.2 (amend1 §4): парная per-trade dip-дельта RL−v7 (Wilcoxon p<0.01 И нижняя
граница block-bootstrap CI медианы > 0). Контроль: RL обязан бить «случайный выход»
(тот же счётчик выходов, случайные моменты внутри dip-холда). БЕЗ holdout.

Метод. Строим v7-базлайн (копи-режим драйвера) и RL-политику (argmax головы 0,
honor_dip+enforce_v7_exit) на одном разрезе. Пара по бару входа (одинаковый вход
v7). Для dip-сделок: чистая доходность сделки = exit/entry·(1−fee)²−1; fill по
open(bar_решения+1). Дельта в п.п. (×100).
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
import torch as th
from scipy.stats import wilcoxon

from spotrl.bc.build_clone_dataset import _make_env
from spotrl.bc.train_clone import apply_scaler
from spotrl.envs.first_freedom_driver import FirstFreedomDriver, load_expert_arrays
from spotrl.algo.ppo_scaffold import build_scaffold
from spotrl.spec.actions import HEAD_POSITION

FEE = 0.001
BASE = "/home/cubecloud/Data"
DATA = f"{BASE}/rlbinancetrader"
SEED_MODEL = f"{DATA}/bc_clone_v7_policy_reg_cd_vw2"
MANIFEST = SEED_MODEL + ".manifest.json"


def _prices(env):
    """Массивы open/close среды (для fill и контрфактов)."""
    return np.asarray(env._open, np.float64), np.asarray(env._close, np.float64)


def _roll_v7(state, sig, df_ds):
    """Копи-режим: v7-сделки. Возврат: dict entry_bar→(entry_price, exit_bar,
    exit_price, pos_tag) и (open, n)."""
    env = _make_env(state, sig)
    expert, is_sig = load_expert_arrays(df_ds, env._n_bars)
    drv = FirstFreedomDriver(env, expert, is_sig, honor_dip=False)
    drv.reset(seed=0)
    while True:
        _o, _r, _t, trunc, _i = drv.step(np.array([0, 0, 0]))
        if trunc:
            break
    op, _cl = _prices(env)
    trades = {c.entry_bar: (c.entry_price, c.exit_bar, c.exit_price)
              for c in env.book.closed}
    return trades, op


def _roll_rl(state, sig, df_ds, policy, mu, sd, cd_idx):
    """RL-политика (argmax головы 0) через драйвер honor_dip+enforce_v7_exit.
    Возврат: dict entry_bar→(exit_bar, exit_price, pos_tag)."""
    env = _make_env(state, sig)
    expert, is_sig = load_expert_arrays(df_ds, env._n_bars)
    drv = FirstFreedomDriver(env, expert, is_sig, honor_dip=True,
                             enforce_v7_exit=True)
    drv.reset(seed=0)
    while True:
        o = env.observe().astype(np.float32)
        cd = np.float32(o[cd_idx] > 0.0)
        x = apply_scaler(np.concatenate([o, [cd]])[None, :], mu, sd)
        with th.no_grad():
            d = policy.get_distribution(th.as_tensor(x)).distribution
            a = int(d[HEAD_POSITION].logits.argmax(-1).item())  # argmax head0
        _o, _r, _t, trunc, _i = drv.step(np.array([a, 0, 0]))
        if trunc:
            break
    return {c.entry_bar: (c.exit_bar, c.exit_price)
            for c in env.book.closed}, drv.early_exit_count


def _ret(exit_price, entry_price):
    """Чистая доходность сделки (двусторонняя комиссия)."""
    return exit_price / entry_price * (1.0 - FEE) ** 2 - 1.0


def evaluate(ckpt, state, sig, ds_path, label, n_rand=32, seed=0):
    """Парные дельты RL−v7 и RL−random по dip-сделкам одного разреза."""
    mf = json.loads(open(MANIFEST).read())
    mu = np.array(mf["scaler_mu"], np.float32); sd = np.array(mf["scaler_sd"], np.float32)
    cd_idx = mf["obs_cols"].index("a_cooldown_remain")
    df_ds = pd.read_parquet(ds_path)

    from stable_baselines3.common.vec_env import DummyVecEnv
    from spotrl.algo.ppo_scaffold import _dummy_vec_env
    model, _clone = build_scaffold(_dummy_vec_env(39), SEED_MODEL, MANIFEST,
                                   kl_coef=0.0, ent_coef=0.0, gamma=1.0,
                                   n_steps=64, batch_size=64, n_epochs=1)
    model.policy.load_state_dict(th.load(ckpt, map_location="cpu"))
    model.policy.set_training_mode(False)

    v7, op = _roll_v7(state, sig, df_ds)
    rl, early = _roll_rl(state, sig, df_ds, model.policy, mu, sd, cd_idx)

    # pos_tag по бару входа: book entry_bar = decision_bar + 1; датасет хранит
    # pos_tag на баре решения (is_flip_entry).
    ent = df_ds[df_ds["is_flip_entry"].astype(bool)]
    postag = {int(b) + 1: str(t) for b, t in zip(ent["bar"], ent["pos_tag"])}

    rng = np.random.default_rng(seed)
    d_v7, d_rand, n_early = [], [], 0
    for eb, (ep, xb_v7, xp_v7) in v7.items():
        if postag.get(eb) != "dip":
            continue
        xb_rl, xp_rl = rl.get(eb, (xb_v7, xp_v7))
        r_v7 = _ret(xp_v7, ep); r_rl = _ret(xp_rl, ep)
        d_v7.append((r_rl - r_v7) * 100.0)
        if xb_rl < xb_v7:                       # RL вышел РАНЬШЕ
            n_early += 1
            lo, hi = eb + 1, xb_v7               # случайный выход в окне холда
            rr = []
            for _ in range(n_rand):
                rb = int(rng.integers(lo, hi + 1)) if hi > lo else lo
                px = op[min(rb + 1, len(op) - 1)]
                rr.append(_ret(px, ep))
            d_rand.append((r_rl - float(np.mean(rr))) * 100.0)
    return _stats(label, d_v7, d_rand, early, n_early, len(d_v7))


def _boot_ci(x, n=5000, seed=0):
    """95% bootstrap CI медианы (блок по сделке = сам вектор дельт)."""
    rng = np.random.default_rng(seed); x = np.asarray(x)
    if len(x) < 3:
        return (float("nan"), float("nan"))
    meds = [np.median(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def _stats(label, d_v7, d_rand, early_bars, n_early, n_dip):
    """Wilcoxon + bootstrap CI для RL−v7 и RL−random."""
    def blk(name, d):
        """Wilcoxon (one-sided greater) + bootstrap CI медианы для вектора дельт."""
        d = np.asarray([x for x in d])
        if len(d) < 3 or np.allclose(d, 0):
            return {"n": int(len(d)), "median_pp": float(np.median(d)) if len(d) else float("nan"),
                    "wilcoxon_p": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
        nz = d[d != 0]
        try:
            p = float(wilcoxon(nz, alternative="greater").pvalue) if len(nz) else float("nan")
        except Exception:
            p = float("nan")
        lo, hi = _boot_ci(d)
        return {"n": int(len(d)), "n_nonzero": int(len(nz)),
                "median_pp": float(np.median(d)), "mean_pp": float(np.mean(d)),
                "wilcoxon_p_greater": p, "ci_lo_pp": lo, "ci_hi_pp": hi}
    r = {"label": label, "n_dip_trades": n_dip, "n_early_exit_trades": n_early,
         "early_exit_bars_driver": early_bars,
         "RL_vs_v7": blk("v7", d_v7), "RL_vs_random": blk("rand", d_rand)}
    return r


def main():
    """CLI: парная per-trade оценка чекпойнта (RL vs v7 и vs random) по разрезам."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cuts = [("train_2021",
             f"{BASE}/sunday_tests/state_v0/state_v3_causal_2021-01_2024-03.parquet",
             f"{DATA}/v7signals_2021.parquet", f"{DATA}/bc_clone_v7_2021.parquet")]
    res = {"ckpt": args.ckpt, "cuts": []}
    for label, state, sig, ds in cuts:
        r = evaluate(args.ckpt, state, sig, ds, label)
        res["cuts"].append(r)
        print(f"\n=== {label} ===")
        print(f"dip-сделок={r['n_dip_trades']} ранних выходов(сделок)={r['n_early_exit_trades']}")
        for k in ("RL_vs_v7", "RL_vs_random"):
            b = r[k]
            print(f"{k}: median={b['median_pp']:.4f}пп n={b['n']} "
                  f"nonzero={b.get('n_nonzero','-')} Wilcoxon_p={b.get('wilcoxon_p_greater')} "
                  f"CI=[{b.get('ci_lo_pp')}, {b.get('ci_hi_pp')}]")
    if args.out:
        open(args.out, "w").write(json.dumps(res, ensure_ascii=False, indent=2))
        print("saved", args.out)


if __name__ == "__main__":
    main()
