"""K4 PoC оценка чекпойнта: гейт на train + tune-OOS (2024-03..2026-01), K2,
момент выхода, контроли oracle/random. Holdout 2026-01..07 НЕ трогается.

Роллаут RL идёт через тот же драйвер, что в обучении (honor_dip, enforce_v7_exit,
мин-холд) — train/serve-паритет. Tune-фолд ОБРЕЗАЕТСЯ на баре CUT_2024 (exit_bar
всех оцениваемых dip-сделок < 2026-01), роллаут прерывается на этом баре, чтобы
не касаться holdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th

from spotrl.bc.build_clone_dataset import _make_env
from spotrl.bc.train_clone import apply_scaler
from spotrl.envs.first_freedom_driver import FirstFreedomDriver, load_expert_arrays
from spotrl.algo.ppo_scaffold import build_scaffold, _dummy_vec_env
from spotrl.spec.actions import HEAD_POSITION
from spotrl.analysis.eval_gate_v2 import (
    new_gate, oracle_deltas, random_deltas, random_sum_distribution,
    rl_deltas, exit_reason_map, k2_tagged, _dip_trades,
)
from spotrl.analysis.eval_paired_dip import _ret, _prices, SEED_MODEL, MANIFEST

_STRATEGY_RL = Path(__file__).resolve().parents[2] / "strategy_rl"
if str(_STRATEGY_RL) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_RL))

BASE = "/home/cubecloud/Data"
DATA = f"{BASE}/rlbinancetrader"
MIN_HOLD_DAYS = 0.0245
CUT_2024 = 949159      # exit_bar < этого = до 2026-01 (holdout не трогаем)

CUTS = {
    "train_2021": {
        "state": f"{BASE}/sunday_tests/state_v0/state_v3_causal_2021-01_2024-03.parquet",
        "sig": f"{DATA}/v7signals_2021.parquet",
        "ds": f"{DATA}/bc_clone_v7_2021.parquet", "max_bar": None},
    "tune_2024_oos": {
        "state": f"{BASE}/sunday_tests/state_v0/state_v3_causal_2024-03_2026-07.parquet",
        "sig": f"{DATA}/v7signals_2024.parquet",
        "ds": f"{DATA}/bc_clone_v7_2024.parquet", "max_bar": CUT_2024},
}


def _roll_v7(state, sig, df_ds, max_bar):
    """v7 copy-режим, обрыв на max_bar. Возврат: {entry_bar:(ep,xb,xp)}, op."""
    env = _make_env(state, sig)
    expert, is_sig = load_expert_arrays(df_ds, env._n_bars)
    drv = FirstFreedomDriver(env, expert, is_sig, honor_dip=False)
    drv.reset(seed=0)
    while True:
        _o, _r, _t, trunc, _i = drv.step(np.array([0, 0, 0]))
        if trunc or (max_bar is not None and env._t >= max_bar):
            break
    op, _cl = _prices(env)
    return {c.entry_bar: (c.entry_price, c.exit_bar, c.exit_price)
            for c in env.book.closed}, op


def _roll_rl(state, sig, df_ds, policy, mu, sd, cd_idx, max_bar):
    """RL argmax через драйвер (honor_dip, enforce_v7, мин-холд), обрыв на max_bar."""
    env = _make_env(state, sig)
    expert, is_sig = load_expert_arrays(df_ds, env._n_bars)
    drv = FirstFreedomDriver(env, expert, is_sig, honor_dip=True,
                             enforce_v7_exit=True, min_hold_days=MIN_HOLD_DAYS)
    drv.reset(seed=0)
    while True:
        o = env.observe().astype(np.float32)
        cd = np.float32(o[cd_idx] > 0.0)
        x = apply_scaler(np.concatenate([o, [cd]])[None, :], mu, sd)
        with th.no_grad():
            d = policy.get_distribution(th.as_tensor(x)).distribution
            a = int(d[HEAD_POSITION].logits.argmax(-1).item())
        _o, _r, _t, trunc, _i = drv.step(np.array([a, 0, 0]))
        if trunc or (max_bar is not None and env._t >= max_bar):
            break
    return {c.entry_bar: (c.exit_bar, c.exit_price) for c in env.book.closed}, \
        drv.early_exit_count


def _exit_age_stats(v7, rl, postag):
    """Момент раннего выхода внутри удержания: возраст (xb_rl−eb) и дельта по
    ранней/поздней половине холда."""
    ages, early_half, late_half = [], [], []
    for eb, (ep, xb_v7, xp_v7) in v7.items():
        if postag.get(eb) != "dip":
            continue
        xb_rl, xp_rl = rl.get(eb, (xb_v7, xp_v7))
        if xb_rl >= xb_v7:
            continue
        age = xb_rl - eb
        hold = max(1, xb_v7 - eb)
        d = (_ret(xp_rl, ep) - _ret(xp_v7, ep)) * 100.0
        ages.append(age)
        (early_half if age <= hold / 2 else late_half).append(d)
    return {
        "n_early": len(ages),
        "age_p25_50_75": ([float(np.percentile(ages, q)) for q in (25, 50, 75)]
                          if ages else [float("nan")] * 3),
        "delta_early_half_mean": float(np.mean(early_half)) if early_half else float("nan"),
        "delta_late_half_mean": float(np.mean(late_half)) if late_half else float("nan"),
        "n_early_half": len(early_half), "n_late_half": len(late_half),
    }


def eval_cut(ckpt, name, cfg, policy, mu, sd, cd_idx):
    """Полная оценка одного разреза: калибровка, RL-гейт, K2, момент выхода."""
    df_ds = pd.read_parquet(cfg["ds"])
    mb = cfg["max_bar"]
    v7, op = _roll_v7(cfg["state"], cfg["sig"], df_ds, mb)
    op = np.asarray(op, float)
    ent = df_ds[df_ds["is_flip_entry"].astype(bool)]
    postag = {int(b) + 1: str(t) for b, t in zip(ent["bar"], ent["pos_tag"])}

    od, oacted = oracle_deltas(v7, op, postag)
    oracle_gate = new_gate(od, "ORACLE")
    rd = random_deltas(v7, op, postag, len(oacted), seed=0)
    random_gate = new_gate(rd, "RANDOM")
    rrob = random_sum_distribution(v7, op, postag, len(oacted))
    instr_valid = bool(oracle_gate["EFFECT_PRESENT"] and not random_gate["EFFECT_PRESENT"])

    rl, early = _roll_rl(cfg["state"], cfg["sig"], df_ds, policy, mu, sd, cd_idx, mb)
    rl_d, _acted = rl_deltas(v7, rl, op, postag)
    gate = new_gate(rl_d, "RL")
    reason_map, matched = exit_reason_map(cfg["state"], v7)
    od_by_eb = {t[0]: float(g) for t, g in zip(_dip_trades(v7, postag), od)}
    k2 = k2_tagged(v7, rl, op, postag, reason_map, od_by_eb)
    age = _exit_age_stats(v7, rl, postag)

    return {
        "cut": name, "instrument_valid": instr_valid,
        "n_dip": int(sum(1 for eb in v7 if postag.get(eb) == "dip")),
        "RL_gate": gate, "early_exit_driver": int(early),
        "ORACLE_sum_pp": oracle_gate["A_portfolio"]["sum_pp"],
        "RANDOM_sum_pp": random_gate["A_portfolio"]["sum_pp"],
        "RANDOM_share_pos": rrob["share_positive"],
        "K2": {"reason_match": matched / max(1, len(v7)), "groups": k2},
        "exit_timing": age,
    }


def eval_ckpt(ckpt, cuts=("train_2021", "tune_2024_oos")):
    """Оценить чекпойнт по разрезам. Если FF_PSL=1 — строить приор-политику
    (тот же K3, что при обучении; OOS считается тем же замороженным K3)."""
    import os
    mf = json.loads(open(MANIFEST).read())
    mu = np.array(mf["scaler_mu"], np.float32)
    sd = np.array(mf["scaler_sd"], np.float32)
    cd_idx = mf["obs_cols"].index("a_cooldown_remain")
    psl_kw = {}
    if os.environ.get("FF_PSL", "0") == "1":
        k3 = json.loads(open(os.environ.get(
            "FF_K3", f"{DATA}/k3_psl_classifier.json")).read())
        psl_kw = dict(psl_w=np.array(k3["w"], np.float32), psl_b=float(k3["b"]),
                      psl_alpha=float(os.environ.get("FF_ALPHA", 1.0)))
    model, _c = build_scaffold(_dummy_vec_env(39), SEED_MODEL, MANIFEST,
                               kl_coef=0.0, ent_coef=0.0, gamma=1.0,
                               n_steps=64, batch_size=64, n_epochs=1, **psl_kw)
    model.policy.load_state_dict(th.load(ckpt, map_location="cpu"),
                                 strict=not bool(psl_kw))
    model.policy.set_training_mode(False)
    return {name: eval_cut(ckpt, name, CUTS[name], model.policy, mu, sd, cd_idx)
            for name in cuts}


def _fmt(c):
    g = c["RL_gate"]["A_portfolio"]
    sl = c["K2"]["groups"]["sl"]
    return (f"[{c['cut']}] valid={c['instrument_valid']} nDip={c['n_dip']} | "
            f"RL Σ={g['sum_pp']:+.2f} CI[{g['ci_lo_pp']:+.2f},{g['ci_hi_pp']:+.2f}] "
            f"EFFECT={c['RL_gate']['EFFECT_PRESENT']} | SL acted={sl['rl_acted']}/{sl['n']} "
            f"Σsl={sl['sum_pp']:+.2f} accMean={sl['acted_mean_pp']} | "
            f"oracle={c['ORACLE_sum_pp']:+.1f} rand={c['RANDOM_sum_pp']:+.1f} | "
            f"exitAgeP50={c['exit_timing']['age_p25_50_75'][1]}")


def main():
    """CLI: оценка чекпойнта гейтом на train + tune-OOS."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cuts", default="train_2021,tune_2024_oos")
    args = ap.parse_args()
    res = eval_ckpt(args.ckpt, tuple(args.cuts.split(",")))
    for name, c in res.items():
        print(_fmt(c))
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
        print("saved", args.out)


if __name__ == "__main__":
    main()
