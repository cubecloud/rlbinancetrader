"""K4 бюджетный прогон ПЕРВОЙ СВОБОДЫ (amend3): мин-холд + β_dip=0.10 +
ent_coef=0.03, разведка на уровне сделки через наблюдаемый мин-холд.

amend3 (`prereg_rl_stage_amend3_2026-07-29.md`):
  * мин-холд по наблюдаемому a_days_in_trade (60 баров ≈ 0.0245): убирает
    мгновенное поглощение (выход на баре входа), из-за которого dip-экспозиция
    схлопывалась в amend1; enforce_v7_exit имеет приоритет над мин-холдом;
  * β_forced=0.369 (полный якорь копирования), β_dip=0.10 (приор «держи dip» —
    клон на dip = STAY; сопротивляется коллапсу «выходить везде»);
  * ent_coef=0.03 (разведка только через энтропию, on-policy корректно);
  * маска кредитует голову 0 на dip-барах возраста ≥ мин-холд (тот же obs-признак).
Лог каждого rollout: KL_forced/KL_dip, p(FLIP)@dip, dip-экспозиция, доля
затронутых SL-сделок, распределение момента раннего выхода, наградная эквити.
STOP: KL_forced>0.05 уст. / dip-коллапс / NaN / катастрофа эквити.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from spotrl.bc.build_clone_dataset import _make_env
from spotrl.envs.first_freedom_driver import FirstFreedomDriver, load_expert_arrays
from spotrl.algo.ppo_scaffold import build_scaffold
from spotrl.algo.head_masking import first_freedom_head_mask
from spotrl.spec.actions import HEAD_POSITION

_STRATEGY_RL = Path(__file__).resolve().parents[2] / "strategy_rl"
if str(_STRATEGY_RL) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_RL))

BASE = "/home/cubecloud/Data"
DATA = f"{BASE}/rlbinancetrader"
SEED_MODEL = os.environ.get("FF_SEED_MODEL",
                            f"{DATA}/bc_clone_v7_policy_reg_cd_vw2")
MANIFEST = os.environ.get("FF_MANIFEST", SEED_MODEL + ".manifest.json")
STATE = f"{BASE}/sunday_tests/state_v0/state_v3_causal_2021-01_2024-03.parquet"
SIG = f"{DATA}/v7signals_2021.parquet"
DS = f"{DATA}/bc_clone_v7_2021.parquet"

BETA = 0.369
KL_FORCED_CAP = 0.05
ENT_COEF = float(os.environ.get("FF_ENT", 0.03))
BDIP = float(os.environ.get("FF_BDIP", 0.10))
MIN_HOLD_DAYS = float(os.environ.get("FF_MINHOLD", 0.0245))   # ≈60 баров
EQUITY_FLOOR = 0.20
N_STEPS = 8192
N_EPOCHS = 5
TOTAL = int(os.environ.get("FF_TOTAL", 2_000_000))
SEED = int(os.environ.get("FF_SEED", 0))
OUT = Path(os.environ.get("FF_OUT", f"{DATA}/first_freedom_k4_seed{SEED}"))


def _sl_bar_array(n_bars):
    """grp[bar]: 1=SL-dip-сделка, 2=recovering-dip, 0=иначе."""
    from label_exits import label_trades
    _st, tr, _bc = label_trades(STATE)
    grp = np.zeros(n_bars, np.int8)
    for t in tr[tr["pos_tag"] == "dip"].itertuples():
        grp[int(t.entry_bar):int(t.exit_bar) + 1] = 1 if t.exit_reason == "sl" else 2
    return grp


class K4Logger(BaseCallback):
    """Метрики каждого rollout + SL-touch + момент выхода + STOP (amend3)."""

    def __init__(self, mu, sd, col_index, clone, grp, metrics_path, ckpt_every=32):
        super().__init__()
        self.mu, self.sd, self.col_index, self.clone = mu, sd, col_index, clone
        self.grp = grp
        self.metrics_path = metrics_path
        self.ckpt_every = ckpt_every
        self.rollout = 0
        self.cum_reward = 0.0
        self._klf_bad = self._collapse = self._eq_bad = 0
        self.stop_reason = None

    def _on_step(self):
        return self.stop_reason is None

    def _on_rollout_start(self):
        drv = self.model.env.venv.envs[0]
        drv.step_bars = []
        drv.early_exit_log = []

    def _on_rollout_end(self):
        self.rollout += 1
        rb = self.model.rollout_buffer
        N = rb.buffer_size * rb.n_envs
        obs = rb.observations.reshape(N, -1).astype(np.float32)
        actions = rb.actions.reshape(N, -1)
        rewards = rb.rewards.reshape(-1).astype(np.float64)
        m = first_freedom_head_mask(obs, self.mu, self.sd, self.col_index,
                                    False, MIN_HOLD_DAYS)
        dip = m[:, HEAD_POSITION] > 0.5
        forced = ~dip
        n_dip = int(dip.sum())

        ot = th.as_tensor(obs)
        with th.no_grad():
            cur = self.model.policy.get_distribution(ot).distribution
            ref = self.clone.get_distribution(ot).distribution
            p_flip = th.softmax(cur[HEAD_POSITION].logits, dim=1)[:, 1].numpy()
            ent = cur[HEAD_POSITION].entropy().numpy()
            kl = th.distributions.kl_divergence(
                ref[HEAD_POSITION], cur[HEAD_POSITION]).numpy()
        p_flip_dip = float(p_flip[dip].mean()) if n_dip else float("nan")
        ent_dip = float(ent[dip].mean()) if n_dip else float("nan")
        kl_dip = float(kl[dip].mean()) if n_dip else float("nan")
        kl_forced = float(kl[forced].mean()) if forced.sum() else float("nan")
        roll_reward = float(rewards.sum())
        self.cum_reward += roll_reward
        reward_equity = float(np.exp(self.cum_reward))

        drv = self.model.env.venv.envs[0]
        early = int(drv.early_exit_count)
        eelog = list(drv.early_exit_log)
        ee_bars = np.array([b for b, a in eelog], np.int64)
        ee_age = np.array([a for b, a in eelog], np.int64)
        sl_touched = int(np.unique(ee_bars[self.grp[ee_bars] == 1]).size) \
            if ee_bars.size else 0
        rec_touched = int(np.unique(ee_bars[self.grp[ee_bars] == 2]).size) \
            if ee_bars.size else 0
        # распределение момента выхода внутри удержания (возраст в барах)
        age_q = ([float(np.percentile(ee_age, q)) for q in (25, 50, 75)]
                 if ee_age.size else [float("nan")] * 3)

        nan_flag = bool(np.isnan(kl_forced) or not np.isfinite(roll_reward))
        dip_collapsed = n_dip < 50

        rec = {"rollout": self.rollout, "steps": int(self.num_timesteps),
               "kl_forced": kl_forced, "kl_dip": kl_dip,
               "p_flip_dip": p_flip_dip, "entropy_dip": ent_dip,
               "early_exit_vs_v7": early, "minhold_blocked": int(drv.minhold_blocked),
               "sl_trades_touched": sl_touched, "recovering_touched": rec_touched,
               "exit_age_p25_50_75": age_q,
               "dip_frac": n_dip / N, "n_dip": n_dip,
               "rollout_reward": roll_reward, "reward_equity": reward_equity,
               "time": time.time()}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        self._klf_bad = self._klf_bad + 1 if kl_forced > KL_FORCED_CAP else 0
        self._collapse = self._collapse + 1 if dip_collapsed else 0
        self._eq_bad = self._eq_bad + 1 if reward_equity < EQUITY_FLOOR else 0
        if nan_flag:
            self.stop_reason = "NaN/inf"
        elif self._collapse >= 3:
            self.stop_reason = f"dip-коллапс 3 rollout (n_dip={n_dip})"
        elif self._klf_bad >= 3:
            self.stop_reason = f"KL_forced>{KL_FORCED_CAP} 3 rollout (={kl_forced:.4f})"
        elif self._eq_bad >= 3:
            self.stop_reason = f"эквити<{EQUITY_FLOOR} 3 rollout (={reward_equity:.3f})"

        if self.rollout % self.ckpt_every == 0 or self.stop_reason:
            th.save(self.model.policy.state_dict(),
                    str(OUT / f"ckpt_{self.num_timesteps}.pt"))
        if self.rollout % 4 == 0 or self.rollout <= 3 or self.stop_reason:
            print(f"[r{self.rollout} s{self.num_timesteps}] KLf={kl_forced:.3e} "
                  f"KLd={kl_dip:.3e} pF@dip={p_flip_dip:.3%} ent={ent_dip:.2e} "
                  f"early={early} SLtouch={sl_touched} recTouch={rec_touched} "
                  f"exitAge50={age_q[1]} eq={reward_equity:.3f} dipf={n_dip/N:.3f}"
                  + (f"  STOP: {self.stop_reason}" if self.stop_reason else ""))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics_path = OUT / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    mf = json.loads(Path(MANIFEST).read_text())
    mu = np.array(mf["scaler_mu"], np.float32)
    sd = np.array(mf["scaler_sd"], np.float32)
    col_index = {n: i for i, n in enumerate(mf["obs_cols"])}
    df_ds = pd.read_parquet(DS)

    def make_env():
        env = _make_env(STATE, SIG)
        expert, is_sig = load_expert_arrays(df_ds, env._n_bars)
        drv = FirstFreedomDriver(env, expert, is_sig, honor_dip=True,
                                 enforce_v7_exit=True, min_hold_days=MIN_HOLD_DAYS)
        drv.log_bars = True
        return drv

    model, clone = build_scaffold(
        DummyVecEnv([make_env]), SEED_MODEL, MANIFEST, kl_coef=BETA,
        ent_coef=ENT_COEF, kl_dip_coef=BDIP, gamma=1.0, n_steps=N_STEPS,
        batch_size=N_STEPS, n_epochs=N_EPOCHS, seed=SEED,
        min_hold_days=MIN_HOLD_DAYS)

    grp = _sl_bar_array(model.env.venv.envs[0].env._n_bars + 2)

    manifest = {
        "run": "first_freedom_k4", "seed": SEED,
        "amend": "prereg_rl_stage_amend3_2026-07-29.md",
        "seed_model": SEED_MODEL, "beta_forced": BETA, "beta_dip": BDIP,
        "ent_coef": ENT_COEF, "min_hold_days": MIN_HOLD_DAYS,
        "min_hold_bars_approx": 60, "gamma": 1.0,
        "gae_lambda": float(model.gae_lambda),
        "n_steps": N_STEPS, "n_epochs": N_EPOCHS, "total_timesteps": TOTAL,
        "reward": "log_equity_increment_gamma1", "advantage_norm": "dip_subset",
        "driver": "FirstFreedomDriver honor_dip enforce_v7_exit min_hold_days",
        "auto_stop": "dip-collapse / KL_forced>0.05 / NaN / equity<0.20",
        "train_window": "2021-01..2024-03", "holdout_untouched": "2026-01..07",
        "started": time.time()}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    cb = K4Logger(mu, sd, col_index, clone, grp, str(metrics_path),
                  ckpt_every=int(os.environ.get("FF_CKPT", 16)))
    print(f"=== START K4 seed={SEED} beta_forced={BETA} beta_dip={BDIP} "
          f"ent={ENT_COEF} min_hold_days={MIN_HOLD_DAYS} total={TOTAL} ===")
    model.learn(total_timesteps=TOTAL, callback=cb, progress_bar=False)
    th.save(model.policy.state_dict(), str(OUT / "final_policy.pt"))
    manifest.update({"finished": time.time(), "stop_reason": cb.stop_reason,
                     "rollouts": cb.rollout,
                     "final_reward_equity": float(np.exp(cb.cum_reward))})
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"=== DONE rollouts={cb.rollout} stop={cb.stop_reason} "
          f"eq={np.exp(cb.cum_reward):.4f} ===")


if __name__ == "__main__":
    main()
