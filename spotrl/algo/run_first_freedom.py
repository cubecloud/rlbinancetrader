"""Бюджетный прогон ПЕРВОЙ СВОБОДЫ (ранний выход dip) — amend1 (раздельный якорь
+ adaptive ent_coef). 1 сид sanity, обучение 2021-01..2024-03.

Правки amend1 (`prereg_rl_stage_amend1_2026-07-28.md`):
  * KL-якорь РАЗДЕЛЬНЫЙ: полная сила β на FORCED-барах, НУЛЬ на dip (в каркасе,
    `ppo_scaffold`). На dip политика СВОБОДНА двигаться — KL_dip расти ОЖИДАЕМО.
  * adaptive ent_coef: контроллер держит p(FLIP)@dip в коридоре [1%,5%] (таргет 3%).
  * enforce_v7_exit=True: агент может выйти РАНЬШЕ, но не пропустить выход v7.
STOP: KL_FORCED>0.05 устойчиво (копирование сломалось), p(FLIP)@dip вне коридора
устойчиво ПОСЛЕ warmup, коллапс энтропии, NaN, катастрофа эквити.
"""
from __future__ import annotations

import json
import os
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

BASE = "/home/cubecloud/Data"
DATA = f"{BASE}/rlbinancetrader"
SEED_MODEL = f"{DATA}/bc_clone_v7_policy_reg_cd_vw2"
MANIFEST = SEED_MODEL + ".manifest.json"
STATE = f"{BASE}/sunday_tests/state_v0/state_v3_causal_2021-01_2024-03.parquet"
SIG = f"{DATA}/v7signals_2021.parquet"
DS = f"{DATA}/bc_clone_v7_2021.parquet"

BETA = 0.369                 # KL-якорь на FORCED-барах (полная сила)
KL_FORCED_CAP = 0.05
ENT_COEF = 0.01              # фиксированный малый entropy-бонус (шум разведки)
# Контроллер держит p(FLIP)@dip в коридоре, регулируя СИЛУ dip-якоря
# (не ent_coef: энтропия управляет шумом, а не средним — см. ОБНОВЛЕНИЕ 4).
TARGET_PFLIP = 0.03          # таргет коридора [1%,5%]
BDIP_LR = float(os.environ.get("FF_BDIP_LR", 0.0))    # 0 → ФИКСИРОВАННЫЙ якорь
BDIP_INIT = float(os.environ.get("FF_BDIP_INIT", 0.05))  # фикс. слабый dip-якорь
BDIP_MIN, BDIP_MAX = 1e-3, 5.0
EQUITY_FLOOR = 0.20          # катастрофа наградной эквити (старт 1.0)
WARMUP_ROLLOUTS = 12
N_STEPS = 8192
N_EPOCHS = 5
TOTAL = int(os.environ.get("FF_TOTAL", 2_000_000))
SEED = int(os.environ.get("FF_SEED", 0))
OUT = Path(os.environ.get("FF_OUT", f"{DATA}/first_freedom_amend1_seed{SEED}"))


class AdaptiveLogger(BaseCallback):
    """Метрики каждого rollout + adaptive ent_coef + STOP-условия (amend1)."""

    def __init__(self, mu, sd, col_index, clone, metrics_path, ckpt_every=32):
        """Замыкание на скейлер/клон/пути; счётчики устойчивых нарушений."""
        super().__init__()
        self.mu, self.sd, self.col_index, self.clone = mu, sd, col_index, clone
        self.metrics_path = metrics_path
        self.ckpt_every = ckpt_every
        self.rollout = 0
        self.cum_reward = 0.0
        self._klf_bad = self._pflip_bad = self._ent_bad = 0
        self.stop_reason = None

    def _on_step(self) -> bool:
        """Прервать learn(), если взведён STOP-флаг."""
        return self.stop_reason is None

    def _on_rollout_end(self) -> None:
        """Метрики, adaptive ent_coef, STOP."""
        self.rollout += 1
        rb = self.model.rollout_buffer
        N = rb.buffer_size * rb.n_envs
        obs = rb.observations.reshape(N, -1).astype(np.float32)
        actions = rb.actions.reshape(N, -1)
        rewards = rb.rewards.reshape(-1).astype(np.float64)
        m = first_freedom_head_mask(obs, self.mu, self.sd, self.col_index, False)
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
        emp_flip_dip = (float((actions[dip, HEAD_POSITION] == 1).mean())
                        if n_dip else float("nan"))
        flip_at_dip = int((actions[dip, HEAD_POSITION] == 1).sum()) if n_dip else 0
        ent_dip = float(ent[dip].mean()) if n_dip else float("nan")
        kl_dip = float(kl[dip].mean()) if n_dip else float("nan")
        kl_forced = float(kl[forced].mean()) if forced.sum() else float("nan")
        roll_reward = float(rewards.sum())
        self.cum_reward += roll_reward
        reward_equity = float(np.exp(self.cum_reward))
        try:
            early = int(self.model.env.envs[0].early_exit_count)
        except Exception:
            early = -1
        # p_flip_dip=nan только при коллапсе dip (n_dip=0) — это ловит счётчик
        # коллапса, не NaN-стоп; NaN-стоп только по forced-KL/награде.
        nan_flag = bool(np.isnan(kl_forced) or not np.isfinite(roll_reward))

        # dip-экспозиция схлопнулась (агент выходит из dip мгновенно) → метрика
        # p(FLIP)@dip на горстке баров недостоверна; не трогаем контроллер, копим
        # счётчик коллапса (STOP-условие).
        dip_collapsed = n_dip < 50
        # --- adaptive dip-anchor (FF_ADAPT=1): p(FLIP)@dip>target → якорь СЛАБ →
        #     усилить; <target → ослабить. FF_ADAPT=0 → ФИКСИРОВАННЫЙ якорь. ---
        if BDIP_LR > 0 and not dip_collapsed:
            meas = min(max(p_flip_dip, 1e-5), 0.999)
            log_b = np.log(self.model._kl_dip_coef)
            log_b += BDIP_LR * (np.log(meas) - np.log(TARGET_PFLIP))
            self.model._kl_dip_coef = float(np.clip(np.exp(log_b), BDIP_MIN, BDIP_MAX))

        rec = {"rollout": self.rollout, "steps": int(self.num_timesteps),
               "kl_forced": kl_forced, "kl_dip": kl_dip,
               "p_flip_dip": p_flip_dip, "emp_flip_dip": emp_flip_dip,
               "flip_at_dip": flip_at_dip, "early_exit_vs_v7": early,
               "entropy_dip": ent_dip, "kl_dip_coef": self.model._kl_dip_coef,
               "dip_frac": n_dip / N, "n_dip": n_dip,
               "rollout_reward": roll_reward, "reward_equity": reward_equity,
               "time": time.time()}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        # --- STOP (amend1 §4): коллапс dip-экспозиции / KL_forced>cap / NaN /
        #     катастрофа эквити. p(FLIP)@dip СНЯТ с STOP (только лог-диагностика —
        #     метрика плохо определена для поглощающего выхода). ---
        self._klf_bad = self._klf_bad + 1 if kl_forced > KL_FORCED_CAP else 0
        self._collapse = getattr(self, "_collapse", 0) + 1 if dip_collapsed else 0
        self._eq_bad = (getattr(self, "_eq_bad", 0) + 1
                        if reward_equity < EQUITY_FLOOR else 0)
        if nan_flag:
            self.stop_reason = "NaN/inf в метриках"
        elif self._collapse >= 3:
            self.stop_reason = (f"dip-экспозиция схлопнулась 3 rollout подряд "
                                f"(n_dip={n_dip}) — агент выходит из dip мгновенно")
        elif self._klf_bad >= 3:
            self.stop_reason = (f"KL_forced>{KL_FORCED_CAP} 3 rollout подряд "
                                f"(={kl_forced:.4f}) — копирование сломалось")
        elif self._eq_bad >= 3:
            self.stop_reason = (f"наградная эквити < {EQUITY_FLOOR} 3 rollout "
                                f"подряд (={reward_equity:.3f}) — катастрофа")

        if self.rollout % self.ckpt_every == 0 or self.stop_reason:
            th.save(self.model.policy.state_dict(),
                    str(OUT / f"ckpt_{self.num_timesteps}.pt"))
        if self.rollout % 4 == 0 or self.rollout <= 3 or self.stop_reason:
            print(f"[r{self.rollout} s{self.num_timesteps}] KLf={kl_forced:.4e} "
                  f"KLd={kl_dip:.3e} p(FLIP)@dip={p_flip_dip:.3%} ent={ent_dip:.2e} "
                  f"bdip={self.model._kl_dip_coef:.4f} early={early} eq={reward_equity:.4f} "
                  f"dipf={n_dip / N:.3f}"
                  + (f"  STOP: {self.stop_reason}" if self.stop_reason else ""))


def main():
    """Собрать драйвер+каркас (раздельный якорь), запустить обучение."""
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
        """Среда копирования + драйвер первой свободы (enforce_v7_exit)."""
        env = _make_env(STATE, SIG)
        expert, is_sig = load_expert_arrays(df_ds, env._n_bars)
        return FirstFreedomDriver(env, expert, is_sig, honor_dip=True,
                                  enforce_v7_exit=True)

    model, clone = build_scaffold(
        DummyVecEnv([make_env]), SEED_MODEL, MANIFEST, kl_coef=BETA,
        ent_coef=ENT_COEF, kl_dip_coef=BDIP_INIT, gamma=1.0, n_steps=N_STEPS,
        batch_size=N_STEPS, n_epochs=N_EPOCHS, seed=SEED)

    manifest = {
        "run": "first_freedom_early_exit_dip_amend1", "seed": SEED,
        "amend": "prereg_rl_stage_amend1_2026-07-28.md",
        "seed_model": SEED_MODEL, "beta_kl_anchor_forced": BETA,
        "kl_forced_cap": KL_FORCED_CAP,
        "anchor": "forced=beta(0.369); dip=fixed weak bdip=0.05 (amend1 §4)",
        "ent_coef": ENT_COEF, "dip_anchor_adaptive": (BDIP_LR > 0),
        "bdip_fixed": BDIP_INIT, "bdip_lr": BDIP_LR,
        "gate": "p(FLIP)@dip=диагностика (снят с STOP); успех=парная per-trade 4.2",
        "auto_stop": "dip-collapse / KL_forced>0.05 / NaN / equity<0.20",
        "warmup_rollouts": WARMUP_ROLLOUTS,
        "gamma": 1.0, "n_steps": N_STEPS, "n_epochs": N_EPOCHS,
        "total_timesteps": TOTAL, "advantage_norm": "dip_subset",
        "reward": "log_equity_increment_gamma1",
        "driver": "FirstFreedomDriver honor_dip=True enforce_v7_exit=True",
        "train_window": "2021-01..2024-03", "state": STATE, "signals": SIG,
        "holdout_untouched": "2026-01..2026-07", "started": time.time()}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    cb = AdaptiveLogger(mu, sd, col_index, clone, str(metrics_path),
                        ckpt_every=int(os.environ.get("FF_CKPT", 32)))
    mode = "adaptive" if BDIP_LR > 0 else "fixed"
    print(f"=== START amend1 first-freedom seed={SEED} beta_forced={BETA} "
          f"dip-anchor={mode} bdip={BDIP_INIT} (p(FLIP) диагностика) ===")
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
