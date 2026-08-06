"""K4 предзапуск (amend3 §5): решающая проверка B1/B2 ДО бюджета.

B1 — направление advantage по группам: на первом rollout (клон+ent, driver)
средний GAE-advantage на STAY-сэмплированных dip-барах должен быть ВЫШЕ на
восстанавливающихся сделках, чем на SL-сделках (тогда PPO усиливает STAY на
recovering и толкает FLIP на SL). Плюс reward-local Δ(t) должен указывать верно.
B2 — ранжирует ли критик (V выше на recovering, чем на SL); константа = риск.
Дополнительно: сравнение credit-горизонта gae_lambda 0.95 (дефолт) vs 1.0 —
доходит ли последствие раннего выхода (SL за ~1000 баров) до advantage.

НЕ обучение: один rollout, train НЕ применяется (STOP в callback до train()).
"""
from __future__ import annotations

import argparse
import json
import sys
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
SEED_MODEL = f"{DATA}/bc_clone_v7_policy_reg_cd_vw2"
MANIFEST = SEED_MODEL + ".manifest.json"
STATE = f"{BASE}/sunday_tests/state_v0/state_v3_causal_2021-01_2024-03.parquet"
SIG = f"{DATA}/v7signals_2021.parquet"
DS = f"{DATA}/bc_clone_v7_2021.parquet"
FEE = 0.001


def _trade_tag_arrays(n_bars):
    """Для каждого абсолютного бара: group(0=none,1=sl,2=recovering) и exit_price
    сделки, к которой бар принадлежит (dip-сделки из label_trades)."""
    from label_exits import label_trades
    _st, tr, _bc = label_trades(STATE)
    dip = tr[tr["pos_tag"] == "dip"]
    grp = np.zeros(n_bars, np.int8)
    xp = np.zeros(n_bars, np.float64)
    for t in dip.itertuples():
        a, b = int(t.entry_bar), int(t.exit_bar)
        g = 1 if t.exit_reason == "sl" else 2
        grp[a:b + 1] = g
        xp[a:b + 1] = float(t.exit_price)
    return grp, xp


def _gae_lambda1(rewards, values, ep_starts):
    """GAE с gamma=1, lambda=1 (Монте-Карло): adv(t)=Rtg(t)-V(t), Rtg с бутстрапом
    хвоста V[-1]; сброс на границах эпизода."""
    n = len(rewards)
    adv = np.zeros(n)
    rtg = values[-1]
    for t in range(n - 1, -1, -1):
        if t + 1 < n and ep_starts[t + 1] > 0.5:
            rtg = 0.0                      # новая эпизода впереди
        rtg = rewards[t] + rtg
        adv[t] = rtg - values[t]
    return adv


class _OneRollout(BaseCallback):
    """Захватить буфер на первом rollout_end (ДО train) и остановить learn."""

    def __init__(self, mu, sd, col_index):
        """Замыкание на скейлер/индексы; result заполняется на rollout_end."""
        super().__init__()
        self.mu, self.sd, self.col_index = mu, sd, col_index
        self.done = False
        self.result = None

    def _on_step(self):
        return not self.done

    def _on_rollout_end(self):
        if self.done:
            return
        self.done = True
        rb = self.model.rollout_buffer
        N = rb.buffer_size * rb.n_envs
        obs = rb.observations.reshape(N, -1).astype(np.float32)
        actions = rb.actions.reshape(N, -1)
        adv95 = rb.advantages.reshape(-1).astype(np.float64)
        values = rb.values.reshape(-1).astype(np.float64)
        rewards = rb.rewards.reshape(-1).astype(np.float64)
        ep_starts = rb.episode_starts.reshape(-1).astype(np.float64)
        adv1 = _gae_lambda1(rewards, values, ep_starts)

        m = first_freedom_head_mask(obs, self.mu, self.sd, self.col_index, False)
        dip = m[:, HEAD_POSITION] > 0.5

        drv = self.model.env.venv.envs[0]
        sb = drv.step_bars
        L = min(len(sb), N)
        bars = np.array([sb[i][0] for i in range(L)], np.int64)
        exec_pos = np.array([sb[i][3] for i in range(L)], np.int64)
        # выровнять по L
        dip = dip[:L]; adv95 = adv95[:L]; adv1 = adv1[:L]; values = values[:L]
        obs = obs[:L]; actions = actions[:L]

        n_bars_state = int(bars.max()) + 2
        grp_arr, xp_arr = _trade_tag_arrays(max(n_bars_state, 1))
        grp = grp_arr[bars]                          # 1=sl 2=recovering
        exit_px = xp_arr[bars]

        # reward-local Δ(t) = log(open[bar+1]*(1-fee)/exit_price_of_trade)
        from spotrl.analysis.engine_run import load_ohlc
        op = load_ohlc(STATE)["Open"].to_numpy(np.float64)
        nxt = np.clip(bars + 1, 0, len(op) - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            delta_loc = np.log(op[nxt] * (1 - FEE)) - np.log(np.where(exit_px > 0, exit_px, np.nan))

        # p(FLIP)@dip и сэмплы FLIP
        with th.no_grad():
            d = self.model.policy.get_distribution(th.as_tensor(obs)).distribution
            p_flip = th.softmax(d[HEAD_POSITION].logits, dim=1)[:, 1].numpy()
        samp = actions[:, HEAD_POSITION]

        def grp_stats(gid, name):
            """Статистика advantage/V/p_flip по группе dip-баров (sl/rec)."""
            gm = dip & (grp == gid)
            stay = gm & (exec_pos == 0)
            return {
                "group": name, "n_dip_bars": int(gm.sum()),
                "adv_lambda0.95_stay_mean": float(adv95[stay].mean()) if stay.any() else float("nan"),
                "adv_lambda1.0_stay_mean": float(adv1[stay].mean()) if stay.any() else float("nan"),
                "reward_local_delta_mean": float(np.nanmean(delta_loc[gm])) if gm.any() else float("nan"),
                "critic_V_mean": float(values[gm].mean()) if gm.any() else float("nan"),
                "p_flip_mean": float(p_flip[gm].mean()) if gm.any() else float("nan"),
                "n_flip_sampled": int((samp[gm] == 1).sum()),
            }

        sl = grp_stats(1, "sl")
        rec = grp_stats(2, "recovering")
        crit_std = float(values[dip].std())
        b1_adv95 = rec["adv_lambda0.95_stay_mean"] > sl["adv_lambda0.95_stay_mean"]
        b1_adv1 = rec["adv_lambda1.0_stay_mean"] > sl["adv_lambda1.0_stay_mean"]
        b1_reward = (sl["reward_local_delta_mean"] > 0 > rec["reward_local_delta_mean"])
        b2_critic = rec["critic_V_mean"] > sl["critic_V_mean"] and crit_std > 1e-6

        self.result = {
            "gae_lambda_configured": float(self.model.gae_lambda),
            "n_dip_bars_total": int(dip.sum()),
            "n_sl_dip_bars": sl["n_dip_bars"], "n_recovering_dip_bars": rec["n_dip_bars"],
            "SL": sl, "RECOVERING": rec,
            "critic_V_std_over_dip": crit_std,
            "B1_direction_lambda0.95": bool(b1_adv95),
            "B1_direction_lambda1.0": bool(b1_adv1),
            "B1_reward_local_direction": bool(b1_reward),
            "B2_critic_ranks": bool(b2_critic),
            "p_flip_dip_overall": float(p_flip[dip].mean()) if dip.any() else float("nan"),
            "n_flip_sampled_dip": int((samp[dip] == 1).sum()),
            "n_flip_sampled_on_SL": int((samp[dip & (grp == 1)] == 1).sum()),
        }


def run(probe_steps=262144, ent_coef=0.03, beta=0.369, bdip=0.10, seed=0):
    """Собрать каркас+драйвер, один rollout, вернуть B1/B2-диагностику."""
    mf = json.loads(Path(MANIFEST).read_text())
    mu = np.array(mf["scaler_mu"], np.float32)
    sd = np.array(mf["scaler_sd"], np.float32)
    col_index = {n: i for i, n in enumerate(mf["obs_cols"])}
    df_ds = pd.read_parquet(DS)

    def make_env():
        """Среда+драйвер первой свободы для пробы (log_bars=True)."""
        env = _make_env(STATE, SIG)
        expert, is_sig = load_expert_arrays(df_ds, env._n_bars)
        drv = FirstFreedomDriver(env, expert, is_sig, honor_dip=True,
                                 enforce_v7_exit=True, min_hold=0)
        drv.log_bars = True
        return drv

    model, _clone = build_scaffold(
        DummyVecEnv([make_env]), SEED_MODEL, MANIFEST, kl_coef=beta,
        ent_coef=ent_coef, kl_dip_coef=bdip, gamma=1.0, n_steps=probe_steps,
        batch_size=probe_steps, n_epochs=1, seed=seed)
    cb = _OneRollout(mu, sd, col_index)
    model.learn(total_timesteps=probe_steps + 10, callback=cb, progress_bar=False)
    return cb.result


def main():
    """CLI: один rollout, печать B1/B2-диагностики."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=262144)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    r = run(probe_steps=args.steps)
    print("\n===== K4 ПРЕДЗАПУСК B1/B2 (rollout 1, клон) =====")
    print(f"gae_lambda(конфиг)={r['gae_lambda_configured']}  "
          f"dip-баров={r['n_dip_bars_total']} (SL={r['n_sl_dip_bars']} "
          f"recovering={r['n_recovering_dip_bars']})")
    for g in ("SL", "RECOVERING"):
        s = r[g]
        print(f"  {g:11s}: adv(λ.95)={s['adv_lambda0.95_stay_mean']:+.3e} "
              f"adv(λ1)={s['adv_lambda1.0_stay_mean']:+.3e} "
              f"Δloc={s['reward_local_delta_mean']:+.4f} "
              f"V={s['critic_V_mean']:+.4f} p(FLIP)={s['p_flip_mean']:.4%}")
    print(f"критик V std(dip)={r['critic_V_std_over_dip']:.3e}")
    print(f"B1 направление adv λ0.95 (rec>sl): {r['B1_direction_lambda0.95']}")
    print(f"B1 направление adv λ1.0  (rec>sl): {r['B1_direction_lambda1.0']}")
    print(f"B1 reward-local (Δsl>0>Δrec):     {r['B1_reward_local_direction']}")
    print(f"B2 критик ранжирует (Vrec>Vsl,std>0): {r['B2_critic_ranks']}")
    print(f"p(FLIP)@dip={r['p_flip_dip_overall']:.4%} "
          f"FLIP-сэмплов dip={r['n_flip_sampled_dip']} на SL={r['n_flip_sampled_on_SL']}")
    if args.out:
        Path(args.out).write_text(json.dumps(r, ensure_ascii=False, indent=2))
        print("saved", args.out)


if __name__ == "__main__":
    main()
