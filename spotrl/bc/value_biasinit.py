"""BIAS-INIT критика перед PPO (замена недостижимого гейта EV>=0.2).

ЗАЧЕМ. value-warmup (value_warmup.py) показал: на ~434 сделках trade-close
return-to-go вне выборки НЕ обобщается (val EV≈0), поэтому порог EV>=0.2
недостижим — это свойство данных, не дефект механики. Но реальная задача warmup
была скромнее: убрать SCALE-MISMATCH между СЛУЧАЙНЫМ критиком (выход O(1)) и
таргетом-advantage (O(0.03)), из-за которого первый же PPO-апдейт advantage
считается по критику с EV≈-272 и разрушает политику.

Тот же эффект достигается ДЁШЕВО и БЕЗ риска мемоизации (рекомендация handoff
impl_gate_final_valuewarmup): инициализировать ПОСЛЕДНИЙ линейный слой критика
(`policy.value_net`, Linear(hidden,1)) так, чтобы выход имел ПРАВИЛЬНЫЙ МАСШТАБ:
  * bias = mean(G)  — среднее return-to-go до конца сделки по ВСЕМ барам;
  * weight = 0      — выход = константа (= bias) для любого входа.
Тогда EV критика по всем барам = 1 − Var(G−mean)/Var(G) = РОВНО 0 (не -272), а
логиты политики бит-в-бит неизменны (трогаем только value_net).

ВАЖНО: weight=0 НЕ замораживает критика. Градиент MSE dL/dW = 2(V−G)·h, h≠0 →
на первом же PPO-апдейте веса уходят от нуля и критик доучивается on-policy.
bias-init фиксирует ТОЛЬКО стартовый масштаб, не «обучает» ценность на шуме.

Семя reg_cd_vw2 = политика reg_cd (бит-в-бит) + масштабно-инициализированный
критик. Именно оно идёт в PPO.

Run (env rlbinancetrader):
  python -m spotrl.bc.value_biasinit \
      --model-in  /home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy_reg_cd \
      --model-out /home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy_reg_cd_vw2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from spotrl.bc.train_clone import apply_scaler, head0_logits, load_pooled
from spotrl.bc.value_warmup import (collect_returns, explained_variance,
                                    _predict_values)

BIASINIT_SEED = 0  # для протокола; сама инициализация детерминирована


def bias_init_critic(policy, target_mean: float) -> Dict:
    """Инициализировать последний слой критика на КОНСТАНТНЫЙ выход = target_mean.

    `policy.value_net` — Linear(hidden, 1): обнуляем weight, ставим bias=target_mean.
    Выход критика становится ровно target_mean для любого входа (масштаб таргета).
    Голова политики (action_net, pi-ветка) НЕ трогается.

    Returns:
        Диагностика {bias_before, bias_after, weight_absmax_after}.
    """
    import torch as th
    lin = policy.value_net
    bias_before = float(lin.bias.data.detach().cpu().numpy().reshape(-1)[0])
    with th.no_grad():
        lin.weight.data.zero_()
        lin.bias.data.fill_(float(target_mean))
    weight_absmax = float(lin.weight.data.abs().max().cpu().numpy())
    bias_after = float(lin.bias.data.detach().cpu().numpy().reshape(-1)[0])
    return {"bias_before": bias_before, "bias_after": bias_after,
            "weight_absmax_after": weight_absmax}


def run_bias_init(data_dir: str, model_in: str, model_out: str,
                  base: str = None) -> Dict:
    """Полный bias-init: mean(G) из кэша reward, инициализация критика, проверка
    (EV_all до/после, логиты бит-в-бит), сохранение reg_cd_vw2 + манифест."""
    from stable_baselines3 import PPO

    mf = json.loads(Path(model_in + ".manifest.json").read_text())
    mu = np.array(mf["scaler_mu"], np.float32)
    sd = np.array(mf["scaler_sd"], np.float32)
    X_raw, y, w, meta = load_pooled(data_dir)
    X = apply_scaler(X_raw, mu, sd)

    # G = trade-close return-to-go (тот же таргет, что value_warmup); из кэша.
    G, _ = collect_returns(data_dir, base)
    g_mean = float(G.mean())

    model = PPO.load(model_in, device="cpu")
    policy = model.policy

    d_before = head0_logits(policy, X)
    v_before = _predict_values(policy, X)
    ev_all_before = explained_variance(G, v_before)

    bi = bias_init_critic(policy, g_mean)

    v_after = _predict_values(policy, X)
    ev_all_after = explained_variance(G, v_after)

    d_after = head0_logits(policy, X)
    logits_bitexact = bool(np.array_equal(d_before, d_after))
    max_logit_delta = float(np.abs(d_before - d_after).max())
    # выход критика — константа = g_mean на всех барах?
    v_const_absdev = float(np.abs(v_after - g_mean).max())

    result = {
        "g_mean": g_mean,
        "ev_all_before": ev_all_before, "ev_all_after": ev_all_after,
        "logits_bitexact": logits_bitexact, "max_logit_delta": max_logit_delta,
        "v_const_absdev": v_const_absdev,
        "bias_before": bi["bias_before"], "bias_after": bi["bias_after"],
        "weight_absmax_after": bi["weight_absmax_after"],
        "G_stats": {"mean": g_mean, "std": float(G.std()),
                    "min": float(G.min()), "max": float(G.max())},
    }

    model.save(model_out)
    manifest = dict(mf)
    manifest["value_biasinit"] = {
        "seed": BIASINIT_SEED, "method": "value_net.weight=0, bias=mean(G)",
        "g_mean": g_mean, "target": "trade_close_return_to_go_gamma1",
        "ev_all_before": ev_all_before, "ev_all_after": ev_all_after,
        "logits_bitexact": logits_bitexact, "max_logit_delta": max_logit_delta,
        "v_const_absdev": v_const_absdev, "source_model": model_in,
        "note": ("bias-init критика вместо недостижимого EV>=0.2. Последний слой "
                 "value_net: weight=0, bias=mean(G)=0.0044 → EV_all из -272 стал "
                 "~0 (константа-среднее), логиты политики бит-в-бит неизменны. "
                 "weight=0 НЕ замораживает критика: PPO доучивает on-policy. Это "
                 "семя reg_cd_vw2 для PPO. См. impl_biasinit_prereg_2026-07-28."),
    }
    Path(model_out + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    result["saved"] = model_out
    return result


def main() -> None:
    """CLI: bias-init критика, печать EV_all до/после + инвариант логитов."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/cubecloud/Data/rlbinancetrader")
    ap.add_argument(
        "--model-in",
        default="/home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy_reg_cd")
    ap.add_argument(
        "--model-out",
        default="/home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy_reg_cd_vw2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    r = run_bias_init(args.data, args.model_in, args.model_out)
    print("=== BIAS-INIT критика (масштаб: EV_all -272→~0, логиты неизменны) ===")
    print(f"mean(G) = {r['g_mean']:.6f}  (bias критика после = {r['bias_after']:.6f})")
    print(f"[SB3-метрика, ВСЕ бары] EV_all ДО={r['ev_all_before']:.4f} "
          f"→ ПОСЛЕ={r['ev_all_after']:.6f} (≈0 = константа-среднее, масштаб выправлен)")
    print(f"выход критика константен: max|V−mean(G)|={r['v_const_absdev']:.2e}, "
          f"weight_absmax={r['weight_absmax_after']:.2e}")
    print(f"логиты головы 0 БИТ-В-БИТ неизменны: {r['logits_bitexact']} "
          f"(max|Δd|={r['max_logit_delta']:.2e})")
    print(f"СОХРАНЕНО: {r['saved']}")
    if args.out:
        Path(args.out).write_text(json.dumps(r, ensure_ascii=False, indent=2))
        print(f"отчёт: {args.out}")


if __name__ == "__main__":
    main()
