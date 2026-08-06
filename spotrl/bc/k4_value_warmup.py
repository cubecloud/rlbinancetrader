"""K4 ЦЕЛЕВОЙ warmup критика (amend3, корень B2): научить V РАНЖИРОВАТЬ исход.

Отличие от value_warmup.py: цель НЕ EV>=0.2 (точная регрессия trade-close
return-to-go, которая вне выборки не обобщается), а РАНЖИРОВАНИЕ двух типов
dip-сделок: V(восстанавливающийся dip) > V(SL-dip), std(V по dip) > 0. Это
грубее точного EV и достижимо (K3: obs отделяет SL от recovery, AUC 0.85).

Почему это чинит K4-коллапс: если V на dip-баре ≈ return-to-go до ЗАКРЫТИЯ
СДЕЛКИ (hold-исход), то advantage сэмплированного FLIP ≈ Δ(t) — trade-local
дельта раннего выхода: положительна на SL (толкает FLIP вверх), отрицательна
на recovering (толкает FLIP вниз). Самосогласовано с наградой (внутри холда
TD≈0). Заменяет КОНСТАНТНЫЙ bias-init критик, который не мог ранжировать.

Дисциплина: обучение критика ТОЛЬКО на 2021-01..2024-03 (эпоха 2024 залезает в
holdout 2026 → НЕ используется). Политика заморожена (логиты бит-в-бит,
проверяется). Сплит train/val ПО СДЕЛКАМ (первые 70% сделок train, последние
30% val) — ранжирование меряется на held-out сделках. Holdout не тронут.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from spotrl.bc import value_warmup as vw
from spotrl.bc.train_clone import apply_scaler, head0_logits
from spotrl.bc.value_warmup import (collect_epoch_rewards, trade_close_return_to_go,
                                    train_value, _predict_values, _trade_segments,
                                    explained_variance)

# --- ре-тюн регрессора под РАНЖИРОВАНИЕ (не EV): слабая регуляризация, больше
#     эпох, обучение на IN-POSITION барах (flat target=0 тянет V к среднему). ---
WEIGHT_DECAY = float(os.environ.get("FF_WD", 1e-5))
N_EPOCHS = int(os.environ.get("FF_EPOCHS", 80))
IN_POS_ONLY = os.environ.get("FF_INPOS_ONLY", "1") == "1"

_STRATEGY_RL = Path(__file__).resolve().parents[2] / "strategy_rl"
if str(_STRATEGY_RL) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_RL))

BASE = "/home/cubecloud/Data"
DATA = f"{BASE}/rlbinancetrader"
STATE = f"{BASE}/sunday_tests/state_v0/state_v3_causal_2021-01_2024-03.parquet"
SIG = f"{DATA}/v7signals_2021.parquet"
DS = f"{DATA}/bc_clone_v7_2021.parquet"
MODEL_IN = f"{DATA}/bc_clone_v7_policy_reg_cd_vw2"
MODEL_OUT = f"{DATA}/bc_clone_v7_policy_reg_cd_vwk4"
TRAIN_FRAC = 0.70


def _dip_sl_groups(n_bars):
    """grp[bar]: 1=SL-dip, 2=recovering-dip, 0=иначе (из label_trades)."""
    from label_exits import label_trades
    _st, tr, _bc = label_trades(STATE)
    grp = np.zeros(n_bars, np.int8)
    for t in tr[tr["pos_tag"] == "dip"].itertuples():
        grp[int(t.entry_bar):int(t.exit_bar) + 1] = 1 if t.exit_reason == "sl" else 2
    return grp


def run():
    """Warmup критика на 2021 + проверка ранжирования V_rec>V_sl на held-out."""
    from stable_baselines3 import PPO

    mf = json.loads(Path(MODEL_IN + ".manifest.json").read_text())
    mu = np.array(mf["scaler_mu"], np.float32)
    sd = np.array(mf["scaler_sd"], np.float32)
    col_index = {n: i for i, n in enumerate(mf["obs_cols"])}

    df = pd.read_parquet(DS)
    # X: стандартизованные 39-мерные obs (raw 38 + a_cooldown_active).
    raw = df[[c for c in mf["obs_cols"] if c != "a_cooldown_active"]].to_numpy(np.float32)
    cd = (df["a_cooldown_remain"].to_numpy() > 0).astype(np.float32)[:, None]
    X = apply_scaler(np.concatenate([raw, cd], axis=1), mu, sd)
    in_pos = df["a_in_position"].to_numpy(np.float32)

    cache = Path(DATA) / "value_warmup_rewards_2021.npy"
    R = collect_epoch_rewards("2021", STATE, SIG, df, cache)
    G = trade_close_return_to_go(R, in_pos)

    grp = _dip_sl_groups(int(df["bar"].max()) + 2)[df["bar"].to_numpy()]
    dip = grp > 0

    # сплит ПО СДЕЛКАМ: первые 70% сегментов (по появлению) → train.
    seg = _trade_segments(in_pos)
    seg_ids = np.array([s for s in np.unique(seg) if s >= 0])
    n_tr = int(len(seg_ids) * TRAIN_FRAC)
    tr_ids = set(seg_ids[:n_tr].tolist())
    is_tr_seg = np.array([(s in tr_ids) if s >= 0 else True for s in seg])
    # train-бары: обучаем на IN-POSITION train-барах (flat target=0 тянул бы V к
    # среднему и убивал ранжирование — диагностика показала V≈const). val = dip
    # held-out.
    inpos = in_pos > 0.5
    train_mask = is_tr_seg & (inpos if IN_POS_ONLY else np.ones_like(inpos, bool))
    train_idx = np.where(train_mask)[0]

    model = PPO.load(MODEL_IN, device="cpu")
    policy = model.policy
    d_before = head0_logits(policy, X)
    v_before = _predict_values(policy, X)

    vw.WEIGHT_DECAY = WEIGHT_DECAY          # ре-тюн под ранжирование
    train_value(policy, X[train_idx], G[train_idx], seed=0, n_epochs=N_EPOCHS)

    v_after = _predict_values(policy, X)
    d_after = head0_logits(policy, X)
    logits_bitexact = bool(np.array_equal(d_before, d_after))

    from sklearn.metrics import roc_auc_score

    def _rank(msk):
        d = dip & msk
        sl_ = d & (grp == 1)
        rc_ = d & (grp == 2)
        y = (grp[d] == 1).astype(int)
        auc = float(roc_auc_score(y, -v_after[d])) if y.min() != y.max() else float("nan")
        return (float(v_after[sl_].mean()), float(v_after[rc_].mean()), auc)

    v_sl_tr, v_rec_tr, auc_tr = _rank(is_tr_seg)
    # --- B2': ранжирование на HELD-OUT (val) dip-барах ---
    val = dip & (~is_tr_seg)
    sl = val & (grp == 1)
    rec = val & (grp == 2)
    v_sl, v_rec, auc_val = _rank(~is_tr_seg)
    v_std_dip = float(v_after[dip].std())
    ranks = bool(v_rec > v_sl and v_std_dip > 1e-6 and auc_val > 0.6)
    # advantage FLIP ≈ Δ_loc = exit_now − hold; критерий по знаку V vs hold-исход
    # проверяется отдельно в k4_prelaunch (реальный GAE). Здесь — ранжирование V.
    ev_val_dip = explained_variance(G[val], v_after[val])

    model.save(MODEL_OUT)
    manifest = dict(mf)
    manifest["k4_value_warmup"] = {
        "goal": "rank V(recovering-dip) > V(SL-dip), not EV>=0.2",
        "target": "trade_close_return_to_go_gamma1", "train_epoch": "2021_only",
        "train_frac_trades": TRAIN_FRAC, "logits_bitexact": logits_bitexact,
        "V_sl_val": v_sl, "V_recovering_val": v_rec, "V_std_dip": v_std_dip,
        "ranks_recovering_above_sl": ranks, "ev_val_dip": ev_val_dip,
        "source_model": MODEL_IN,
        "note": "критик прогрет на 2021 (holdout 2026 не тронут); политика бит-в-бит.",
    }
    Path(MODEL_OUT + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))

    res = {
        "logits_bitexact": logits_bitexact,
        "weight_decay": WEIGHT_DECAY, "n_epochs": N_EPOCHS,
        "in_pos_only": IN_POS_ONLY,
        "V_before_std_dip": float(v_before[dip].std()),
        "V_after_std_dip": v_std_dip,
        "V_sl_train": v_sl_tr, "V_rec_train": v_rec_tr, "AUC_train": auc_tr,
        "V_sl_val": v_sl, "V_recovering_val": v_rec, "AUC_val": auc_val,
        "V_margin_rec_minus_sl": v_rec - v_sl,
        "n_val_sl_bars": int(sl.sum()), "n_val_rec_bars": int(rec.sum()),
        "ev_val_dip": ev_val_dip,
        "B2_ranks": ranks, "saved": MODEL_OUT,
    }
    return res


def main():
    """CLI: warmup критика и печать B2'-ранжирования."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    r = run()
    print("=== K4 ЦЕЛЕВОЙ WARMUP КРИТИКА (ранжирование, 2021 only) ===")
    print(f"конфиг: wd={r['weight_decay']} epochs={r['n_epochs']} inpos_only={r['in_pos_only']}")
    print(f"логиты головы 0 бит-в-бит: {r['logits_bitexact']}")
    print(f"V std по dip: ДО={r['V_before_std_dip']:.3e} ПОСЛЕ={r['V_after_std_dip']:.3e}")
    print(f"TRAIN dip: V_sl={r['V_sl_train']:+.5f} V_rec={r['V_rec_train']:+.5f} "
          f"AUC(-V→SL)={r['AUC_train']:.3f}")
    print(f"VAL   dip: V_sl={r['V_sl_val']:+.5f} V_rec={r['V_recovering_val']:+.5f} "
          f"AUC(-V→SL)={r['AUC_val']:.3f} (запас rec−sl={r['V_margin_rec_minus_sl']:+.5f})")
    print(f"val-баров SL={r['n_val_sl_bars']} rec={r['n_val_rec_bars']}  "
          f"EV(val,dip)={r['ev_val_dip']:.4f}")
    print(f"B2' РАНЖИРУЕТ (V_rec>V_sl, std>0, AUC_val>0.6): {r['B2_ranks']}")
    print(f"СОХРАНЕНО: {r['saved']}")
    if args.out:
        Path(args.out).write_text(json.dumps(r, ensure_ascii=False, indent=2))
        print("saved", args.out)


if __name__ == "__main__":
    main()
