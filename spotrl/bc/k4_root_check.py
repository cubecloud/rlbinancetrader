"""K4 корневая проверка (координатор 2026-07-29): разделяющая проба.

Гипотеза координатора: amend1-коллапс, K4-константный критик, warmup-OOS-инверсия —
ОДИН корень: value-сигнал тонет в дисперсии γ=1 поминутного return-to-go, тогда
как решение по-сделочное. Даже классификационный сид критика размоется, т.к.
SB3 каждую эпоху регрессирует value-нет на те же GAE-возвраты.

Две задачи (бюджет НЕ запускать, holdout не трогать):
1. Радиус терминальной per-trade награды: (a) CB читает equity-кривую, не скаляр
   награды (подтверждено кодом spot_env: dd по world_equity, reward отдельный
   скаляр); (b) численно T7: Σ поминутного reward за сделку == log(1+return).
2. Разделяющая проба: засеять критик КЛАССИФИКАЦИОННЫМ (не MSE-на-магнитуде)
   ранжирующим лоссом (K3-сигнал), подтвердить B2' на INIT (V_rec>V_sl OOS),
   затем 300k sanity и ПЕРЕПРОВЕРИТЬ B2' В КОНЦЕ — держится или инвертируется.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from spotrl.bc.train_clone import apply_scaler, head0_logits
from spotrl.bc.value_warmup import (collect_epoch_rewards, trade_close_return_to_go,
                                    _predict_values, _trade_segments, value_params)

_STRATEGY_RL = Path(__file__).resolve().parents[2] / "strategy_rl"
if str(_STRATEGY_RL) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_RL))

BASE = "/home/cubecloud/Data"
DATA = f"{BASE}/rlbinancetrader"
STATE = f"{BASE}/sunday_tests/state_v0/state_v3_causal_2021-01_2024-03.parquet"
SIG = f"{DATA}/v7signals_2021.parquet"
DS = f"{DATA}/bc_clone_v7_2021.parquet"
MODEL_IN = f"{DATA}/bc_clone_v7_policy_reg_cd_vw2"
MODEL_OUT = f"{DATA}/bc_clone_v7_policy_reg_cd_vwk4cls"
TRAIN_FRAC = 0.70


def _load_arrays():
    """X(std 39), in_pos, grp(1=sl,2=rec), G(trade-close Rtg), split по сделкам."""
    mf = json.loads(Path(MODEL_IN + ".manifest.json").read_text())
    mu = np.array(mf["scaler_mu"], np.float32)
    sd = np.array(mf["scaler_sd"], np.float32)
    df = pd.read_parquet(DS)
    raw = df[[c for c in mf["obs_cols"] if c != "a_cooldown_active"]].to_numpy(np.float32)
    cd = (df["a_cooldown_remain"].to_numpy() > 0).astype(np.float32)[:, None]
    X = apply_scaler(np.concatenate([raw, cd], axis=1), mu, sd)
    in_pos = df["a_in_position"].to_numpy(np.float32)
    from label_exits import label_trades
    _st, tr, _bc = label_trades(STATE)
    grp_full = np.zeros(int(df["bar"].max()) + 2, np.int8)
    for t in tr[tr["pos_tag"] == "dip"].itertuples():
        grp_full[int(t.entry_bar):int(t.exit_bar) + 1] = 1 if t.exit_reason == "sl" else 2
    grp = grp_full[df["bar"].to_numpy()]
    cache = Path(DATA) / "value_warmup_rewards_2021.npy"
    R = collect_epoch_rewards("2021", STATE, SIG, df, cache)
    G = trade_close_return_to_go(R, in_pos)
    seg = _trade_segments(in_pos)
    ids = np.array([s for s in np.unique(seg) if s >= 0])
    n_tr = int(len(ids) * TRAIN_FRAC)
    tr_set = set(ids[:n_tr].tolist())
    is_tr = np.array([(s in tr_set) if s >= 0 else True for s in seg])
    return X, in_pos, grp, G, R, is_tr, df, tr


def part1b_t7(R, df, trades, n_show=4):
    """T7 численно: Σ reward за сделку == log(1+return_pct)? (телескоп log-эквити)."""
    bar2row = {int(b): i for i, b in enumerate(df["bar"].to_numpy())}
    dip = trades[trades["pos_tag"] == "dip"]
    rows = []
    for t in dip.head(n_show).itertuples():
        a, b = int(t.entry_bar), int(t.exit_bar)
        ra = bar2row.get(a)
        rb = bar2row.get(b)
        if ra is None or rb is None:
            continue
        sum_r = float(R[ra:rb + 1].sum())
        log_ret = float(np.log(1.0 + t.return_pct))
        rows.append({"entry_bar": a, "exit_bar": b, "return_pct": float(t.return_pct),
                     "sum_reward": sum_r, "log_1p_return": log_ret,
                     "abs_diff": abs(sum_r - log_ret)})
    return rows


def build_classification_seed(seed=0):
    """Засеять критик ранжирующим (классификационным) лоссом: p_SL(obs) логрег на
    train dip-барах → калибр. value-таргет → обучить value_net (политика бит-в-бит).
    Возврат B2' на INIT (val-dip AUC, запас)."""
    from stable_baselines3 import PPO
    import torch.nn.functional as F

    X, in_pos, grp, G, R, is_tr, df, _tr = _load_arrays()
    dip = grp > 0
    tr_dip = dip & is_tr

    # 1) p_SL: логрег (линейный, OOS-обобщается — K3 LogReg val AUC 0.83) на train dip.
    y_sl = (grp == 1).astype(int)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X[tr_dip], y_sl[tr_dip])
    p_sl = clf.predict_proba(X)[:, 1]

    # 2) калибр. value-таргет по групповым средним trade-close Rtg (train).
    v_sl_mean = float(G[tr_dip & (grp == 1)].mean())
    v_rec_mean = float(G[tr_dip & (grp == 2)].mean())
    target = (v_sl_mean * p_sl + v_rec_mean * (1.0 - p_sl)).astype(np.float32)

    model = PPO.load(MODEL_IN, device="cpu")
    policy = model.policy
    d_before = head0_logits(policy, X)

    # 3) обучить ТОЛЬКО value_net регрессией на НИЗКОДИСПЕРСНЫЙ калибр. таргет
    #    (train dip-бары + flat target 0 для масштаба вне позиции).
    flat_idx = np.where(in_pos < 0.5)[0]
    flat_sub = flat_idx[::max(1, len(flat_idx) // len(np.where(tr_dip)[0]))]
    tr_idx = np.concatenate([np.where(tr_dip)[0], flat_sub])
    tgt = target.copy(); tgt[flat_idx] = 0.0
    Xt = th.as_tensor(X); Tt = th.as_tensor(tgt)
    opt = th.optim.Adam(value_params(policy), lr=1e-3, weight_decay=1e-5)
    th.manual_seed(seed)
    for _ in range(60):
        policy.set_training_mode(True)
        perm = np.random.permutation(len(tr_idx))
        for i in range(0, len(tr_idx), 65536):
            b = tr_idx[perm[i:i + 65536]]
            feats = policy.extract_features(Xt[b])
            f = feats[0] if isinstance(feats, tuple) else feats
            _, lv = policy.mlp_extractor(f)
            v = policy.value_net(lv).squeeze(-1)
            loss = F.mse_loss(v, Tt[b])
            opt.zero_grad(); loss.backward(); opt.step()

    d_after = head0_logits(policy, X)
    bitexact = bool(np.array_equal(d_before, d_after))
    V = _predict_values(policy, X)
    b2 = _b2_report(V, grp, dip, is_tr)
    b2["logits_bitexact"] = bitexact
    b2["clf_val_auc"] = float(roc_auc_score(y_sl[dip & ~is_tr], p_sl[dip & ~is_tr]))
    b2["v_sl_mean_train"] = v_sl_mean
    b2["v_rec_mean_train"] = v_rec_mean

    model.save(MODEL_OUT)
    mf = json.loads(Path(MODEL_IN + ".manifest.json").read_text())
    mf["k4_cls_seed"] = {"goal": "rank V via K3 classification prior (not MSE-mag)",
                         "logits_bitexact": bitexact, "b2_init": b2,
                         "source": MODEL_IN}
    Path(MODEL_OUT + ".manifest.json").write_text(json.dumps(mf, ensure_ascii=False, indent=2))
    return b2


def _b2_report(V, grp, dip, is_tr):
    """AUC(-V→SL) и запас V_rec−V_sl на TRAIN и VAL dip-барах."""
    out = {}
    for name, msk in (("train", is_tr), ("val", ~is_tr)):
        d = dip & msk
        y = (grp[d] == 1).astype(int)
        auc = float(roc_auc_score(y, -V[d])) if y.min() != y.max() else float("nan")
        vsl = float(V[dip & msk & (grp == 1)].mean())
        vrec = float(V[dip & msk & (grp == 2)].mean())
        out[name] = {"auc_negV_to_SL": auc, "V_sl": vsl, "V_rec": vrec,
                     "margin_rec_minus_sl": vrec - vsl}
    out["V_std_dip"] = float(V[dip].std())
    out["B2_ranks_val"] = bool(out["val"]["margin_rec_minus_sl"] > 0
                               and out["V_std_dip"] > 1e-6
                               and out["val"]["auc_negV_to_SL"] > 0.6)
    return out


def eval_b2_ckpt(ckpt_path):
    """B2' на состоянии критика ПОСЛЕ обучения (загрузить state_dict в политику)."""
    from stable_baselines3 import PPO
    X, in_pos, grp, G, R, is_tr, df, _tr = _load_arrays()
    dip = grp > 0
    model = PPO.load(MODEL_OUT, device="cpu")
    model.policy.load_state_dict(th.load(ckpt_path, map_location="cpu"))
    V = _predict_values(model.policy, X)
    return _b2_report(V, grp, dip, is_tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["seed", "eval"], default="seed")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "seed":
        X, in_pos, grp, G, R, is_tr, df, tr = _load_arrays()
        t7 = part1b_t7(R, df, tr)
        print("=== Задача 1(b): T7 численно (Σ reward за сделку == log(1+return)) ===")
        for r in t7:
            print(f"  сделка [{r['entry_bar']}..{r['exit_bar']}] return={r['return_pct']:+.5f} "
                  f"Σreward={r['sum_reward']:+.6f} log(1+ret)={r['log_1p_return']:+.6f} "
                  f"|Δ|={r['abs_diff']:.2e}")
        b2 = build_classification_seed()
        print("\n=== Задача 2: B2' на INIT (классификационный сид) ===")
        print(f"логиты бит-в-бит: {b2['logits_bitexact']}  clf_val_AUC(p_SL)={b2['clf_val_auc']:.3f}")
        print(f"TRAIN dip AUC(-V→SL)={b2['train']['auc_negV_to_SL']:.3f} "
              f"запас={b2['train']['margin_rec_minus_sl']:+.5f}")
        print(f"VAL   dip AUC(-V→SL)={b2['val']['auc_negV_to_SL']:.3f} "
              f"запас={b2['val']['margin_rec_minus_sl']:+.5f}  V_std_dip={b2['V_std_dip']:.3e}")
        print(f"B2' INIT РАНЖИРУЕТ (val): {b2['B2_ranks_val']}")
        res = {"t7": t7, "b2_init": b2, "seed_model": MODEL_OUT}
    else:
        b2 = eval_b2_ckpt(args.ckpt)
        print("=== B2' ПОСЛЕ обучения (критик после in-loop апдейтов) ===")
        print(f"TRAIN dip AUC={b2['train']['auc_negV_to_SL']:.3f} "
              f"запас={b2['train']['margin_rec_minus_sl']:+.5f}")
        print(f"VAL   dip AUC={b2['val']['auc_negV_to_SL']:.3f} "
              f"запас={b2['val']['margin_rec_minus_sl']:+.5f}  V_std={b2['V_std_dip']:.3e}")
        print(f"B2' ПОСЛЕ РАНЖИРУЕТ (val): {b2['B2_ranks_val']}")
        res = {"b2_after": b2, "ckpt": args.ckpt}
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
        print("saved", args.out)


if __name__ == "__main__":
    main()
