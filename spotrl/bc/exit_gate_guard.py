"""ШАГ-1 страж финального гейта выхода: доказать, что ВСЕ текущие провалы
per-decision recall/overlap выхода лежат ЦЕЛИКОМ внутри класса механического
тейка M = {exit_reason=='tp' И m_exit_flag==0} (среда закрывает сама,
spot_env.py:292-294), БЕЗ снижения порога 0.99.

M — чистая функция v7-меток (НЕ логитов клона): джерримендеринг-ассерт.
must-copy = unambig_exit \ M = legflip | b2b | signal-confirmed-tp (tp&m_exit_flag==1).

Скрипт НЕ переобучает reg_cd. Считает на saved reg_cd (in-sample, обе эпохи):
  (i)  overlap_exit_nonlatch против reference = must-copy (min d по must-copy);
  (ii) recall reg_cd на must-copy;
  и печатает ЛИЧНОСТЬ каждого промаха (в M или нет).
Дополнительно воспроизводит out-of-epoch клоны гейта (пункт 4) и проверяет,
что промахи по recall ⊆ M.

Run (env rlbinancetrader):
  python -m spotrl.bc.exit_gate_guard \
      --model /home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy_reg_cd
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from spotrl.bc.honest_gate import (EPOCHS, _assert_mask_purity,
                                    build_exclusion_mask, build_unambiguous_flip,
                                    _load_epoch_frames)
from spotrl.bc.train_clone import apply_scaler, head0_logits, load_pooled


def mechanical_tp_mask(df) -> np.ndarray:
    """M — АПРИОРИ множество механических тейков: exit_reason=='tp' И
    m_exit_flag==0. Чистая функция v7-меток, НОЛЬ обращений к логитам клона."""
    reason = df["exit_reason"].to_numpy().astype(str)
    is_flip_exit = df["is_flip_exit"].to_numpy().astype(bool)
    m_exit_flag = df["m_exit_flag"].to_numpy() > 0.5
    return is_flip_exit & (reason == "tp") & (~m_exit_flag)


def run_guard(data_dir: str, model_path: str) -> Dict:
    """In-sample страж на saved reg_cd: overlap выхода против must-copy ref и
    recall reg_cd на must-copy, с личностью каждого промаха (в M или вне)."""
    from stable_baselines3 import PPO

    _assert_mask_purity()
    frames = _load_epoch_frames(data_dir)
    excl = {e: build_exclusion_mask(frames[e]) for e in EPOCHS}
    unamb = {e: build_unambiguous_flip(frames[e]) for e in EPOCHS}
    M = {e: mechanical_tp_mask(frames[e]) for e in EPOCHS}

    mf = json.loads(Path(model_path + ".manifest.json").read_text())
    mu = np.array(mf["scaler_mu"], np.float32)
    sd = np.array(mf["scaler_sd"], np.float32)
    X_raw, y, w, meta = load_pooled(data_dir)
    X = apply_scaler(X_raw, mu, sd)
    policy = PPO.load(model_path, device="cpu").policy
    d_all = head0_logits(policy, X)
    d = (d_all[:, 1] - d_all[:, 0]).astype(np.float64)

    out = {"per_epoch": {}}
    for e in EPOCHS:
        m = meta["per_epoch"][e]
        di = d[m["idx"]]
        is_stay = m["is_stay"]; is_exit = m["is_exit"]
        inpos = frames[e]["a_in_position"].to_numpy() > 0.5

        unambig = unamb[e]["exit_unambig"]           # legflip|tp|b2b
        Me = M[e]
        must_copy = unambig & (~Me)                  # legflip|b2b|signal-tp
        signal_tp = (frames[e]["exit_reason"].to_numpy().astype(str) == "tp") \
            & (frames[e]["is_flip_exit"].to_numpy().astype(bool)) \
            & (frames[e]["m_exit_flag"].to_numpy() > 0.5)

        # ДЖЕРРИМЕНДЕРИНГ-АССЕРТ / целостность M:
        assert (Me & signal_tp).sum() == 0, "M пересекается с signal-tp"
        assert (must_copy & signal_tp).sum() == signal_tp.sum(), \
            "signal-tp обязан остаться в must-copy"
        assert (must_copy & Me).sum() == 0, "механич. tp просочился в must-copy"

        # --- не-латч STAY на выход-стороне ---
        latch_stay = excl[e]["exclude"]
        nonlatch_stay = is_stay & (~latch_stay)
        inpos_nl = nonlatch_stay & inpos

        # СТАРЫЙ reference: min по ВСЕМ exit-барам.
        min_exit_all = float(di[is_exit].min())
        ov_old = int((inpos_nl & (d[m["idx"]] >= min_exit_all)).sum())
        # НОВЫЙ reference: min по must-copy (unambig\M).
        min_exit_mc = float(di[must_copy].min())
        ov_new = int((inpos_nl & (d[m["idx"]] >= min_exit_mc)).sum())

        # характеристика overlap-баров при НОВОМ ref (pos_tag + d) — для отчёта.
        ov_idx = np.where(inpos_nl & (d[m["idx"]] >= min_exit_mc))[0]
        tag_tr = frames[e]["a_pos_tag_transition"].to_numpy() > 0.5
        tag_dip = frames[e]["a_pos_tag_dip"].to_numpy() > 0.5
        exit_sig = frames[e]["m_exit_flag"].to_numpy() > 0.5
        ov_chars = []
        for j in ov_idx[np.argsort(-di[ov_idx])][:8]:
            tag = ("transition" if tag_tr[j] else "dip" if tag_dip[j] else "flat")
            ov_chars.append({"idx": int(j), "d": round(float(di[j]), 3),
                             "pos_tag": tag, "exit_sig": bool(exit_sig[j])})

        # in-sample recall reg_cd на must-copy и на всём unambig.
        rec_unambig = float((di[unambig] > 0).mean())
        rec_mustcopy = float((di[must_copy] > 0).mean())
        miss_unambig = unambig & (di <= 0)
        miss_mustcopy = must_copy & (di <= 0)
        # промахи целиком в M?
        miss_in_M = int((miss_unambig & Me).sum())
        miss_out_M = int((miss_unambig & (~Me)).sum())

        out["per_epoch"][e] = {
            "n_M": int(Me.sum()),
            "n_signal_tp": int(signal_tp.sum()),
            "n_unambig": int(unambig.sum()),
            "n_must_copy": int(must_copy.sum()),
            "min_exit_all_ref": min_exit_all,
            "min_exit_mustcopy_ref": min_exit_mc,
            "overlap_exit_old_ref": ov_old,
            "overlap_exit_new_ref": ov_new,
            "recall_insample_unambig": rec_unambig,
            "recall_insample_mustcopy": rec_mustcopy,
            "n_miss_unambig": int(miss_unambig.sum()),
            "n_miss_mustcopy": int(miss_mustcopy.sum()),
            "n_miss_in_M": miss_in_M,
            "n_miss_out_M": miss_out_M,
            "overlap_new_ref_chars": ov_chars,
        }
    return out


def run_outofepoch_guard(data_dir: str, seed: int, n_epochs: int) -> Dict:
    """Воспроизвести пункт-4 out-of-epoch клоны и проверить: промахи recall ⊆ M."""
    from spotrl.bc.train_clone import build_policy, fit_scaler, train

    frames = _load_epoch_frames(data_dir)
    unamb = {e: build_unambiguous_flip(frames[e]) for e in EPOCHS}
    M = {e: mechanical_tp_mask(frames[e]) for e in EPOCHS}

    X_raw, y, w, meta = load_pooled(data_dir)
    mu, sd = fit_scaler(X_raw)
    Xn = apply_scaler(X_raw, mu, sd)

    res = {}
    for tr_ep, te_ep in (("2021", "2024"), ("2024", "2021")):
        idx = meta["per_epoch"][tr_ep]["idx"]
        _, policy = build_policy(Xn.shape[1], seed)
        train(policy, Xn[idx], y[idx], w[idx], seed, n_epochs=n_epochs)
        m_te = meta["per_epoch"][te_ep]
        lg = head0_logits(policy, Xn[m_te["idx"]])
        d_te = (lg[:, 1] - lg[:, 0]).astype(np.float64)
        unambig = unamb[te_ep]["exit_unambig"]
        Me = M[te_ep]
        must_copy = unambig & (~Me)
        miss = unambig & (d_te <= 0)
        reason_te = frames[te_ep]["exit_reason"].to_numpy().astype(str)
        sig_tp = (reason_te == "tp") & (frames[te_ep]["m_exit_flag"].to_numpy() > 0.5) \
            & frames[te_ep]["is_flip_exit"].to_numpy().astype(bool)
        miss_out = miss & (~Me)
        miss_chars = [{"idx": int(j), "exit_reason": reason_te[j],
                       "d": round(float(d_te[j]), 3), "in_M": bool(Me[j]),
                       "is_signal_tp": bool(sig_tp[j])}
                      for j in np.where(miss)[0]]
        res[f"{tr_ep}->{te_ep}"] = {
            "miss_out_M_reasons": [reason_te[j] for j in np.where(miss_out)[0]],
            "miss_chars": miss_chars,
            "recall_unambig_out": float((d_te[unambig] > 0).mean()),
            "recall_mustcopy_out": float((d_te[must_copy] > 0).mean()),
            "n_unambig": int(unambig.sum()),
            "n_must_copy": int(must_copy.sum()),
            "n_miss_unambig": int(miss.sum()),
            "n_miss_in_M": int((miss & Me).sum()),
            "n_miss_out_M": int((miss & (~Me)).sum()),
        }
    return res


def main() -> None:
    """CLI: прогнать ШАГ-1 страж и напечатать числа overlap/recall + личности промахов."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/cubecloud/Data/rlbinancetrader")
    ap.add_argument("--model",
                    default="/home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy_reg_cd")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-epochs", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ins = run_guard(args.data, args.model)
    oof = run_outofepoch_guard(args.data, args.seed, args.n_epochs)

    print("=== ШАГ-1 СТРАЖ: механич. tp M = {exit_reason=tp И m_exit_flag=0} ===")
    for e in EPOCHS:
        p = ins["per_epoch"][e]
        print(f"[{e}] |M|={p['n_M']} signal_tp={p['n_signal_tp']} "
              f"unambig={p['n_unambig']} must-copy={p['n_must_copy']}")
        print(f"     overlap: старый_ref={p['overlap_exit_old_ref']} "
              f"НОВЫЙ_ref(must-copy)={p['overlap_exit_new_ref']} "
              f"| min_d: all={p['min_exit_all_ref']:.3f} "
              f"must-copy={p['min_exit_mustcopy_ref']:.3f}")
        print(f"     recall in-sample: unambig={p['recall_insample_unambig']:.4f} "
              f"must-copy={p['recall_insample_mustcopy']:.4f}")
        print(f"     промахи unambig={p['n_miss_unambig']} → в M={p['n_miss_in_M']} "
              f"ВНЕ M={p['n_miss_out_M']} | must-copy промахов={p['n_miss_mustcopy']}")
    print("\n=== ПУНКТ-4 out-of-epoch (переобученные кросс-эпоховые клоны) ===")
    for k, r in oof.items():
        print(f"[{k}] recall unambig={r['recall_unambig_out']:.4f}(n={r['n_unambig']}) "
              f"→ must-copy={r['recall_mustcopy_out']:.4f}(n={r['n_must_copy']}) "
              f"| промахи={r['n_miss_unambig']} в M={r['n_miss_in_M']} ВНЕ M={r['n_miss_out_M']}")
        if r["miss_out_M_reasons"]:
            print(f"       ВНЕ-M промахи (причины выхода): {r['miss_out_M_reasons']}")
        for c in r["miss_chars"]:
            print(f"       промах idx={c['idx']} reason={c['exit_reason']} "
                  f"d={c['d']} in_M={c['in_M']} signal_tp={c['is_signal_tp']}")
    print("\n=== overlap-бары при must-copy ref (топ по d, in-sample reg_cd) ===")
    for e in EPOCHS:
        for c in ins["per_epoch"][e]["overlap_new_ref_chars"]:
            print(f"[{e}] idx={c['idx']} d={c['d']} pos_tag={c['pos_tag']} exit_sig={c['exit_sig']}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"insample": ins, "outofepoch": oof}, ensure_ascii=False, indent=2))
        print(f"\nСохранено: {args.out}")


if __name__ == "__main__":
    main()
