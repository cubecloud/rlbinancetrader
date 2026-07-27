"""ДИАГНОСТИКА (не гейт): раздельная калибровка сдвига логита для ВХОДА и ВЫХОДА.

Вопрос: провал пред-зарег. цели (FPR<=3e-7 И recall>=99.5%) — артефакт ОДНОГО
глобального скалярного сдвига на пул вход+выход, или он недостижим и при СВОЁМ
пороге для каждой стороны?

Раздельность легитимна: сторона выбирается НАБЛЮДАЕМЫМ в obs признаком
a_in_position (in_pos). Порог-по-стороне = разворачиваемая на serve политика
(if in_pos: use s_exit else s_enter), а не подглядывание в метки.
  * ВХОД : FLIP=is_entry, знаменатель FPR = STAY & in_pos=False (flat WAIT).
  * ВЫХОД: FLIP=is_exit,  знаменатель FPR = STAY & in_pos=True  (HOLD).

Переиспользует харнесс дословно: head0_logits/apply_scaler/load_pooled и
ПРАВИЛО калибровки calibrate_shift (FPR-anchored s_hi), плюс цикл сэмплирования
как в eval_gate.teacher_forced. s* НЕ тюнится.

Run:
  python -m spotrl.bc.diag_perside --data /home/cubecloud/Data/rlbinancetrader \
      --model /home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy_reg --seeds 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spotrl.bc.train_clone import (
    FPR_TARGET, RECALL_FLOOR, _logit, apply_scaler, head0_logits, load_pooled,
)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, np.float64), -60, 60)))


def _calibrate_side(d_by_epoch):
    """FPR-anchored s* по стороне — ТО ЖЕ определение, что calibrate_shift, но
    через бисекцию монотонных функций (mean(sig(.+s)) растёт с s), а не сетку.

    s_hi = max s: per-epoch max FPR <= FPR_TARGET; s_lo = min s: per-epoch min
    recall >= RECALL_FLOOR; s* = s_hi; feasible = s_lo <= s_hi. Идентично
    train_clone.calibrate_shift (та же монотонность), только root-finding быстрее.
    """
    stays = [v["stay"] for v in d_by_epoch.values()]
    flips = [v["flip_all"] for v in d_by_epoch.values()]

    def fpr(s):
        """Per-epoch max ожидаемого FPR при сдвиге s."""
        return max(float(_sig(st + s).mean()) for st in stays)

    def recall(s):
        """Per-epoch min ожидаемого recall при сдвиге s."""
        return min(float(_sig(fl + s).mean()) for fl in flips)

    lo, hi = -200.0, 200.0
    # s_hi: max s с fpr(s) <= FPR_TARGET (fpr растёт с s)
    a, b = lo, hi
    for _ in range(200):
        m = 0.5 * (a + b)
        if fpr(m) <= FPR_TARGET:
            a = m
        else:
            b = m
    s_hi = a
    # s_lo: min s с recall(s) >= RECALL_FLOOR (recall растёт с s)
    a, b = lo, hi
    for _ in range(200):
        m = 0.5 * (a + b)
        if recall(m) >= RECALL_FLOOR:
            b = m
        else:
            a = m
    s_lo = b
    feasible = s_lo <= s_hi
    gap = min(fl.min() - st.max() for st, fl in zip(stays, flips))
    return {"s_lo": s_lo, "s_hi": s_hi, "s_star": s_hi, "feasible": feasible,
            "gap_nat": float(gap),
            "gap_needed": _logit(RECALL_FLOOR) - _logit(FPR_TARGET)}


def _sampled(stay_p, flip_p, s, seeds):
    """Сэмплированные (>=seeds сидов) FPR (по stay) и recall (по flip) при сдвиге s."""
    ps, pf = _sig(stay_p + s), _sig(flip_p + s)
    rng = np.random.default_rng(0)
    fprs, recs = [], []
    for _ in range(seeds):
        fprs.append(float((rng.random(len(ps)) < ps).mean()))
        recs.append(float((rng.random(len(pf)) < pf).mean()))
    return (float(np.mean(fprs)), float(np.max(fprs)),
            float(np.mean(recs)), float(np.min(recs)))


def run(data_dir, model_path, seeds):
    """Собрать per-side per-epoch картину достижимости пред-зарег. цели."""
    from stable_baselines3 import PPO
    mf = json.loads(Path(model_path + ".manifest.json").read_text())
    mu = np.array(mf["scaler_mu"], np.float32)
    sd = np.array(mf["scaler_sd"], np.float32)
    s_global = float(mf["shift"]["s_star"])

    X_raw, y, w, meta = load_pooled(data_dir)
    X = apply_scaler(X_raw, mu, sd)
    policy = PPO.load(model_path, device="cpu").policy
    d_all = head0_logits(policy, X)
    d = (d_all[:, 1] - d_all[:, 0]).astype(np.float64)

    report = {"s_global": s_global, "seeds": seeds, "sides": {}, "sanity": {}}

    # --- Санити партиции (load-bearing) ---
    for epoch, m in meta["per_epoch"].items():
        entry, exit_, stay, in_pos = (m["is_entry"], m["is_exit"],
                                      m["is_stay"], m["in_pos"])
        n_stay = int(stay.sum())
        flat_stay = stay & (~in_pos)
        inpos_stay = stay & in_pos
        san = {
            "entry_all_flat": bool((entry & in_pos).sum() == 0),
            "exit_all_inpos": bool((exit_ & (~in_pos)).sum() == 0),
            "stay_partition_ok": bool(int(flat_stay.sum()) + int(inpos_stay.sum()) == n_stay),
            "n_stay": n_stay,
            "n_flat_stay": int(flat_stay.sum()),
            "n_inpos_stay": int(inpos_stay.sum()),
            "n_entry": int(entry.sum()),
            "n_exit": int(exit_.sum()),
        }
        report["sanity"][epoch] = san

    # --- Дешёвый дискриминатор: argmax-ложные-FLIP при s_global по in_pos ---
    report["argmax_false_flip_by_side"] = {}
    for epoch, m in meta["per_epoch"].items():
        di = d[m["idx"]]
        stay = m["is_stay"]; in_pos = m["in_pos"]
        false = stay & (di + s_global > 0)
        report["argmax_false_flip_by_side"][epoch] = {
            "total": int(false.sum()),
            "in_pos": int((false & in_pos).sum()),
            "flat": int((false & (~in_pos)).sum()),
        }

    # --- Раздельная калибровка: своё s* на сторону (обе эпохи одновременно) ---
    for side, flip_key, stay_in_pos in (("entry", "is_entry", False),
                                        ("exit", "is_exit", True)):
        d_by_epoch = {}
        for epoch, m in meta["per_epoch"].items():
            di = d[m["idx"]]
            stay_mask = m["is_stay"] & (m["in_pos"] == stay_in_pos)
            d_by_epoch[epoch] = {"stay": di[stay_mask], "flip_all": di[m[flip_key]]}
        cal = _calibrate_side(d_by_epoch)  # то же определение, бисекция
        s_star = cal["s_star"]
        per_epoch = {}
        for epoch in d_by_epoch:
            stay_d = d_by_epoch[epoch]["stay"]; flip_d = d_by_epoch[epoch]["flip_all"]
            fm, fmax, rm, rmin = _sampled(stay_d, flip_d, s_star, seeds)
            # expectation (несмещённые пределы при беск. сидах)
            fpr_exp = float(_sig(stay_d + s_star).mean())
            rec_exp = float(_sig(flip_d + s_star).mean())
            per_epoch[epoch] = {
                "gap_nat": float(flip_d.min() - stay_d.max()),
                "fpr_mean": fm, "fpr_max": fmax,
                "fpr_expect": fpr_exp, "recall_expect": rec_exp,
                "recall_mean": rm, "recall_min": rmin,
                "n_stay": int(len(stay_d)), "n_flip": int(len(flip_d)),
            }
        # expectation-feasible: s_lo<=s_hi (по fpr_mean=expectation) И recall>=floor
        recall_ok = all(pe["recall_expect"] >= RECALL_FLOOR for pe in per_epoch.values())
        report["sides"][side] = {
            "s_lo": cal["s_lo"], "s_hi": cal["s_hi"], "s_star": s_star,
            "feasible_expectation": bool(cal["feasible"]),
            "recall_ok_expectation": bool(recall_ok),
            "achievable": bool(cal["feasible"] and recall_ok),
            "gap_needed": cal["gap_needed"],
            "per_epoch": per_epoch,
        }

    # --- Шаг 4: непроходящие бары ОБЕИХ сторон (перекрытие ранга d) ---
    overlap = {"entry": {}, "exit": {}}
    for epoch, m in meta["per_epoch"].items():
        di = d[m["idx"]]
        # ВХОД: flat-STAY с d >= min(d entry-flip)
        entry_d = di[m["is_entry"]]
        flat_stay_mask = m["is_stay"] & (~m["in_pos"])
        min_entry = float(entry_d.min())
        ov_e = flat_stay_mask & (di >= min_entry)
        overlap["entry"][epoch] = {
            "min_flip_d": min_entry, "max_stay_d": float(di[flat_stay_mask].max()),
            "n_stay_overlap": int(ov_e.sum()),
        }
        # ВЫХОД: in-pos-STAY с d >= min(d exit-flip)
        exit_d = di[m["is_exit"]]
        inpos_stay_mask = m["is_stay"] & m["in_pos"]
        min_exit = float(exit_d.min())
        ov_x = inpos_stay_mask & (di >= min_exit)
        overlap["exit"][epoch] = {
            "min_flip_d": min_exit, "max_stay_d": float(di[inpos_stay_mask].max()),
            "n_stay_overlap": int(ov_x.sum()),
        }
    report["overlap"] = overlap

    return report, meta, d


def _describe_nature(data_dir, meta, d):
    """Природа перекрывающихся выходных STAY-баров: латч transition&bull?"""
    import pandas as pd
    out = {}
    for epoch in ("2021", "2024"):
        df = pd.read_parquet(Path(data_dir) / f"bc_clone_v7_{epoch}.parquet")
        m = meta["per_epoch"][epoch]
        di = d[m["idx"]]
        cols = df.columns
        trans_col = "a_pos_tag_transition"
        tr = df[trans_col].to_numpy() > 0.5 if trans_col in cols else None
        info = {"regime_cols": [c for c in cols if c.startswith("m_regime")]}
        # выход
        exit_d = di[m["is_exit"]]
        inpos_stay = m["is_stay"] & m["in_pos"]
        ov_x = inpos_stay & (di >= float(exit_d.min()))
        info["exit_n_overlap"] = int(ov_x.sum())
        if tr is not None:
            info["exit_n_overlap_transition"] = int((ov_x & tr).sum())
        # вход
        entry_d = di[m["is_entry"]]
        flat_stay = m["is_stay"] & (~m["in_pos"])
        ov_e = flat_stay & (di >= float(entry_d.min()))
        info["entry_n_overlap"] = int(ov_e.sum())
        if tr is not None:
            info["entry_n_overlap_transition"] = int((ov_e & tr).sum())
        out[epoch] = info
    return out


def main():
    """CLI: прогнать per-side диагностику и напечатать сводку + опц. JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/cubecloud/Data/rlbinancetrader")
    ap.add_argument("--model",
                    default="/home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy_reg")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report, meta, d = run(args.data, args.model, args.seeds)
    report["nature"] = _describe_nature(args.data, meta, d)

    print(f"s_global (пул) = {report['s_global']:.3f}\n")
    print("=== САНИТИ ПАРТИЦИИ ===")
    for e, s in report["sanity"].items():
        print(f"[{e}] entry_all_flat={s['entry_all_flat']} exit_all_inpos={s['exit_all_inpos']} "
              f"stay_part_ok={s['stay_partition_ok']}")
        print(f"     n_stay={s['n_stay']:,} flat={s['n_flat_stay']:,} inpos={s['n_inpos_stay']:,} "
              f"entry={s['n_entry']} exit={s['n_exit']}")
    print("\n=== ARGMAX-ЛОЖНЫЕ-FLIP при s_global, по стороне ===")
    for e, a in report["argmax_false_flip_by_side"].items():
        print(f"[{e}] всего={a['total']} in_pos(выход)={a['in_pos']} flat(вход)={a['flat']}")
    print("\n=== РАЗДЕЛЬНАЯ КАЛИБРОВКА (своё s* на сторону) ===")
    for side, sd in report["sides"].items():
        print(f"\n[{side.upper()}] s_lo={sd['s_lo']:.3f} s_hi(s*)={sd['s_hi']:.3f} "
              f"feasible_exp={sd['feasible_expectation']} recall_ok_exp={sd['recall_ok_expectation']} "
              f"=> ДОСТИЖИМО={sd['achievable']}")
        for e, pe in sd["per_epoch"].items():
            print(f"   [{e}] gap={pe['gap_nat']:.2f} nat | "
                  f"FPR mean={pe['fpr_mean']:.2e} max={pe['fpr_max']:.2e} exp={pe['fpr_expect']:.2e} | "
                  f"recall mean={pe['recall_mean']:.4f} min={pe['recall_min']:.4f} exp={pe['recall_expect']:.4f}")
    print("\n=== ПЕРЕКРЫТИЕ РАНГА d (STAY с d >= min(d flip) на своей стороне) ===")
    for side in ("entry", "exit"):
        for e, o in report["overlap"][side].items():
            n = report["nature"][e]
            tr = n.get(f"{side}_n_overlap_transition", "?")
            print(f"[{side} {e}] min(d flip)={o['min_flip_d']:.2f} max(d stay)={o['max_stay_d']:.2f} "
                  f"| перекрывающих STAY={o['n_stay_overlap']} из них transition={tr}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\nСохранено: {args.out}")


if __name__ == "__main__":
    main()
