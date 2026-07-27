"""Кросс-эпоховый АНТИ-МЕМОИЗАЦИОННЫЙ страж клона (разводит «выучил»/«запомнил»).

Зачем (критично). Операционный копировальный гейт (eval_gate) меряется на клоне,
обученном на ОБЕИХ эпохах — истинного holdout нет, поэтому in-sample прохождение
само по себе не отличает выученное правило от мемоизации по шумовым осям (ценовые/
объёмные признаки в 38-мерном obs). Этот модуль — ОТДЕЛЬНЫЙ страж: обучить на
ОДНОЙ эпохе, замерить на ДРУГОЙ.

Метрика — AUC (разделимость), НЕ L∞-зазор. Прежняя версия этого модуля мерила
`min_FLIP(d)−max_STAY(d) >= 20.31` — но диагностика (`diag_obs_sufficiency`) прямо
отвергла L∞-зазор как меру обобщения: под капнутыми логитами он ограничен ~23 nat
и говорит о калибровке, а не о переносе семантики. Правило v7 идентично в обеих
эпохах → клон, выучивший ПРАВИЛО, обязан дать out-of-epoch AUC не хуже, чем
маломощная регуляризованная диагностика (вход ~0.98, выход ~0.94); клон,
запомнивший бары, — провалится (AUC → 0.5). Раздельно ВХОД (flat-бары, ENTER vs
WAIT) и ВЫХОД (in-position, EXIT vs HOLD), обе стороны (2021↔2024).

Это go/no-go для вердикта «готов к RL» как АНТИ-МЕМОИЗАЦИОННЫЙ страж (не
операционный копировальный порог — их разводим явно, см. отчёт).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from spotrl.bc.train_clone import (apply_scaler, build_policy, fit_scaler,
                                    head0_logits, load_pooled, train)

# Пороги-полы AUC из диагностики obs (diag_obs_sufficiency_2026-07-26.md):
# LR L2 out-of-epoch — вход 0.979/0.985, выход 0.941/0.936. Клон обязан быть не
# хуже (он мощнее LR). Пол консервативный (минус запас на шум малой выборки).
# Пороги подняты (честный гейт v2, pre-reg 2026-07-27, пункт 5): вход 0.98,
# выход 0.94 out-of-epoch. Прежние 0.95/0.90 — SUPERSEDED (были заниженным полом
# по маломощной LR-диагностике; MLP-клон дал 0.991-1.0, запас есть).
AUC_FLOOR_ENTRY = 0.98
AUC_FLOOR_EXIT = 0.94


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC (доля правильно упорядоченных пар pos/neg). NaN если один класс."""
    from sklearn.metrics import roc_auc_score
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, score))


def _recall_fpr_at_train_thr(s_tr, y_tr, s_te, y_te, target=0.995):
    """Порог по train (recall>=target на train-позитивах), затем recall/FPR на test."""
    pos = np.sort(s_tr[y_tr == 1])
    if len(pos) == 0:
        return float("nan"), float("nan")
    k = int(np.ceil((1 - target) * len(pos)))
    thr = pos[min(k, len(pos) - 1)]
    pred = s_te >= thr
    rec = float(pred[y_te == 1].mean()) if (y_te == 1).any() else float("nan")
    fpr = float(pred[y_te == 0].mean()) if (y_te == 0).any() else float("nan")
    return rec, fpr


@dataclass
class CrossEpochResult:
    """AUC клона, обученного на train_epoch, замеренный in- и out-of-epoch."""

    train_epoch: str
    test_epoch: str
    auc_entry_in: float
    auc_entry_out: float
    auc_exit_in: float
    auc_exit_out: float
    recall_entry_out: float
    fpr_entry_out: float
    recall_exit_out: float
    fpr_exit_out: float
    generalizes: bool

    def describe(self) -> dict:
        """Сериализуемая сводка."""
        return {k: (float(v) if isinstance(v, np.floating) else v)
                for k, v in self.__dict__.items()}


def _side_scores(policy, Xn, meta, epoch):
    """d, и маски (flat/in_pos, is_entry/is_exit) для одной эпохи."""
    m = meta["per_epoch"][epoch]
    lg = head0_logits(policy, Xn[m["idx"]])
    d = lg[:, 1] - lg[:, 0]
    return d, m


def _entry_exit_auc(policy, Xn, meta, tr_ep, te_ep):
    """AUC вход (flat) и выход (in-pos) in- и out-of-epoch + recall/FPR out."""
    d_tr, m_tr = _side_scores(policy, Xn, meta, tr_ep)
    d_te, m_te = _side_scores(policy, Xn, meta, te_ep)

    def side(d, m, in_pos_side, pos_mask):
        """score/y на срезе стороны (вход: flat; выход: in-position)."""
        sel = m["in_pos"] if in_pos_side else ~m["in_pos"]
        return d[sel], pos_mask(m)[sel].astype(int)

    ent = lambda m: m["is_entry"]
    ext = lambda m: m["is_exit"]
    s_ent_tr, y_ent_tr = side(d_tr, m_tr, False, ent)
    s_ent_te, y_ent_te = side(d_te, m_te, False, ent)
    s_ext_tr, y_ext_tr = side(d_tr, m_tr, True, ext)
    s_ext_te, y_ext_te = side(d_te, m_te, True, ext)

    r_ent, f_ent = _recall_fpr_at_train_thr(s_ent_tr, y_ent_tr, s_ent_te, y_ent_te)
    r_ext, f_ext = _recall_fpr_at_train_thr(s_ext_tr, y_ext_tr, s_ext_te, y_ext_te)
    return {
        "auc_entry_in": _auc(s_ent_tr, y_ent_tr),
        "auc_entry_out": _auc(s_ent_te, y_ent_te),
        "auc_exit_in": _auc(s_ext_tr, y_ext_tr),
        "auc_exit_out": _auc(s_ext_te, y_ext_te),
        "recall_entry_out": r_ent, "fpr_entry_out": f_ent,
        "recall_exit_out": r_ext, "fpr_exit_out": f_ext,
    }


def collision_d_dist(policy, Xn, meta, epoch):
    """d-распределение на СВОРОТНОЙ популяции exit_flag=1 & transition & bull.

    Диагностика (diag_obs): вся коллизия «сигнал есть, v7 держит» 100% объяснена
    a_pos_tag_transition & m_regime_code>=bull (латч allow_signal_exit). MLP МОЖЕТ
    выразить эту конъюнкцию (LR — нет, потому дал зазор 0.5/−2.6). Печатаем d для
    held vs exited на этих барах: если held<exited по d — правило выучено, а не шум.
    """
    import pandas as pd
    from pathlib import Path
    # берём сырые колонки эпохи для селекторов (не в obs-мете).
    df = pd.read_parquet(Path(collision_d_dist.data_dir) /
                         f"bc_clone_v7_{epoch}.parquet")
    m = meta["per_epoch"][epoch]
    d = head0_logits(policy, Xn[m["idx"]])
    d = d[:, 1] - d[:, 0]
    sel = ((df["a_in_position"].to_numpy() > 0.5)
           & (df["m_exit_flag"].to_numpy() > 0.5)
           & (df["a_pos_tag_transition"].to_numpy() > 0.5)
           & (df["m_regime_code"].to_numpy() > 1.5))
    exited = sel & df["is_flip_exit"].to_numpy().astype(bool)
    held = sel & ~df["is_flip_exit"].to_numpy().astype(bool)
    def q(mask):
        """min/med/max d по маске (или nan)."""
        v = d[mask]
        return (float(np.min(v)), float(np.median(v)), float(np.max(v))) if len(v) else (np.nan,)*3
    return {"n_held": int(held.sum()), "n_exited": int(exited.sum()),
            "d_held": q(held), "d_exited": q(exited)}


collision_d_dist.data_dir = "/home/cubecloud/Data/rlbinancetrader"


def run_cross_epoch(data_dir: str, seed: int = 0,
                    n_epochs: int = 40) -> List[CrossEpochResult]:
    """Обучить на КАЖДОЙ эпохе отдельно, замерить AUC на другой. Обе стороны."""
    collision_d_dist.data_dir = data_dir
    X_raw, y, w, meta = load_pooled(data_dir)
    mu, sd = fit_scaler(X_raw)
    Xn = apply_scaler(X_raw, mu, sd)
    out: List[CrossEpochResult] = []
    for tr_ep, te_ep in (("2021", "2024"), ("2024", "2021")):
        idx = meta["per_epoch"][tr_ep]["idx"]
        _, policy = build_policy(Xn.shape[1], seed)
        train(policy, Xn[idx], y[idx], w[idx], seed, n_epochs=n_epochs)
        a = _entry_exit_auc(policy, Xn, meta, tr_ep, te_ep)
        gen = (a["auc_entry_out"] >= AUC_FLOOR_ENTRY
               and a["auc_exit_out"] >= AUC_FLOOR_EXIT)
        out.append(CrossEpochResult(
            train_epoch=tr_ep, test_epoch=te_ep,
            auc_entry_in=a["auc_entry_in"], auc_entry_out=a["auc_entry_out"],
            auc_exit_in=a["auc_exit_in"], auc_exit_out=a["auc_exit_out"],
            recall_entry_out=a["recall_entry_out"], fpr_entry_out=a["fpr_entry_out"],
            recall_exit_out=a["recall_exit_out"], fpr_exit_out=a["fpr_exit_out"],
            generalizes=bool(gen)))
    return out


def main() -> None:
    """CLI: прогнать кросс-эпоховый AUC-страж, вывести вердикт."""
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/cubecloud/Data/rlbinancetrader")
    ap.add_argument("--out", default=None)
    ap.add_argument("--collision", action="store_true",
                    help="печать d-распределения своротной популяции")
    args = ap.parse_args()
    res = run_cross_epoch(args.data)
    for r in res:
        print(f"train={r.train_epoch}->{r.test_epoch}: "
              f"ВХОД AUC in={r.auc_entry_in:.4f} out={r.auc_entry_out:.4f} "
              f"(пол {AUC_FLOOR_ENTRY}) recall_out={r.recall_entry_out:.3f} "
              f"fpr_out={r.fpr_entry_out:.2e}")
        print(f"                    ВЫХОД AUC in={r.auc_exit_in:.4f} "
              f"out={r.auc_exit_out:.4f} (пол {AUC_FLOOR_EXIT}) "
              f"recall_out={r.recall_exit_out:.3f} fpr_out={r.fpr_exit_out:.2e} "
              f"| обобщает={r.generalizes}")
    ok = all(r.generalizes for r in res)
    print(f"АНТИ-МЕМОИЗАЦИОННЫЙ СТРАЖ (AUC out >= полов, обе стороны): {ok}")
    if args.collision:
        X_raw, y, w, meta = load_pooled(args.data)
        mu, sd = fit_scaler(X_raw)
        Xn = apply_scaler(X_raw, mu, sd)
        idx = meta["per_epoch"]["2021"]["idx"]
        _, policy = build_policy(Xn.shape[1], 0)
        train(policy, Xn[idx], y[idx], w[idx], 0)
        for ep in ("2021", "2024"):
            c = collision_d_dist(policy, Xn, meta, ep)
            tag = "IN" if ep == "2021" else "OUT"
            print(f"  [{ep} {tag}] сворот: held={c['n_held']} d(min/med/max)={c['d_held']}"
                  f" | exited={c['n_exited']} d={c['d_exited']}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            [r.describe() for r in res], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
