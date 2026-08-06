"""K3-классификатор p_SL для инъекции в политику (amend5, путь 1).

Обучает ЛИНЕЙНЫЙ логрег p_SL(obs_std39) = sigmoid(w·obs+b) на dip-барах
train-2021 (2021-01..2024-03), SL-метка из label_trades. ЗАМОРАЖИВАЕТСЯ:
те же w,b применяются и при обучении RL (на 2021), и при OOS-оценке (tune-2024,
который классификатор НЕ видел) → нет утечки в наблюдении. Признаки строго
причинны (obs бара). Сохраняет {w(39), b, feature=obs_cols} в JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from spotrl.bc.k4_root_check import _load_arrays, MODEL_IN

DATA = "/home/cubecloud/Data/rlbinancetrader"
OUT = f"{DATA}/k3_psl_classifier.json"


def run():
    """Обучить линейный p_SL на ВСЕХ dip-барах 2021, сохранить веса."""
    X, in_pos, grp, G, R, is_tr, df, tr = _load_arrays()
    dip = grp > 0
    y = (grp == 1).astype(int)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(X[dip], y[dip])
    w = clf.coef_[0].astype(np.float64)
    b = float(clf.intercept_[0])
    auc_all = float(roc_auc_score(y[dip], clf.decision_function(X[dip])))
    # held-out справочно (последние 30% сделок 2021): подтверждение обобщения
    val = dip & (~is_tr)
    auc_val = float(roc_auc_score(y[val], clf.decision_function(X[val]))) \
        if y[val].min() != y[val].max() else float("nan")
    mf = json.loads(Path(MODEL_IN + ".manifest.json").read_text())
    out = {"w": w.tolist(), "b": b, "obs_cols": mf["obs_cols"],
           "auc_train_all_dip": auc_all, "auc_val_heldout_dip": auc_val,
           "trained_on": "2021-01..2024-03 dip bars (frozen; applied to OOS too)"}
    Path(OUT).write_text(json.dumps(out, ensure_ascii=False))
    print(f"K3 p_SL логрег обучен: AUC(all-dip)={auc_all:.3f} "
          f"AUC(held-out dip)={auc_val:.3f}")
    print(f"|w| max={np.abs(w).max():.3f} b={b:.3f}  → {OUT}")
    return out


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run()
