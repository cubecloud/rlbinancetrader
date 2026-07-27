"""Диагностика непроходящих баров честного гейта (2026-07-27).

Отвечает по КАЖДОЙ популяции (вход 26/2, выход 34/22): (a) training/калибровка
(признаки НАБЛЮДАЕМЫ, малая модель отделяет и обобщает кросс-эпохово) или
(b) obs-дыра (нужен ненаблюдаемый вход). НЕ переобучает клон.

Тесты (обе популяции раздельно):
  1. Малая регуляризованная модель (LR L2 / дерево гл.4) на НАБЛЮДАЕМЫХ obs —
     отделяет ли hold-бары от истинных FLIP? (in-sample + кросс-эпохово).
  2. Коллизия: ближайший истинный FLIP той же стороны в стандартизованном obs
     (L2/Linf). Дистанция ≈0 при разной метке = obs-дыра.
  3. Дамп 4 колонок (m_entry_signal, m_trans_entry_signal, a_cooldown_remain,
     a_cb_active) на РЕАЛЬНЫХ 26/2 барах, определённых d клона — разрешение §1.1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

from spotrl.bc.train_clone import apply_scaler, head0_logits, load_pooled, obs_columns
from spotrl.bc.honest_gate import build_exclusion_mask

DATA = "/home/cubecloud/Data/rlbinancetrader"
MODEL = "/home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy_reg"
EPOCHS = ("2021", "2024")


def load_frames():
    return {e: pd.read_parquet(Path(DATA) / f"bc_clone_v7_{e}.parquet") for e in EPOCHS}


def compute_d(frames, cols, mu, sd):
    from stable_baselines3 import PPO
    policy = PPO.load(MODEL, device="cpu").policy
    d = {}
    for e in EPOCHS:
        X = apply_scaler(frames[e][cols].to_numpy(np.float32), mu, sd)
        lg = head0_logits(policy, X)
        d[e] = (lg[:, 1] - lg[:, 0]).astype(np.float64)
    return d


def small_models(Xtr, ytr, Xte, yte):
    """LR-L2 (C=0.1) и дерево гл.4. Возвращает in/out AUC и train-accuracy."""
    out = {}
    for name, clf in (("lr_l2", LogisticRegression(C=0.1, max_iter=2000,
                                                    class_weight="balanced")),
                      ("tree4", DecisionTreeClassifier(max_depth=4,
                                                       class_weight="balanced",
                                                       random_state=0))):
        clf.fit(Xtr, ytr)
        p_in = clf.predict_proba(Xtr)[:, 1]
        auc_in = roc_auc_score(ytr, p_in) if len(np.unique(ytr)) > 1 else float("nan")
        if Xte is not None and len(np.unique(yte)) > 1:
            p_out = clf.predict_proba(Xte)[:, 1]
            auc_out = roc_auc_score(yte, p_out)
        else:
            auc_out = float("nan")
        out[name] = {"auc_in": float(auc_in), "auc_out": float(auc_out)}
    return out


def nn_collision(Xneg, Xpos):
    """Для каждого neg-бара мин. дистанция до pos (L2, Linf) в scaler-пространстве."""
    if len(Xneg) == 0 or len(Xpos) == 0:
        return {"min_l2": float("nan"), "min_linf": float("nan"), "n_near0": 0}
    # попарно; популяции малы (десятки против сотен) — прямой расчёт ок
    l2s, linfs = [], []
    for x in Xneg:
        diff = Xpos - x
        l2s.append(float(np.sqrt((diff ** 2).sum(1)).min()))
        linfs.append(float(np.abs(diff).max(1).min()))
    l2s, linfs = np.array(l2s), np.array(linfs)
    return {"min_l2": float(l2s.min()), "median_l2": float(np.median(l2s)),
            "min_linf": float(linfs.min()), "median_linf": float(np.median(linfs)),
            "n_near0_linf<0.1": int((linfs < 0.1).sum())}


def main():
    frames = load_frames()
    cols = obs_columns(frames["2021"])
    X_raw, y, w, meta = load_pooled(DATA)
    from spotrl.bc.train_clone import fit_scaler
    mf = json.loads(Path(MODEL + ".manifest.json").read_text())
    mu = np.array(mf["scaler_mu"], np.float32)
    sd = np.array(mf["scaler_sd"], np.float32)
    d = compute_d(frames, cols, mu, sd)

    report = {"epochs": {}}
    # scaled obs per epoch, cached
    Xs = {e: apply_scaler(frames[e][cols].to_numpy(np.float32), mu, sd) for e in EPOCHS}

    # ---- построить популяции по СЕМАНТИКЕ (без d) для тестов 1-3 ----
    pops = {}
    for e in EPOCHS:
        df = frames[e]
        excl = build_exclusion_mask(df)["exclude"]
        inpos = df["a_in_position"].to_numpy() > 0.5
        stay = df["bc_action"].to_numpy() == 0
        exit_flag = df["m_exit_flag"].to_numpy() > 0.5
        is_entry = df["is_flip_entry"].to_numpy().astype(bool)
        is_exit = df["is_flip_exit"].to_numpy().astype(bool)
        m_ent = df["m_entry_signal"].to_numpy() > 0.5
        m_tr = df["m_trans_entry_signal"].to_numpy() > 0.5
        cd = df["a_cooldown_remain"].to_numpy()
        cb = df["a_cb_active"].to_numpy() > 0.5

        # ВЫХОД hold: in-pos STAY, не-латч, без сигнала выхода
        exit_hold = inpos & stay & (~excl) & (~exit_flag)
        # ВХОД hold (семантика): flat STAY, есть грубый сигнал, не заблокирован obs
        entry_hold_sem = (~inpos) & stay & (m_ent | m_tr) & (cd <= 0.0) & (~cb)

        pops[e] = dict(is_entry=is_entry, is_exit=is_exit,
                       exit_hold=exit_hold, entry_hold_sem=entry_hold_sem,
                       inpos=inpos, stay=stay, exit_flag=exit_flag,
                       m_ent=m_ent, m_tr=m_tr, cd=cd, cb=cb, excl=excl)

    # ============ РАЗРЕШЕНИЕ §1.1: реальные d-бары гейта ============
    # exit-side: inpos non-latch STAY, d >= min(true exit d)
    # entry-side: flat non-latch STAY, d >= min(true entry d)
    dump = {}
    for e in EPOCHS:
        P = pops[e]
        di = d[e]
        is_stay = P["stay"]
        latch = P["excl"]
        nonlatch_stay = is_stay & (~latch)
        min_entry_d = di[P["is_entry"]].min()
        min_exit_d = di[P["is_exit"]].min()
        flat_nl = nonlatch_stay & (~P["inpos"])
        inpos_nl = nonlatch_stay & P["inpos"]
        gate_entry = flat_nl & (di >= min_entry_d)
        gate_exit = inpos_nl & (di >= min_exit_d)

        # --- дамп 4 колонок реальных gate_entry баров ---
        idx_e = np.where(gate_entry)[0]
        edump = {
            "n": int(gate_entry.sum()),
            "m_entry_signal=1": int(P["m_ent"][gate_entry].sum()),
            "m_trans=1": int(P["m_tr"][gate_entry].sum()),
            "any_sig=1": int((P["m_ent"] | P["m_tr"])[gate_entry].sum()),
            "cooldown_remain>0": int((P["cd"][gate_entry] > 0).sum()),
            "cb_active=1": int(P["cb"][gate_entry].sum()),
            "unblocked(cd<=0&cb=0)": int(((P["cd"] <= 0) & (~P["cb"]))[gate_entry].sum()),
            "sig&unblocked": int(((P["m_ent"] | P["m_tr"]) & (P["cd"] <= 0) & (~P["cb"]))[gate_entry].sum()),
            "cd_values_sample": [float(x) for x in P["cd"][gate_entry][:15]],
        }
        # weakest true exit reason
        min_exit_row = np.where(P["is_exit"])[0][np.argmin(di[P["is_exit"]])]
        xdump = {
            "n": int(gate_exit.sum()),
            "exit_flag=1_on_holds": int(P["exit_flag"][gate_exit].sum()),
            "weakest_true_exit_reason": str(frames[e]["exit_reason"].to_numpy()[min_exit_row]),
            "weakest_true_exit_m_exit_flag": float(frames[e]["m_exit_flag"].to_numpy()[min_exit_row]),
            "min_exit_d": float(min_exit_d),
            "min_entry_d": float(min_entry_d),
        }
        dump[e] = {"entry_gate": edump, "exit_gate": xdump}

    # ============ ТЕСТЫ 1 (малая модель) + 2 (коллизия) ============
    # для каждой стороны: pos = true flip, neg = hold pop. Train одна эпоха,
    # test другая (кросс-эпохово).
    popcounts = {e: {k: int(pops[e][k].sum()) for k in
                     ("is_entry", "is_exit", "exit_hold", "entry_hold_sem")}
                 for e in EPOCHS}
    results = {}
    for side, poskey, negkey in (("entry", "is_entry", "entry_hold_sem"),
                                 ("exit", "is_exit", "exit_hold")):
        results[side] = {}
        for tr, te in (("2021", "2024"), ("2024", "2021")):
            Ptr, Pte = pops[tr], pops[te]
            pos_tr, neg_tr = Ptr[poskey], Ptr[negkey]
            pos_te, neg_te = Pte[poskey], Pte[negkey]
            if neg_tr.sum() == 0 or neg_te.sum() == 0:
                results[side][f"{tr}->{te}"] = {
                    "n_pos_tr": int(pos_tr.sum()), "n_neg_tr": int(neg_tr.sum()),
                    "n_neg_te": int(neg_te.sum()),
                    "note": "neg-популяция ПУСТА — семантически неотделимых нет"}
                continue
            Xtr = np.vstack([Xs[tr][pos_tr], Xs[tr][neg_tr]])
            ytr = np.r_[np.ones(pos_tr.sum()), np.zeros(neg_tr.sum())]
            Xte = np.vstack([Xs[te][pos_te], Xs[te][neg_te]])
            yte = np.r_[np.ones(pos_te.sum()), np.zeros(neg_te.sum())]
            sm = small_models(Xtr, ytr, Xte, yte)
            # коллизия: neg-бары train-эпохи против pos train-эпохи
            coll = nn_collision(Xs[tr][neg_tr], Xs[tr][pos_tr])
            results[side][f"{tr}->{te}"] = {
                "n_pos_tr": int(pos_tr.sum()), "n_neg_tr": int(neg_tr.sum()),
                "n_neg_te": int(neg_te.sum()),
                "small_models": sm, "collision_neg_vs_pos": coll,
            }

    report = {"popcounts": popcounts, "resolve_1_1_gate_bars": dump, "tests": results}
    outp = Path("/home/cubecloud/Python/projects/rlbinancetrader/handoff/diag_failing_bars.json")
    outp.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("saved", outp)


if __name__ == "__main__":
    main()
