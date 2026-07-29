"""К3 — проба ДОСТИЖИМОСТИ резерва Э0.2 из наблюдения агента (supervised).

Вопрос. Можно ли из наблюдения политики (те же 39 признаков, что видит агент на
dip-in-position баре) ЗАРАНЕЕ отличить сделку, идущую в SL, от восстанавливающейся?
Если нет — резерв Э0.2 (сосредоточен в 37 SL-сделках, K2) недостижим никаким RL.

Гранулярность. Метка ставится на dip-in-position БАР (ровно то, на чём политика
условливается при решении FLIP): y=1, если сделка, к которой принадлежит бар,
закрывается по SL (exit_reason из label_trades, reason_match=1.0 в K2), иначе y=0.
Бар-уровень выбран потому, что агент решает по-барно; трейд-уровень (одна метка
на сделку) даётся как сводка.

Честность OOS. Обучение и оценка НЕ пересекаются по сделкам: сделки упорядочены
по entry_bar, первые 70% -> train, последние 30% -> test (эмбарго естественное:
интервалы сделок не пересекаются). Финальный holdout 2026-01..07 НЕ трогается.
tune-fold 2024-03..2026-01 не используется (holdout-смежность, и вердикт не
обучаем на нём). CI AUC — bootstrap по ТЕСТ-СДЕЛКАМ (не по барам): бары внутри
сделки автокоррелированы, ресэмпл сделок не завышает CI.

Контроль утечки. Метки перемешиваются НА УРОВНЕ СДЕЛКИ (весь набор баров сделки
меняет метку синхронно), модель переобучается — AUC обязан упасть к 0.5.

ПОРОГ ЗАФИКСИРОВАН ДО ЗАМЕРА (не подгонять под вывод):
  ДОСТИЖИМ  <=>  нижняя граница 95% CI OOS-AUC (GBM, бар-уровень) > 0.60
                 И shuffle-control AUC in [0.45, 0.55].
  Если точечный AUC > 0.60, но CI_lo <= 0.60 -> СЛАБО/indeterminate.
  Если AUC ~ 0.5 -> НЕДОСТИЖИМ из текущего наблюдения.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

_STRATEGY_RL = Path(__file__).resolve().parents[2] / "strategy_rl"
if str(_STRATEGY_RL) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_RL))

BASE = "/home/cubecloud/Data"
DATA = f"{BASE}/rlbinancetrader"
STATE_2021 = f"{BASE}/sunday_tests/state_v0/state_v3_causal_2021-01_2024-03.parquet"
DS_2021 = f"{DATA}/bc_clone_v7_2021.parquet"

TRAIN_FRAC = 0.70            # доля сделок (по entry_bar) в train
AUC_CI_LO_THRESHOLD = 0.60   # ДОСТИЖИМ, если нижняя граница CI OOS-AUC > этого
SHUFFLE_BAND = (0.45, 0.55)  # shuffle-control обязан попасть сюда

# признаки-константы на dip-in-position подмножестве -> исключаются
_CONST_ON_DIP = ("a_in_position", "a_pos_tag_dip", "a_pos_tag_none",
                 "a_pos_tag_transition")


def _assign_trades(dip_bars: np.ndarray, trades: pd.DataFrame):
    """Каждому dip-in-position бару -> id сделки и метка y (SL?). Интервалы
    [entry_bar, exit_bar] из label_trades (dip-сделки), непересекающиеся."""
    dip = trades[trades["pos_tag"] == "dip"].sort_values("entry_bar")
    eb = dip["entry_bar"].to_numpy()
    xb = dip["exit_bar"].to_numpy()
    reason = dip["exit_reason"].to_numpy()
    trade_id = np.full(len(dip_bars), -1, np.int64)
    y = np.zeros(len(dip_bars), np.int8)
    j = np.searchsorted(eb, dip_bars, side="right") - 1
    valid = (j >= 0) & (j < len(eb))
    inside = valid & (dip_bars <= xb[np.clip(j, 0, len(eb) - 1)])
    trade_id[inside] = j[inside]
    y[inside] = (reason[np.clip(j, 0, len(eb) - 1)] == "sl")[inside].astype(np.int8)
    return trade_id, y, len(dip)


def _boot_auc_ci(y, p, groups, n=1000, seed=0):
    """95% bootstrap-CI AUC ресэмплом по ГРУППАМ (сделкам)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    aucs = []
    for _ in range(n):
        gs = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        aucs.append(roc_auc_score(yy, p[idx]))
    a = np.asarray(aucs)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), float(a.mean())


def _fit_eval(Xtr, ytr, Xte, yte, gte, model):
    """Обучить модель, вернуть OOS-AUC + CI (bootstrap по тест-сделкам)."""
    model.fit(Xtr, ytr)
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(Xte)[:, 1]
    else:
        p = model.decision_function(Xte)
    auc = float(roc_auc_score(yte, p))
    lo, hi, mean = _boot_auc_ci(yte, p, gte)
    return auc, lo, hi, mean, p


def run(early_only: bool = False, seed: int = 0):
    """Полная проба на 2021-01..2024-03. early_only=True — только первая половина
    удержания каждой dip-сделки (проверка РАННЕЙ отличимости)."""
    from label_exits import label_trades
    _st, trades, _bc = label_trades(STATE_2021)

    df = pd.read_parquet(DS_2021)
    df["a_cooldown_active"] = (df["a_cooldown_remain"] > 0).astype(np.float32)
    m = (df["a_in_position"] > 0.5) & (df["a_pos_tag_dip"] > 0.5)
    d = df[m].reset_index(drop=True)
    bars = d["bar"].to_numpy()
    trade_id, y, n_dip = _assign_trades(bars, trades)
    keep = trade_id >= 0
    d, bars, trade_id, y = d[keep].reset_index(drop=True), bars[keep], \
        trade_id[keep], y[keep]

    if early_only:
        # позиция бара внутри сделки; берём первую половину удержания
        order = np.argsort(trade_id, kind="stable")
        frac = np.empty(len(trade_id))
        for tid in np.unique(trade_id):
            idx = np.where(trade_id == tid)[0]
            frac[idx] = np.arange(len(idx)) / max(1, len(idx) - 1)
        em = frac <= 0.5
        d, bars, trade_id, y = d[em].reset_index(drop=True), bars[em], \
            trade_id[em], y[em]

    feat_cols = [c for c in _obs_cols() if c not in _CONST_ON_DIP]
    X = d[feat_cols].to_numpy(np.float32)

    # split по сделкам: первые 70% (по entry_bar-порядку id) -> train
    uid = np.unique(trade_id)
    n_tr = int(len(uid) * TRAIN_FRAC)
    tr_ids, te_ids = set(uid[:n_tr].tolist()), set(uid[n_tr:].tolist())
    is_tr = np.array([t in tr_ids for t in trade_id])
    Xtr, ytr = X[is_tr], y[is_tr]
    Xte, yte, gte = X[~is_tr], y[~is_tr], trade_id[~is_tr]

    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    gbm = HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.05,
        class_weight="balanced", random_state=seed)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")

    gbm_auc, gbm_lo, gbm_hi, gbm_bmean, _ = _fit_eval(Xtr, ytr, Xte, yte, gte, gbm)
    lr_auc, lr_lo, lr_hi, _, _ = _fit_eval(Xtr_s, ytr, Xte_s, yte, gte, lr)

    # shuffle-control (перемешать метки НА УРОВНЕ СДЕЛКИ, train). Одна
    # перестановка при 12 SL-сделках в тесте очень шумна -> усредняем по 25.
    tr_uid = np.array(sorted(tr_ids))
    lab_by_tid = {t: int(y[trade_id == t][0]) for t in tr_uid}
    base_lab = np.array([lab_by_tid[t] for t in tr_uid])
    shuf_aucs = []
    for s in range(25):
        rng = np.random.default_rng(seed + 100 + s)
        shuf_lab = dict(zip(tr_uid, rng.permutation(base_lab)))
        ytr_shuf = np.array([shuf_lab[t] for t in trade_id[is_tr]], np.int8)
        gbm_s = HistGradientBoostingClassifier(
            max_depth=4, max_iter=200, learning_rate=0.05,
            class_weight="balanced", random_state=seed)
        gbm_s.fit(Xtr, ytr_shuf)
        shuf_aucs.append(float(roc_auc_score(yte, gbm_s.predict_proba(Xte)[:, 1])))
    shuf_arr = np.asarray(shuf_aucs)
    shuf_auc = float(shuf_arr.mean())
    shuf_sd = float(shuf_arr.std())
    shuf_p975 = float(np.percentile(shuf_arr, 97.5))

    # важности признаков (permutation на GBM, тест)
    top = _perm_importance(gbm, Xte, yte, feat_cols, seed)

    reachable = bool(gbm_lo > AUC_CI_LO_THRESHOLD
                     and SHUFFLE_BAND[0] <= shuf_auc <= SHUFFLE_BAND[1])
    return {
        "cut": "early_first_half" if early_only else "all_dip_bars",
        "n_dip_trades": int(n_dip),
        "n_bars_total": int(len(y)), "n_bars_pos_sl": int(y.sum()),
        "sl_bar_frac": float(y.mean()),
        "n_trades_train": int(len(tr_ids)), "n_trades_test": int(len(te_ids)),
        "n_sl_trades_test": int(sum(lab_by_tid_test(y, trade_id, te_ids))),
        "GBM": {"oos_auc": gbm_auc, "ci_lo": gbm_lo, "ci_hi": gbm_hi,
                "ci_mean": gbm_bmean},
        "LogReg": {"oos_auc": lr_auc, "ci_lo": lr_lo, "ci_hi": lr_hi},
        "shuffle_control_auc": shuf_auc,
        "shuffle_control_sd": shuf_sd,
        "shuffle_control_p975": shuf_p975,
        "baseline_random_auc": 0.5,
        "threshold_ci_lo": AUC_CI_LO_THRESHOLD,
        "shuffle_band": list(SHUFFLE_BAND),
        "REACHABLE": reachable,
        "top_features": top,
    }


def lab_by_tid_test(y, trade_id, te_ids):
    """Метки SL по тест-сделкам (для подсчёта числа SL в тесте)."""
    return [int(y[trade_id == t][0]) for t in sorted(te_ids)]


def _perm_importance(model, Xte, yte, feat_cols, seed, top_k=8):
    """Permutation importance по AUC на тесте (падение AUC при перемешивании)."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    drops = []
    for j, name in enumerate(feat_cols):
        Xp = Xte.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        auc = roc_auc_score(yte, model.predict_proba(Xp)[:, 1])
        drops.append((name, float(base - auc)))
    drops.sort(key=lambda z: -z[1])
    return [{"feature": n, "auc_drop": d} for n, d in drops[:top_k]]


def _obs_cols():
    """39 признаков наблюдения политики (manifest порядок)."""
    mf = json.loads(open(f"{DATA}/bc_clone_v7_policy_reg_cd_vw2.manifest.json").read())
    return list(mf["obs_cols"])


def main():
    """CLI: проба достижимости (all-bars + early-half), durable JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = {"all": run(early_only=False), "early": run(early_only=True)}
    for k, r in res.items():
        g = r["GBM"]
        print(f"\n=== cut={r['cut']} ===")
        print(f"dip-сделок={r['n_dip_trades']} баров={r['n_bars_total']} "
              f"SL-баров={r['n_bars_pos_sl']} ({r['sl_bar_frac']:.3f}) | "
              f"train/test сделок={r['n_trades_train']}/{r['n_trades_test']} "
              f"SL в тесте={r['n_sl_trades_test']}")
        print(f"GBM OOS-AUC={g['oos_auc']:.3f} CI[{g['ci_lo']:.3f},{g['ci_hi']:.3f}] "
              f"| LogReg AUC={r['LogReg']['oos_auc']:.3f} "
              f"CI[{r['LogReg']['ci_lo']:.3f},{r['LogReg']['ci_hi']:.3f}]")
        print(f"shuffle-control AUC={r['shuffle_control_auc']:.3f} "
              f"(band {r['shuffle_band']}) | порог CI_lo>{r['threshold_ci_lo']}")
        print(f"REACHABLE = {r['REACHABLE']}")
        print("top признаки (AUC drop):",
              ", ".join(f"{f['feature']}={f['auc_drop']:.3f}" for f in r["top_features"]))
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
        print("\nsaved", args.out)


if __name__ == "__main__":
    main()
