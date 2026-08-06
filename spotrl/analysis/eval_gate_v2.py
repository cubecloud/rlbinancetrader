"""Гейт 4.2-v2: переопределённая метрика эффекта раннего выхода + калибровка.

Проблема старого гейта (amend1 §4). Критерий «медиана парной дельты RL−v7 по
ВСЕМ dip-сделкам > 0» механически даёт 0: агент действует на меньшинстве
сделок (напр. 22 из 245), значит >90% дельт — структурные нули, и медиана с
bootstrap-CI схлопывается в [0,0] для ЛЮБОЙ редко-действующей политики,
включая идеальный oracle. То есть прибор запрещает успех целому классу
правильных гипотез.

Новая метрика (см. amend2 §4). Два критерия, оба на одном векторе per-trade
дельт по всем dip-сделкам:

  (A) ПОРТФЕЛЬНЫЙ: Σ дельт по ВСЕМ dip-сделкам > 0, нижняя граница 95%
      bootstrap-CI суммы (ресэмпл сделок с возвратом) > 0. Нули нейтральны для
      суммы — совокупный эффект на портфель не размывается.
  (B) КАЧЕСТВО ДЕЙСТВИЙ: на acted-subset (сделки, где политика реально
      отклонилась от v7) односторонний Wilcoxon p < 0.01 (дельты > 0) И нижняя
      граница bootstrap-CI медианы acted-дельт > 0.

Контекст (не гейт): доля acted = n_acted / n_dip и распределение acted-дельт.

Вердикт «ЭФФЕКТ ЕСТЬ» = (A) И (B). Калибровка прибора (обязательна):
  * ORACLE (лучший ранний выход по будущей цене) ОБЯЗАН пройти гейт;
  * RANDOM (тот же счётчик выходов, случайный момент) ОБЯЗАН провалиться.
Если oracle не проходит ИЛИ random проходит — метрика сломана.

Единицы. Дельта per-trade = (_ret(exit, entry) − _ret(v7_exit, entry))·100 п.п.,
_ret с двусторонней комиссией (та же, что в eval_paired_dip). Все политики
(v7-база, oracle, random, RL) считаются в ОДНОЙ конвенции — сравнимо.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from scipy.stats import wilcoxon

from spotrl.analysis.eval_paired_dip import (
    BASE, DATA, MANIFEST, SEED_MODEL, _prices, _ret, _roll_rl, _roll_v7,
)

_STRATEGY_RL = Path(__file__).resolve().parents[2] / "strategy_rl"
if str(_STRATEGY_RL) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_RL))


# ----------------------------------------------------------------------------
# bootstrap helpers
# ----------------------------------------------------------------------------
def _boot_sum_ci(x, n=5000, seed=0):
    """95% bootstrap-CI СУММЫ (ресэмпл сделок с возвратом, размер сохранён)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) < 3:
        return float("nan"), float("nan")
    sums = [x[rng.integers(0, len(x), len(x))].sum() for _ in range(n)]
    return float(np.percentile(sums, 2.5)), float(np.percentile(sums, 97.5))


def _boot_median_ci(x, n=5000, seed=0):
    """95% bootstrap-CI медианы."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) < 3:
        return float("nan"), float("nan")
    meds = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def new_gate(deltas_all, label=""):
    """Гейт 4.2-v2 на векторе per-trade дельт (п.п.) по ВСЕМ dip-сделкам.

    deltas_all: np.array длины n_dip; 0 = политика не отклонилась от v7.
    Возвращает dict с критериями A (портфель) и B (качество действий) и pass.
    """
    d = np.asarray(deltas_all, float)
    n = int(len(d))
    acted_mask = d != 0.0
    acted = d[acted_mask]
    n_acted = int(acted_mask.sum())

    total = float(d.sum())
    mean = float(d.mean()) if n else float("nan")
    sum_lo, sum_hi = _boot_sum_ci(d)
    gate_A = bool(np.isfinite(sum_lo) and sum_lo > 0.0)

    if n_acted >= 3 and not np.allclose(acted, 0.0):
        try:
            w_p = float(wilcoxon(acted, alternative="greater").pvalue)
        except Exception:
            w_p = float("nan")
        amed = float(np.median(acted))
        amean = float(np.mean(acted))
        amed_lo, amed_hi = _boot_median_ci(acted)
    else:
        w_p = amed = amean = amed_lo = amed_hi = float("nan")
    gate_B = bool(np.isfinite(w_p) and w_p < 0.01
                  and np.isfinite(amed_lo) and amed_lo > 0.0)

    return {
        "label": label, "n_dip": n, "n_acted": n_acted,
        "acted_share": (n_acted / n) if n else float("nan"),
        "A_portfolio": {
            "sum_pp": total, "mean_pp": mean,
            "ci_lo_pp": sum_lo, "ci_hi_pp": sum_hi, "pass": gate_A,
        },
        "B_acted_quality": {
            "n_acted": n_acted, "acted_median_pp": amed, "acted_mean_pp": amean,
            "wilcoxon_p_greater": w_p,
            "acted_median_ci_lo_pp": amed_lo, "acted_median_ci_hi_pp": amed_hi,
            "pass": gate_B,
        },
        "EFFECT_PRESENT": bool(gate_A and gate_B),
    }


# ----------------------------------------------------------------------------
# per-trade policies in the frozen paired space
# ----------------------------------------------------------------------------
def _dip_trades(v7, postag):
    """Список (entry_bar, entry_price, xb_v7, xp_v7, r_v7) по dip-сделкам."""
    out = []
    for eb, (ep, xb_v7, xp_v7) in v7.items():
        if postag.get(eb) != "dip":
            continue
        out.append((eb, ep, xb_v7, xp_v7, _ret(xp_v7, ep)))
    return out


def oracle_deltas(v7, op, postag):
    """ORACLE: лучший СТРОГО РАННИЙ выход по будущему open, если бьёт v7.

    Окно решения [eb+1 .. xb_v7-1], fill = open(d+1). Дельта ≥ 0 по построению
    (не берём, если не лучше v7). Возврат: (deltas[n_dip], acted_entry_bars).
    """
    n = len(op)
    deltas, acted = [], []
    for eb, ep, xb_v7, xp_v7, r_v7 in _dip_trades(v7, postag):
        best = 0.0
        for d in range(eb + 1, xb_v7):
            f = d + 1
            if f >= n:
                break
            g = (_ret(op[f], ep) - r_v7) * 100.0
            if g > best:
                best = g
        deltas.append(best)
        if best > 0.0:
            acted.append(eb)
    return np.asarray(deltas, float), acted


def random_deltas(v7, op, postag, n_acted, seed):
    """RANDOM: тот же счётчик выходов, что у oracle (n_acted), но случайный
    момент на случайно выбранных dip-сделках. Одна реализация (для гейта).

    Возврат: deltas[n_dip] (0 на не-выбранных сделках).
    """
    rng = np.random.default_rng(seed)
    n = len(op)
    trades = _dip_trades(v7, postag)
    idx_eligible = [i for i, t in enumerate(trades) if t[2] - 1 >= t[0] + 1]
    k = min(n_acted, len(idx_eligible))
    chosen = set(rng.choice(idx_eligible, size=k, replace=False).tolist()) \
        if k > 0 else set()
    deltas = []
    for i, (eb, ep, xb_v7, xp_v7, r_v7) in enumerate(trades):
        if i in chosen:
            d = int(rng.integers(eb + 1, xb_v7))       # случайный ранний бар
            f = min(d + 1, n - 1)
            deltas.append((_ret(op[f], ep) - r_v7) * 100.0)
        else:
            deltas.append(0.0)
    return np.asarray(deltas, float)


def random_sum_distribution(v7, op, postag, n_acted, R=200, seed0=1000):
    """Устойчивость random: распределение Σ дельт по R реализациям."""
    sums = [float(random_deltas(v7, op, postag, n_acted, seed0 + r).sum())
            for r in range(R)]
    s = np.asarray(sums)
    return {"R": R, "sum_mean_pp": float(s.mean()),
            "sum_p2.5_pp": float(np.percentile(s, 2.5)),
            "sum_p97.5_pp": float(np.percentile(s, 97.5)),
            "share_positive": float((s > 0).mean())}


def rl_deltas(v7, rl, op, postag):
    """RL: per-trade дельта argmax-политики (из _roll_rl) vs v7."""
    deltas, acted = [], []
    for eb, ep, xb_v7, xp_v7, r_v7 in _dip_trades(v7, postag):
        xb_rl, xp_rl = rl.get(eb, (xb_v7, xp_v7))
        g = (_ret(xp_rl, ep) - r_v7) * 100.0
        deltas.append(g)
        if xb_rl != xb_v7:
            acted.append(eb)
    return np.asarray(deltas, float), acted


# ----------------------------------------------------------------------------
# K2: тегирование по причине выхода v7 и знаку oracle-резерва
# ----------------------------------------------------------------------------
def exit_reason_map(state, v7):
    """book entry_bar -> exit_reason v7 (join label_trades по entry_bar)."""
    from label_exits import label_trades
    _st, lab, _bc = label_trades(state)
    lab_eb = lab["entry_bar"].to_numpy()
    lab_reason = lab["exit_reason"].to_numpy()
    order = np.argsort(lab_eb)
    lab_eb_s, lab_reason_s = lab_eb[order], lab_reason[order]
    out, matched = {}, 0
    for eb in v7:
        j = int(np.searchsorted(lab_eb_s, eb))
        best = None
        for jj in (j - 1, j):
            if 0 <= jj < len(lab_eb_s) and abs(int(lab_eb_s[jj]) - eb) <= 1:
                best = str(lab_reason_s[jj])
        if best is not None:
            matched += 1
        out[eb] = best
    return out, matched


def k2_tagged(v7, rl, op, postag, reason_map, oracle_delta_by_eb):
    """Дельта RL−v7 РАЗДЕЛЬНО по SL-сделкам и восстанавливающимся, и по знаку
    oracle-резерва (есть запас / нет)."""
    groups = {"sl": [], "recovering": [],
              "reserve_pos": [], "reserve_zero": []}
    rl_acted_on = {"sl": 0, "recovering": 0, "reserve_pos": 0, "reserve_zero": 0}
    for eb, ep, xb_v7, xp_v7, r_v7 in _dip_trades(v7, postag):
        xb_rl, xp_rl = rl.get(eb, (xb_v7, xp_v7))
        g = (_ret(xp_rl, ep) - r_v7) * 100.0
        acted = xb_rl != xb_v7
        rsn = reason_map.get(eb)
        gkey = "sl" if rsn == "sl" else "recovering"
        groups[gkey].append(g)
        if acted:
            rl_acted_on[gkey] += 1
        rkey = "reserve_pos" if oracle_delta_by_eb.get(eb, 0.0) > 0 else "reserve_zero"
        groups[rkey].append(g)
        if acted:
            rl_acted_on[rkey] += 1

    def summ(name):
        """Сводка группы: n, acted, сумма/среднее дельт (пп)."""
        a = np.asarray(groups[name], float)
        nz = a[a != 0]
        return {"n": int(len(a)), "rl_acted": rl_acted_on[name],
                "sum_pp": float(a.sum()), "mean_pp": float(a.mean()) if len(a) else float("nan"),
                "acted_mean_pp": float(nz.mean()) if len(nz) else float("nan")}
    return {k: summ(k) for k in groups}


# ----------------------------------------------------------------------------
def _build_postag(ds_path):
    """postag: book entry_bar (=dataset bar+1) -> pos_tag (dip-фильтр eval)."""
    df = pd.read_parquet(ds_path)
    ent = df[df["is_flip_entry"].astype(bool)]
    return {int(b) + 1: str(t) for b, t in zip(ent["bar"], ent["pos_tag"])}


def _load_rl(ckpt):
    """Загрузить argmax-политику чекпойнта (как в eval_paired_dip.evaluate)."""
    mf = json.loads(open(MANIFEST).read())
    mu = np.array(mf["scaler_mu"], np.float32)
    sd = np.array(mf["scaler_sd"], np.float32)
    cd_idx = mf["obs_cols"].index("a_cooldown_remain")
    from spotrl.algo.ppo_scaffold import _dummy_vec_env, build_scaffold
    model, _clone = build_scaffold(_dummy_vec_env(39), SEED_MODEL, MANIFEST,
                                   kl_coef=0.0, ent_coef=0.0, gamma=1.0,
                                   n_steps=64, batch_size=64, n_epochs=1)
    model.policy.load_state_dict(th.load(ckpt, map_location="cpu"))
    model.policy.set_training_mode(False)
    return model.policy, mu, sd, cd_idx


def run(state, sig, ds_path, ckpt=None, label="train_2021"):
    """К1 (калибровка oracle/random) + К2 (тегирование RL), один разрез."""
    df_ds = pd.read_parquet(ds_path)
    v7, op = _roll_v7(state, sig, df_ds)
    postag = _build_postag(ds_path)
    op = np.asarray(op, float)

    # --- К1: калибровка прибора ---
    od, oacted = oracle_deltas(v7, op, postag)
    n_acted_oracle = len(oacted)
    oracle_gate = new_gate(od, "ORACLE")
    rd = random_deltas(v7, op, postag, n_acted_oracle, seed=0)
    random_gate = new_gate(rd, "RANDOM")
    random_robust = random_sum_distribution(v7, op, postag, n_acted_oracle)

    res = {
        "label": label, "ckpt": ckpt,
        "n_dip": int(sum(1 for eb in v7 if postag.get(eb) == "dip")),
        "calibration": {
            "ORACLE": oracle_gate,
            "RANDOM": random_gate,
            "RANDOM_robustness": random_robust,
            "instrument_valid": bool(oracle_gate["EFFECT_PRESENT"]
                                     and not random_gate["EFFECT_PRESENT"]),
        },
    }

    # --- К1.3 + К2: RL чекпойнт ---
    if ckpt:
        policy, mu, sd, cd_idx = _load_rl(ckpt)
        rl, early = _roll_rl(state, sig, df_ds, policy, mu, sd, cd_idx)
        rl_d, rl_acted = rl_deltas(v7, rl, op, postag)
        res["RL_new_gate"] = new_gate(rl_d, "RL_r20")
        res["RL_new_gate"]["early_exit_count_driver"] = int(early)
        reason_map, matched = exit_reason_map(state, v7)
        oracle_delta_by_eb = {eb: float(g) for eb, g in
                              zip([t[0] for t in _dip_trades(v7, postag)], od)}
        res["K2_tagging"] = {
            "reason_match_rate": matched / max(1, len(v7)),
            "groups": k2_tagged(v7, rl, op, postag, reason_map,
                                oracle_delta_by_eb),
        }
    return res


def _fmt_gate(g):
    a, b = g["A_portfolio"], g["B_acted_quality"]
    return (f"{g['label']}: EFFECT={g['EFFECT_PRESENT']} | "
            f"A(sum)={a['sum_pp']:+.2f}пп CI[{a['ci_lo_pp']:+.2f},{a['ci_hi_pp']:+.2f}] "
            f"pass={a['pass']} | acted={g['n_acted']}/{g['n_dip']} "
            f"B(med)={b['acted_median_pp']:.3f} p={b['wilcoxon_p_greater']} "
            f"CI_lo={b['acted_median_ci_lo_pp']} pass={b['pass']}")


def main():
    """CLI: калибровка гейта + (опц.) ре-оценка чекпойнта новой меркой."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    state = f"{BASE}/sunday_tests/state_v0/state_v3_causal_2021-01_2024-03.parquet"
    sig = f"{DATA}/v7signals_2021.parquet"
    ds = f"{DATA}/bc_clone_v7_2021.parquet"
    res = run(state, sig, ds, ckpt=args.ckpt)

    print("\n===== К1: КАЛИБРОВКА ПРИБОРА (train_2021) =====")
    print(_fmt_gate(res["calibration"]["ORACLE"]))
    print(_fmt_gate(res["calibration"]["RANDOM"]))
    rr = res["calibration"]["RANDOM_robustness"]
    print(f"RANDOM устойчивость: Σ mean={rr['sum_mean_pp']:+.2f}пп "
          f"[{rr['sum_p2.5_pp']:+.2f},{rr['sum_p97.5_pp']:+.2f}] "
          f"доля Σ>0={rr['share_positive']:.2f}")
    print(f"ПРИБОР ВАЛИДЕН (oracle прошёл И random провалился): "
          f"{res['calibration']['instrument_valid']}")

    if args.ckpt:
        print("\n===== К1.3: чекпойнт r20 новой меркой (факт, не вердикт) =====")
        print(_fmt_gate(res["RL_new_gate"]))
        print(f"ранних выходов (драйвер) = {res['RL_new_gate']['early_exit_count_driver']}")
        print("\n===== К2: тегирование дельты RL−v7 =====")
        k2 = res["K2_tagging"]
        print(f"reason_match_rate = {k2['reason_match_rate']:.3f}")
        for name, s in k2["groups"].items():
            print(f"  {name:12s}: n={s['n']:3d} rl_acted={s['rl_acted']:3d} "
                  f"Σ={s['sum_pp']:+.2f}пп mean={s['mean_pp']:+.4f} "
                  f"acted_mean={s['acted_mean_pp']}")

    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
        print("\nsaved", args.out)


if __name__ == "__main__":
    main()
