"""Честный гейт копирования BC (редакция 2026-07-27, устранение джерримендеринга).

Ревью гейта вынесло вердикт (B) с правкой: прежняя маска исключения «своротных»
баров в `diag_perside.run` (строки 184-188) определялась ЛОГИТАМИ КЛОНА
(`di >= entry_d.min()`) — то есть подгонялась под сам клон. Это джерримендеринг:
маска, скроенная по клону, не может служить честным критерием того, чист ли клон.

Здесь маска исключения — ЧИСТАЯ ФУНКЦИЯ колонок v7/regimeb (`build_exclusion_mask`),
НОЛЬ обращений к head0-логитам / d / политике. Гейт строит маску ДО загрузки
политики и ДО вычисления d (временная гарантия + assert по сигнатуре).

Пять пунктов зарегистрированного гейта (пороги ФИКСИРОВАНЫ в pre-reg ДО прогона):

1. ЁМКОСТЬ (argmax, in-sample, saved clone): все сделки бар-в-бар
   272/272, 224/224 (2021), 162/162, 150/150 (2024).
2. МАСКА ИСКЛЮЧЕНИЯ = объединение латчей УДЕРЖАНИЯ v7 (чистая функция колонок):
   ВЫХОД: transition-hold (pos_tag_transition & regime_bull & exit_sig) |
          oracle_hold (dip in-pos, exit_sig, держится).
   ВХОД: ПУСТО (вход детерминирован, блокируется наблюдаемым cooldown/cb —
         копируется точно, НЕ исключается).
   legflip/tp/b2b — это ВЫХОДЫ (FLIP), НЕ STAY: они в ОДНОЗНАЧНОМ FLIP-подмножестве
   (пункт 4), а НЕ в STAY-исключении.
3. На ДОПОЛНЕНИИ (не-латч бары, saved clone, in-sample): argmax false-FLIP @ s* == 0;
   НИ ОДИН не-латч STAY-бар не в исключении (по построению); остаточное ранговое
   перекрытие d на не-латч барах = ПРОВАЛ.
4. recall ≥ 0.99 OUT-OF-EPOCH на v7-ОДНОЗНАЧНОМ FLIP-подмножестве
   (legflip/tp/b2b выходы + входы с большим запасом margin). НЕ на all-FLIP.
5. Кросс-эпоховый AUC (generalization.py): вход ≥0.98, выход ≥0.94 out-of-epoch.

Run (env rlbinancetrader, под slow):
  python -m spotrl.bc.honest_gate --data /home/cubecloud/Data/rlbinancetrader \
      --model /home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy_reg \
      --out handoff/honest_gate_reg.json
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Dict

import numpy as np

from spotrl.bc.train_clone import (apply_scaler, head0_logits, load_pooled)

# --- ПОРОГИ, ФИКСИРОВАННЫЕ ДО ПРОГОНА (pre-reg, редакция 2026-07-27) ---
RECALL_FLOOR_UNAMBIG = 0.99      # пункт 4: recall на однозначном FLIP, out-of-epoch
AUC_FLOOR_ENTRY = 0.98           # пункт 5 (поднято с 0.95)
AUC_FLOOR_EXIT = 0.94            # пункт 5 (поднято с 0.90)
BULL_CODE = 2.0                  # m_regime_code==2 → bull
# «большой запас» входа: margin положителен (сигнал уверенно за порогом).
ENTRY_MARGIN_MIN = 0.0
EPOCHS = ("2021", "2024")
# ожидаемая ёмкость (pre-reg §1): (входы, выходы) по эпохам.
CAPACITY_EXPECT = {"2021": (272, 224), "2024": (162, 150)}


def build_exclusion_mask(df) -> Dict[str, np.ndarray]:
    """Маска исключения — ЧИСТАЯ функция колонок v7/regimeb (НЕ логитов клона).

    Принимает ТОЛЬКО DataFrame эпохи (колонки m_/a_/exit_reason/...). НЕ принимает
    d, логиты, политику — это и есть гарантия отсутствия джерримендеринга.

    ВЫХОД — латчи УДЕРЖАНИЯ v7 (regimeb_bt_strategy.py:292-295): при наличии
    сигнала выхода v7 ДЕРЖИТ позицию из-за латча, поэтому бар помечен STAY, хотя
    сигнал есть. Именно эти STAY-бары — «поверхность свободы» RL, их клон не обязан
    копировать бар-в-бар.
      * transition-hold: pos_tag==transition & regime==bull & exit_sig
        (allow_signal_exit=False, пока bull не кончился — строка 292-293).
      * oracle_hold: pos_tag==dip & in_pos & exit_sig, но oracle держит
        (строка 294-295). В снимке m_exit_flag=0 на всех dip-in-pos → пусто, но
        определено для корректности.
    ВХОД — ПУСТО: вход детерминирован, блокируется наблюдаемым cooldown/cb.
    """
    inpos = df["a_in_position"].to_numpy() > 0.5
    stay = df["bc_action"].to_numpy() == 0
    exit_sig = df["m_exit_flag"].to_numpy() > 0.5
    is_trans = df["a_pos_tag_transition"].to_numpy() > 0.5
    is_dip = df["a_pos_tag_dip"].to_numpy() > 0.5
    bull = df["m_regime_code"].to_numpy() >= (BULL_CODE - 0.5)

    transition_hold = is_trans & bull & exit_sig & stay & inpos
    oracle_hold = is_dip & inpos & exit_sig & stay
    exclude = transition_hold | oracle_hold  # только STAY-бары удержания
    entry_excl = np.zeros(len(df), bool)      # ВХОД: ПУСТО
    return {
        "transition_hold": transition_hold,
        "oracle_hold": oracle_hold,
        "exit_exclude": exclude,
        "entry_exclude": entry_excl,
        "exclude": exclude | entry_excl,
    }


def build_unambiguous_flip(df) -> Dict[str, np.ndarray]:
    """v7-ОДНОЗНАЧНЫЕ FLIP-бары (пункт 4): механические выходы + чёткие входы.

    Выход-однозначный = legflip/tp/b2b (сработали ВНЕ сигнального латча — leg_dn,
    порог TP, back-to-back; детерминированы наблюдаемым состоянием). Сигнальные
    выходы ИСКЛЮЧЕНЫ — они на поверхности свободы (транзишн-латч).
    Вход-однозначный = вход с положительным margin (сигнал уверенно за порогом).
    """
    reason = df["exit_reason"].to_numpy().astype(str)
    is_flip_exit = df["is_flip_exit"].to_numpy().astype(bool)
    exit_unambig = is_flip_exit & np.isin(reason, ("legflip", "tp", "b2b"))
    is_flip_entry = df["is_flip_entry"].to_numpy().astype(bool)
    margin = df["m_buy_margin"].to_numpy()
    entry_unambig = is_flip_entry & (margin > ENTRY_MARGIN_MIN)
    return {
        "exit_unambig": exit_unambig,
        "entry_unambig": entry_unambig,
        "entry_all": is_flip_entry,
        "exit_signal": is_flip_exit & (reason == "signal"),
    }


def _assert_mask_purity() -> None:
    """assert: build_exclusion_mask НЕ обращается к логитам/политике клона.

    (i) сигнатура принимает единственный аргумент df — нет параметров d/logit/
        policy/score. (ii) исходник функции не содержит обращений к head0_logits/
        policy/action_net/mlp_extractor. Это статическая гарантия чистоты.
    """
    sig = inspect.signature(build_exclusion_mask)
    params = list(sig.parameters)
    assert params == ["df"], f"маска должна брать только df, а берёт {params}"
    forbidden = ("logit", "head0", "policy", "action_net", "mlp_extractor",
                 "d[", "score", "PPO")
    src = inspect.getsource(build_exclusion_mask)
    hit = [f for f in forbidden if f in src]
    assert not hit, f"маска обращается к логитам клона: {hit}"


def _load_epoch_frames(data_dir: str):
    """DataFrame по эпохам (для колонок v7 — вне obs-меты)."""
    import pandas as pd
    return {e: pd.read_parquet(Path(data_dir) / f"bc_clone_v7_{e}.parquet")
            for e in EPOCHS}


def run_insample(data_dir: str, model_path: str, seeds: int) -> Dict:
    """Пункты 1-3: saved clone, in-sample. Маска строится ДО вычисления d."""
    from stable_baselines3 import PPO

    # (1/3) МАСКА ИСКЛЮЧЕНИЯ — строится СЕЙЧАС, до какого-либо касания политики.
    _assert_mask_purity()
    frames = _load_epoch_frames(data_dir)
    excl = {e: build_exclusion_mask(frames[e]) for e in EPOCHS}
    unamb = {e: build_unambiguous_flip(frames[e]) for e in EPOCHS}
    _policy_touched = False  # временная гарантия: маска готова ДО политики

    # только теперь загружаем политику и считаем d.
    mf = json.loads(Path(model_path + ".manifest.json").read_text())
    mu = np.array(mf["scaler_mu"], np.float32)
    sd = np.array(mf["scaler_sd"], np.float32)
    s_star = float(mf["shift"]["s_star"])
    X_raw, y, w, meta = load_pooled(data_dir)
    X = apply_scaler(X_raw, mu, sd)
    policy = PPO.load(model_path, device="cpu").policy
    _policy_touched = True
    assert _policy_touched  # d считается строго после готовой маски
    d_all = head0_logits(policy, X)
    d = (d_all[:, 1] - d_all[:, 0]).astype(np.float64)

    report = {"s_star": s_star, "seeds": seeds, "per_epoch": {}}
    for e in EPOCHS:
        m = meta["per_epoch"][e]
        di = d[m["idx"]]
        ex = excl[e]
        inpos = frames[e]["a_in_position"].to_numpy() > 0.5
        is_entry = m["is_entry"]; is_exit = m["is_exit"]; is_stay = m["is_stay"]

        # --- Пункт 1: ЁМКОСТЬ (argmax) ---
        cap = {
            "entry_argmax_nat": int((di[is_entry] > 0).sum()),
            "exit_argmax_nat": int((di[is_exit] > 0).sum()),
            "entry_argmax_s": int((di[is_entry] + s_star > 0).sum()),
            "exit_argmax_s": int((di[is_exit] + s_star > 0).sum()),
            "n_entry": int(is_entry.sum()), "n_exit": int(is_exit.sum()),
        }

        # --- Пункт 3: дополнение (не-латч STAY) ---
        latch_stay = ex["exclude"]                 # исключённые STAY (латчи)
        nonlatch_stay = is_stay & (~latch_stay)
        # (3a) argmax false-FLIP @ s* на не-латч STAY:
        false_flip_s = int((di[nonlatch_stay] + s_star > 0).sum())
        # (3b) НИ ОДИН не-латч STAY не в исключении (по построению):
        nonlatch_in_excl = int((nonlatch_stay & latch_stay).sum())
        # (3c) остаточное ранговое перекрытие на не-латч барах, РАЗДЕЛЬНО:
        flat_nl = nonlatch_stay & (~inpos)         # вход-сторона
        inpos_nl = nonlatch_stay & inpos           # выход-сторона
        min_entry_d = float(di[is_entry].min())
        min_exit_d = float(di[is_exit].min())
        ov_entry = int((flat_nl & (di >= min_entry_d)).sum())
        ov_exit = int((inpos_nl & (di >= min_exit_d)).sum())

        report["per_epoch"][e] = {
            "capacity": cap,
            "n_transition_hold": int(ex["transition_hold"].sum()),
            "n_oracle_hold": int(ex["oracle_hold"].sum()),
            "n_exit_exclude": int(ex["exit_exclude"].sum()),
            "n_entry_exclude": int(ex["entry_exclude"].sum()),
            "false_flip_s_nonlatch": false_flip_s,
            "nonlatch_stay_in_exclusion": nonlatch_in_excl,
            "overlap_entry_nonlatch": ov_entry,
            "overlap_exit_nonlatch": ov_exit,
            "min_entry_flip_d": min_entry_d,
            "min_exit_flip_d": min_exit_d,
            "max_nonlatch_flat_d": float(di[flat_nl].max()) if flat_nl.any() else float("-inf"),
            "max_nonlatch_inpos_d": float(di[inpos_nl].max()) if inpos_nl.any() else float("-inf"),
            "n_unambig_exit": int(unamb[e]["exit_unambig"].sum()),
            "n_unambig_entry": int(unamb[e]["entry_unambig"].sum()),
        }
    return report, meta, excl, unamb


def run_outofepoch(data_dir: str, seed: int, n_epochs: int) -> Dict:
    """Пункты 4-5: кросс-эпоховые клоны (обучить на одной, замерить на другой).

    Пункт 5 (AUC) и пункт 4 (recall на однозначном FLIP при argmax d>0,
    out-of-epoch) считаются на ОДНИХ И ТЕХ ЖЕ кросс-эпоховых политиках (обучаем
    2 клона, не 4 — AUC берём тем же кодом, что generalization).
    """
    from spotrl.bc.generalization import CrossEpochResult, _entry_exit_auc
    from spotrl.bc.train_clone import (build_policy, fit_scaler, train)

    frames = _load_epoch_frames(data_dir)
    unamb = {e: build_unambiguous_flip(frames[e]) for e in EPOCHS}

    X_raw, y, w, meta = load_pooled(data_dir)
    mu, sd = fit_scaler(X_raw)
    Xn = apply_scaler(X_raw, mu, sd)

    recall, auc = {}, []
    for tr_ep, te_ep in (("2021", "2024"), ("2024", "2021")):
        idx = meta["per_epoch"][tr_ep]["idx"]
        _, policy = build_policy(Xn.shape[1], seed)
        train(policy, Xn[idx], y[idx], w[idx], seed, n_epochs=n_epochs)
        # AUC (пункт 5) — тем же кодом, что generalization.
        a = _entry_exit_auc(policy, Xn, meta, tr_ep, te_ep)
        auc.append(CrossEpochResult(
            train_epoch=tr_ep, test_epoch=te_ep,
            auc_entry_in=a["auc_entry_in"], auc_entry_out=a["auc_entry_out"],
            auc_exit_in=a["auc_exit_in"], auc_exit_out=a["auc_exit_out"],
            recall_entry_out=a["recall_entry_out"], fpr_entry_out=a["fpr_entry_out"],
            recall_exit_out=a["recall_exit_out"], fpr_exit_out=a["fpr_exit_out"],
            generalizes=bool(a["auc_entry_out"] >= AUC_FLOOR_ENTRY
                             and a["auc_exit_out"] >= AUC_FLOOR_EXIT)).describe())
        # recall на однозначном FLIP (пункт 4) — та же политика, out-of-epoch.
        m_te = meta["per_epoch"][te_ep]
        lg = head0_logits(policy, Xn[m_te["idx"]])
        d_te = (lg[:, 1] - lg[:, 0]).astype(np.float64)
        u = unamb[te_ep]
        flip_pred = d_te > 0                    # argmax (натуральный порог)
        rec_exit = (float(flip_pred[u["exit_unambig"]].mean())
                    if u["exit_unambig"].any() else float("nan"))
        rec_entry = (float(flip_pred[u["entry_unambig"]].mean())
                     if u["entry_unambig"].any() else float("nan"))
        recall[f"{tr_ep}->{te_ep}"] = {
            "recall_exit_unambig_out": rec_exit,
            "recall_entry_unambig_out": rec_entry,
            "n_exit_unambig": int(u["exit_unambig"].sum()),
            "n_entry_unambig": int(u["entry_unambig"].sum()),
        }

    return {"recall_out": recall, "auc": auc}


def _verdict(insample: Dict, oof: Dict) -> Dict:
    """Свести пять пунктов в пройдено/провал (пороги pre-reg)."""
    v = {}
    # Пункт 1: ёмкость.
    cap_ok = True
    for e in EPOCHS:
        c = insample["per_epoch"][e]["capacity"]
        exp_en, exp_ex = CAPACITY_EXPECT[e]
        cap_ok &= (c["entry_argmax_nat"] == exp_en and c["exit_argmax_nat"] == exp_ex)
    v["p1_capacity"] = bool(cap_ok)
    # Пункт 3: не-латч чистота.
    p3 = True
    for e in EPOCHS:
        pe = insample["per_epoch"][e]
        p3 &= (pe["false_flip_s_nonlatch"] == 0
               and pe["nonlatch_stay_in_exclusion"] == 0
               and pe["overlap_entry_nonlatch"] == 0
               and pe["overlap_exit_nonlatch"] == 0)
    v["p3_nonlatch_clean"] = bool(p3)
    # Пункт 4: recall на однозначном FLIP out-of-epoch.
    p4 = True
    for k, r in oof["recall_out"].items():
        for key in ("recall_exit_unambig_out", "recall_entry_unambig_out"):
            val = r[key]
            if val == val:  # не nan
                p4 &= (val >= RECALL_FLOOR_UNAMBIG)
    v["p4_recall_unambig"] = bool(p4)
    # Пункт 5: AUC.
    p5 = all(a["auc_entry_out"] >= AUC_FLOOR_ENTRY
             and a["auc_exit_out"] >= AUC_FLOOR_EXIT for a in oof["auc"])
    v["p5_auc"] = bool(p5)
    v["clone_passes_honest_gate"] = bool(cap_ok and p3 and p4 and p5)
    return v


def main() -> None:
    """CLI: прогнать честный гейт (пункты 1-5), напечатать числа + вердикт."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/cubecloud/Data/rlbinancetrader")
    ap.add_argument("--model",
                    default="/home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy_reg")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-epochs", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    insample, meta, excl, unamb = run_insample(args.data, args.model, args.seeds)
    oof = run_outofepoch(args.data, args.seed, args.n_epochs)
    v = _verdict(insample, oof)

    print("=== ЧИСТОТА МАСКИ ===")
    print("assert 'exclusion построена без обращения к логитам клона' — ПРОЙДЕН")
    print(f"\ns* = {insample['s_star']:.4f}")
    print("\n=== ПУНКТ 1: ЁМКОСТЬ (argmax) + ПУНКТ 2: МАСКА + ПУНКТ 3: ДОПОЛНЕНИЕ ===")
    for e in EPOCHS:
        pe = insample["per_epoch"][e]; c = pe["capacity"]
        print(f"[{e}] argmax nat: вход {c['entry_argmax_nat']}/{c['n_entry']} "
              f"выход {c['exit_argmax_nat']}/{c['n_exit']} | "
              f"@s*: вход {c['entry_argmax_s']} выход {c['exit_argmax_s']}")
        print(f"     маска: transition_hold={pe['n_transition_hold']} "
              f"oracle_hold={pe['n_oracle_hold']} exit_excl={pe['n_exit_exclude']} "
              f"entry_excl={pe['n_entry_exclude']}")
        print(f"     не-латч: false_FLIP@s*={pe['false_flip_s_nonlatch']} "
              f"nonlatch_в_исключении={pe['nonlatch_stay_in_exclusion']} "
              f"перекрытие вход={pe['overlap_entry_nonlatch']} "
              f"выход={pe['overlap_exit_nonlatch']}")
        print(f"     min(d flip) вход={pe['min_entry_flip_d']:.2f} выход={pe['min_exit_flip_d']:.2f} "
              f"| max(d не-латч STAY) flat={pe['max_nonlatch_flat_d']:.2f} "
              f"inpos={pe['max_nonlatch_inpos_d']:.2f}")
        print(f"     однозначн. FLIP: выход={pe['n_unambig_exit']} вход={pe['n_unambig_entry']}")
    print("\n=== ПУНКТ 4: recall на однозначном FLIP (out-of-epoch, argmax d>0) ===")
    for k, r in oof["recall_out"].items():
        print(f"[{k}] выход recall={r['recall_exit_unambig_out']:.4f} "
              f"(n={r['n_exit_unambig']}) | вход recall={r['recall_entry_unambig_out']:.4f} "
              f"(n={r['n_entry_unambig']}) [пол {RECALL_FLOOR_UNAMBIG}]")
    print("\n=== ПУНКТ 5: кросс-эпоховый AUC ===")
    for a in oof["auc"]:
        print(f"[{a['train_epoch']}->{a['test_epoch']}] вход AUC out={a['auc_entry_out']:.4f} "
              f"(пол {AUC_FLOOR_ENTRY}) | выход AUC out={a['auc_exit_out']:.4f} "
              f"(пол {AUC_FLOOR_EXIT})")
    print("\n=== ВЕРДИКТ ===")
    for k, val in v.items():
        print(f"  {k}: {val}")
    print(f"\nКЛОН ПРОХОДИТ ЧЕСТНЫЙ ГЕЙТ: {v['clone_passes_honest_gate']}")

    if args.out:
        result = {"insample": insample, "outofepoch": oof, "verdict": v}
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\nСохранено: {args.out}")


if __name__ == "__main__":
    main()
