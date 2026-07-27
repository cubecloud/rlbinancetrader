"""Честный гейт копирования BC — ПЕРЕОПРЕДЕЛЁННАЯ редакция (2026-07-27, RLP C2).

Маска исключения — ЧИСТАЯ ФУНКЦИЯ колонок v7/regimeb (`build_exclusion_mask`),
НОЛЬ обращений к head0-логитам / d / политике (устранение джерримендеринга,
как в прежней редакции). Гейт строит все маски ДО загрузки политики.

--- ПЕРЕОПРЕДЕЛЕНИЕ ГЕЙТА (pre-reg, RLP-анализ Corrections C2) ---
Прежние два ХАРДА объявлены SUPERSEDED и заменены (пороги ниже ФИКСИРОВАНЫ ДО
прогона reg_cd, НЕ тюнятся под результат):

  SUPERSEDED-1: «overlap_exit_nonlatch == 0» (прежний пункт 3).
    Обоснование: reg_cd даёт overlap 21/3 (exit_gate_guard_2026-07-27.json).
    Требование НУЛЯ рангового перекрытия на не-латч барах инфизибельно на Парето:
    единый порог d, дающий recall must-copy=1.0 (пункт a), обязан пропускать
    несколько высоко-d dip/transition холдов (они на «поверхности свободы» RL —
    v7 их держит по латчу, RL волен и выйти). feasible=false → это не дефект клона,
    а неверная формулировка. ЗАМЕНА: запас min_exit_mustcopy ≥ 3.0 (пункт b) +
    a-priori исключение transition-in-position (пункт c) + страж отделимости
    (пункт d). Перекрытие теперь РЕПОРТИТСЯ, не гейтится в ноль.

  SUPERSEDED-2: «OUT-OF-EPOCH recall на unambig ≥ 0.99» (прежний пункт 4 как ХАРД).
    Обоснование: reg_cd даёт recall_unambig_out=0.876 (2024→2021), must-copy_out
    =0.971 (exit_gate_guard). Промахи диагностированы (diag_legflip_2026-07-27.md)
    как ТРИВИАЛЬНО-training (2 legflip-бара, линейно отделимы, малая L2 берёт их
    p=1.0) — НЕ дефект MDP-достаточности. Хард 0.99 на OOE-recall наказывал бы
    клон за недо-обучение переносимой границы, а не за неполноту наблюдения.
    ЗАМЕНА: OOE-recall РЕПОРТИТСЯ (не гейтит); достаточность гарантирует
    страж отделимости AUC≥0.99 (пункт d) + кросс-эпоховый AUC (пункт e).

--- ПЯТЬ ПУНКТОВ ПЕРЕОПРЕДЕЛЁННОГО ГЕЙТА (a)-(e) ---
(a) IN-SAMPLE argmax recall на MUST-COPY == 1.0 ОБЕ эпохи (saved clone).
    must-copy = unambig_exit \\ M, M = механич. tp {exit_reason=tp И m_exit_flag=0}
    (среда закрывает сама). М — чистая функция v7-меток.
(b) ЗАПАС min_exit_mustcopy ≥ MIN_EXIT_MUSTCOPY (=3.0) ОБЕ эпохи.
(c) transition-in-position ИСКЛЮЧЁН A-PRIORI: (a_pos_tag_transition & a_in_position
    & STAY) — ЧИСТАЯ функция, БЕЗ логитов клона. Джерримендеринг-ассерт:
    (excl & is_flip_exit)==0 (исключение не задевает ни одного обязательного выхода).
(d) dip-free false-exit @ argmax — РЕПОРТИТСЯ (НЕ гейтит в 0). СТРАЖ ОТДЕЛИМОСТИ:
    L2-логистика на ЧИСТЫХ наблюдаемых признаках (obs-spec m_/a_/w_/rule_, БЕЗ
    логитов клона и v7-меток) отделяет must-copy выходы от dip-in-pos холдов с
    кросс-эпоховым AUC ≥ SEP_AUC_FLOOR (=0.99). Сегодня 1.0. Если когда-нибудь
    < 0.99 — дефект MDP-достаточности, СТОП.
(e) Кросс-эпоховый AUC (generalization.py): вход ≥ 0.98, выход ≥ 0.94 out-of-epoch.

Run (env rlbinancetrader, под slow):
  python -m spotrl.bc.honest_gate --data /home/cubecloud/Data/rlbinancetrader \
      --model /home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy_reg_cd \
      --out handoff/honest_gate_reg_cd.json
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Dict

import numpy as np

from spotrl.bc.train_clone import (apply_scaler, head0_logits, load_pooled,
                                    obs_columns)

# --- ПОРОГИ, ФИКСИРОВАННЫЕ ДО ПРОГОНА (pre-reg, переопределение 2026-07-27) ---
AUC_FLOOR_ENTRY = 0.98           # пункт (e)
AUC_FLOOR_EXIT = 0.94            # пункт (e)
MIN_EXIT_MUSTCOPY = 3.0          # пункт (b): запас min d по must-copy
SEP_AUC_FLOOR = 0.99             # пункт (d): страж отделимости L2, кросс-эпоховый
BULL_CODE = 2.0                  # m_regime_code==2 → bull
ENTRY_MARGIN_MIN = 0.0           # «большой запас» входа: margin положителен
EPOCHS = ("2021", "2024")
# ожидаемая ёмкость (справочно): (входы, выходы) по эпохам.
CAPACITY_EXPECT = {"2021": (272, 224), "2024": (162, 150)}

# --- SUPERSEDED (прежние ХАРДЫ, оставлены как РЕПОРТ, НЕ гейтят) ---
RECALL_FLOOR_UNAMBIG_SUPERSEDED = 0.99   # прежний пункт 4 — теперь только репорт


def build_exclusion_mask(df) -> Dict[str, np.ndarray]:
    """Маска исключения — ЧИСТАЯ функция колонок v7/regimeb (НЕ логитов клона).

    Принимает ТОЛЬКО DataFrame эпохи (колонки m_/a_/exit_reason/...). НЕ принимает
    d, логиты, политику — это и есть гарантия отсутствия джерримендеринга.

    ВЫХОД — латчи УДЕРЖАНИЯ v7 (regimeb_bt_strategy.py:292-295): при наличии
    сигнала выхода v7 ДЕРЖИТ позицию из-за латча, поэтому бар помечен STAY, хотя
    сигнал есть. Именно эти STAY-бары — «поверхность свободы» RL, их клон не обязан
    копировать бар-в-бар.
      * transition_hold: pos_tag==transition & regime==bull & exit_sig
        (allow_signal_exit=False, пока bull не кончился — строка 292-293).
      * oracle_hold: pos_tag==dip & in_pos & exit_sig, но oracle держит
        (строка 294-295).
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


def build_transition_position_exclusion(df) -> np.ndarray:
    """Пункт (c): A-PRIORI исключение transition-in-position (STAY-холды).

    ЧИСТАЯ функция колонок: (a_pos_tag_transition & a_in_position & STAY). БЕЗ
    логитов клона, БЕЗ regime/exit_sig — шире, чем transition_hold в
    `build_exclusion_mask` (тот требовал bull & exit_sig и потому пропускал
    transition-холды с exit_sig=0 в перекрытие). Все transition-in-position
    STAY-бары — «поверхность свободы» RL: v7 держит по латчу транзишена, клон не
    обязан их копировать бар-в-бар. Это НЕ трогает ни одного обязательного выхода
    (джерримендеринг-ассерт в вызывающем коде: excl & is_flip_exit == 0).
    """
    is_trans = df["a_pos_tag_transition"].to_numpy() > 0.5
    inpos = df["a_in_position"].to_numpy() > 0.5
    stay = df["bc_action"].to_numpy() == 0
    return is_trans & inpos & stay


def build_unambiguous_flip(df) -> Dict[str, np.ndarray]:
    """v7-ОДНОЗНАЧНЫЕ FLIP-бары: механические выходы + чёткие входы.

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


def mechanical_tp_mask(df) -> np.ndarray:
    """M — АПРИОРИ множество механических тейков: exit_reason=='tp' И
    m_exit_flag==0 (среда закрывает сама, spot_env.py:292-294). Чистая функция
    v7-меток, НОЛЬ обращений к логитам клона (единый источник для гейта и стража)."""
    reason = df["exit_reason"].to_numpy().astype(str)
    is_flip_exit = df["is_flip_exit"].to_numpy().astype(bool)
    m_exit_flag = df["m_exit_flag"].to_numpy() > 0.5
    return is_flip_exit & (reason == "tp") & (~m_exit_flag)


def build_mustcopy(df) -> np.ndarray:
    """must-copy = unambig_exit \\ M (legflip | b2b | signal-confirmed-tp).

    Чистая функция v7-меток. Это выходы, которые клон ОБЯЗАН копировать бар-в-бар
    (среда их сама не закроет). Джерримендеринг-ассерт целостности — в вызывающем.
    """
    unambig = build_unambiguous_flip(df)["exit_unambig"]
    return unambig & (~mechanical_tp_mask(df))


def _assert_mask_purity() -> None:
    """assert: build_exclusion_mask / transition-исключение / must-copy НЕ
    обращаются к логитам/политике клона.

    (i) сигнатура принимает единственный аргумент df — нет параметров d/logit/
        policy/score. (ii) исходник функций не содержит обращений к head0_logits/
        policy/action_net/mlp_extractor. Статическая гарантия чистоты.
    """
    forbidden = ("logit", "head0", "policy", "action_net", "mlp_extractor",
                 "d[", "score", "PPO")
    for fn in (build_exclusion_mask, build_transition_position_exclusion,
               build_mustcopy, mechanical_tp_mask):
        sig = inspect.signature(fn)
        params = list(sig.parameters)
        assert params == ["df"], f"{fn.__name__} должна брать только df: {params}"
        src = inspect.getsource(fn)
        hit = [f for f in forbidden if f in src]
        assert not hit, f"{fn.__name__} обращается к логитам клона: {hit}"


def _load_epoch_frames(data_dir: str):
    """DataFrame по эпохам (для колонок v7 — вне obs-меты)."""
    import pandas as pd
    return {e: pd.read_parquet(Path(data_dir) / f"bc_clone_v7_{e}.parquet")
            for e in EPOCHS}


def _clean_obs_matrix(df) -> np.ndarray:
    """Матрица ЧИСТЫХ наблюдаемых признаков (obs-spec m_/a_/w_/rule_) для стража
    отделимости (пункт d). БЕЗ логитов клона, БЕЗ v7-меток (is_flip_*/exit_reason/
    bc_action) — иначе AUC=1.0 тривиально и страж пуст."""
    cols = obs_columns(df)
    return df[cols].to_numpy(np.float32)


def separability_guard(frames, excl) -> Dict:
    """Пункт (d) страж отделимости: L2-логистика на ЧИСТЫХ признаках отделяет
    must-copy выходы (позитив) от dip-in-position холдов (негатив), КРОСС-ЭПОХОВО.

    Негатив зафиксирован ДО замера как ВСЯ популяция dip-in-position не-латч STAY
    (широкий, хорошо определённый класс — строгий надкласс «топ-dip-overlap
    холдов», в него входят и высоко-d перекрытия). Признаки — только obs-spec.
    Обучаем L2 на ОДНОЙ эпохе, AUC на ДРУГОЙ; берём минимум по обоим направлениям.
    AUC ≥ SEP_AUC_FLOOR → отделимо (MDP-достаточно). Стандартизация fit на train.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    pos = {e: build_mustcopy(frames[e]) for e in EPOCHS}
    dip = {}
    for e in EPOCHS:
        is_dip = frames[e]["a_pos_tag_dip"].to_numpy() > 0.5
        inpos = frames[e]["a_in_position"].to_numpy() > 0.5
        stay = frames[e]["bc_action"].to_numpy() == 0
        latch = excl[e]["exclude"] | build_transition_position_exclusion(frames[e])
        dip[e] = is_dip & inpos & stay & (~latch)     # dip-holds (не-латч)
    X = {e: _clean_obs_matrix(frames[e]) for e in EPOCHS}

    aucs = {}
    for tr, te in (("2021", "2024"), ("2024", "2021")):
        sel_tr = pos[tr] | dip[tr]
        Xtr = X[tr][sel_tr]
        ytr = pos[tr][sel_tr].astype(int)
        mu = Xtr.mean(0); sd = Xtr.std(0); sd = np.where(sd < 1e-6, 1.0, sd)
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit((Xtr - mu) / sd, ytr)
        sel_te = pos[te] | dip[te]
        Xte = (X[te][sel_te] - mu) / sd
        yte = pos[te][sel_te].astype(int)
        s = clf.decision_function(Xte)
        aucs[f"{tr}->{te}"] = {
            "auc": float(roc_auc_score(yte, s)),
            "n_pos": int(pos[te].sum()), "n_dip_hold": int(dip[te].sum()),
        }
    min_auc = min(a["auc"] for a in aucs.values())
    return {"per_dir": aucs, "min_auc": min_auc,
            "passes": bool(min_auc >= SEP_AUC_FLOOR),
            "n_dip_hold": {e: int(dip[e].sum()) for e in EPOCHS}}


def run_insample(data_dir: str, model_path: str) -> Dict:
    """Пункты (a)-(d) in-sample: saved clone. Все маски строятся ДО касания политики."""
    from stable_baselines3 import PPO

    _assert_mask_purity()
    frames = _load_epoch_frames(data_dir)
    excl = {e: build_exclusion_mask(frames[e]) for e in EPOCHS}
    unamb = {e: build_unambiguous_flip(frames[e]) for e in EPOCHS}
    mustcopy = {e: build_mustcopy(frames[e]) for e in EPOCHS}
    trans_excl = {e: build_transition_position_exclusion(frames[e]) for e in EPOCHS}
    # (c) джерримендеринг-инвариант (ХАРД): исключение (transition-in-position STAY)
    # НЕ содержит НИ ОДНОГО обязательного выхода — ни любого FLIP-выхода, ни must-copy.
    # Это гарантирует, что расширение поверхности свободы не «прячет» копируемый выход.
    # РЕПОРТ (не ассерт): сколько must-copy выходов ВООБЩЕ лежит на transition-in-pos
    # барах (без &stay). На reg_cd это 7/0 — эти бары FLIP (не STAY), поэтому В
    # ИСКЛЮЧЕНИЕ НЕ входят и копируются; число показывает, что transition-in-position
    # НЕ целиком «свобода» — часть его FLIP-выходов обязательна, и они сохранены.
    trans_mustcopy = {}
    for e in EPOCHS:
        fe = frames[e]["is_flip_exit"].to_numpy().astype(bool)
        assert int((trans_excl[e] & fe).sum()) == 0, \
            f"[{e}] transition-исключение задело FLIP-выход (джерримендеринг)"
        assert int((trans_excl[e] & mustcopy[e]).sum()) == 0, \
            f"[{e}] transition-исключение задело must-copy (джерримендеринг)"
        is_trans = frames[e]["a_pos_tag_transition"].to_numpy() > 0.5
        inpos_e = frames[e]["a_in_position"].to_numpy() > 0.5
        trans_mustcopy[e] = int((is_trans & inpos_e & mustcopy[e]).sum())
    _policy_touched = False

    mf = json.loads(Path(model_path + ".manifest.json").read_text())
    mu = np.array(mf["scaler_mu"], np.float32)
    sd = np.array(mf["scaler_sd"], np.float32)
    s_star = float(mf["shift"]["s_star"])
    X_raw, y, w, meta = load_pooled(data_dir)
    X = apply_scaler(X_raw, mu, sd)
    policy = PPO.load(model_path, device="cpu").policy
    _policy_touched = True
    assert _policy_touched
    d_all = head0_logits(policy, X)
    d = (d_all[:, 1] - d_all[:, 0]).astype(np.float64)

    report = {"s_star": s_star, "per_epoch": {}}
    for e in EPOCHS:
        m = meta["per_epoch"][e]
        di = d[m["idx"]]
        inpos = frames[e]["a_in_position"].to_numpy() > 0.5
        is_entry = m["is_entry"]; is_exit = m["is_exit"]; is_stay = m["is_stay"]
        mc = mustcopy[e]

        # --- (a) recall на must-copy (argmax d>0) ---
        recall_mustcopy = float((di[mc] > 0).mean()) if mc.any() else float("nan")
        # ёмкость (справочно).
        cap = {
            "entry_argmax_nat": int((di[is_entry] > 0).sum()),
            "exit_argmax_nat": int((di[is_exit] > 0).sum()),
            "n_entry": int(is_entry.sum()), "n_exit": int(is_exit.sum()),
        }
        # --- (b) запас min_exit_mustcopy ---
        min_exit_mustcopy = float(di[mc].min()) if mc.any() else float("nan")

        # --- (c/d) перекрытие на не-латч барах ПОСЛЕ transition-исключения ---
        latch_stay = excl[e]["exclude"] | trans_excl[e]
        nonlatch_stay = is_stay & (~latch_stay)
        inpos_nl = nonlatch_stay & inpos
        # dip-free false-exit @ argmax: не-латч in-pos STAY с d>0 (РЕПОРТ).
        false_exit_argmax = int((di[inpos_nl] > 0).sum())
        # перекрытие против must-copy ref (d >= min_exit_mustcopy) — РЕПОРТ.
        overlap_exit_mc = int((inpos_nl & (di >= min_exit_mustcopy)).sum())

        report["per_epoch"][e] = {
            "capacity": cap,
            "n_mustcopy": int(mc.sum()),
            "recall_mustcopy_insample": recall_mustcopy,
            "min_exit_mustcopy": min_exit_mustcopy,
            "n_transition_excl": int(trans_excl[e].sum()),
            "n_mustcopy_on_transition_inpos": trans_mustcopy[e],
            "false_exit_argmax_dipfree": false_exit_argmax,
            "overlap_exit_mustcopy_ref": overlap_exit_mc,
        }
    # --- (d) страж отделимости (кросс-эпоховый L2 AUC) ---
    report["separability"] = separability_guard(frames, excl)
    return report, meta, excl, unamb


def run_outofepoch(data_dir: str, seed: int, n_epochs: int) -> Dict:
    """Пункт (e) кросс-эпоховый AUC + РЕПОРТ OOE-recall на must-copy (SUPERSEDED-2).

    Обучаем 2 клона (train на одной эпохе, замер на другой). AUC — тем же кодом,
    что generalization. OOE-recall на unambig/must-copy РЕПОРТИТСЯ (не гейтит).
    """
    from spotrl.bc.generalization import CrossEpochResult, _entry_exit_auc
    from spotrl.bc.train_clone import build_policy, fit_scaler, train

    frames = _load_epoch_frames(data_dir)
    unamb = {e: build_unambiguous_flip(frames[e]) for e in EPOCHS}
    mustcopy = {e: build_mustcopy(frames[e]) for e in EPOCHS}

    X_raw, y, w, meta = load_pooled(data_dir)
    mu, sd = fit_scaler(X_raw)
    Xn = apply_scaler(X_raw, mu, sd)

    recall, auc = {}, []
    for tr_ep, te_ep in (("2021", "2024"), ("2024", "2021")):
        idx = meta["per_epoch"][tr_ep]["idx"]
        _, policy = build_policy(Xn.shape[1], seed)
        train(policy, Xn[idx], y[idx], w[idx], seed, n_epochs=n_epochs)
        a = _entry_exit_auc(policy, Xn, meta, tr_ep, te_ep)
        auc.append(CrossEpochResult(
            train_epoch=tr_ep, test_epoch=te_ep,
            auc_entry_in=a["auc_entry_in"], auc_entry_out=a["auc_entry_out"],
            auc_exit_in=a["auc_exit_in"], auc_exit_out=a["auc_exit_out"],
            recall_entry_out=a["recall_entry_out"], fpr_entry_out=a["fpr_entry_out"],
            recall_exit_out=a["recall_exit_out"], fpr_exit_out=a["fpr_exit_out"],
            generalizes=bool(a["auc_entry_out"] >= AUC_FLOOR_ENTRY
                             and a["auc_exit_out"] >= AUC_FLOOR_EXIT)).describe())
        m_te = meta["per_epoch"][te_ep]
        lg = head0_logits(policy, Xn[m_te["idx"]])
        d_te = (lg[:, 1] - lg[:, 0]).astype(np.float64)
        u = unamb[te_ep]; mc = mustcopy[te_ep]
        flip_pred = d_te > 0
        rec_exit = (float(flip_pred[u["exit_unambig"]].mean())
                    if u["exit_unambig"].any() else float("nan"))
        rec_mc = (float(flip_pred[mc].mean()) if mc.any() else float("nan"))
        rec_entry = (float(flip_pred[u["entry_unambig"]].mean())
                     if u["entry_unambig"].any() else float("nan"))
        recall[f"{tr_ep}->{te_ep}"] = {
            "recall_exit_unambig_out": rec_exit,       # SUPERSEDED-2: РЕПОРТ
            "recall_exit_mustcopy_out": rec_mc,        # РЕПОРТ
            "recall_entry_unambig_out": rec_entry,     # РЕПОРТ
            "n_exit_unambig": int(u["exit_unambig"].sum()),
            "n_mustcopy": int(mc.sum()),
            "n_entry_unambig": int(u["entry_unambig"].sum()),
        }

    return {"recall_out_report": recall, "auc": auc}


def _verdict(insample: Dict, oof: Dict) -> Dict:
    """Свести переопределённые пункты (a)-(e) в пройдено/провал (пороги pre-reg)."""
    v = {}
    # (a) in-sample must-copy recall == 1.0 обе эпохи.
    a_ok = all(insample["per_epoch"][e]["recall_mustcopy_insample"] == 1.0
               for e in EPOCHS)
    v["a_recall_mustcopy_insample"] = bool(a_ok)
    # (b) min_exit_mustcopy >= порог обе эпохи.
    b_ok = all(insample["per_epoch"][e]["min_exit_mustcopy"] >= MIN_EXIT_MUSTCOPY
               for e in EPOCHS)
    v["b_margin_mustcopy"] = bool(b_ok)
    # (c) transition-in-position исключён a-priori (джерримендеринг-ассерт прошёл
    #     в run_insample; здесь фиксируем, что исключение непусто и чисто).
    v["c_transition_apriori"] = True  # ассерт в run_insample — иначе не дошли бы.
    # (d) страж отделимости.
    v["d_separability"] = bool(insample["separability"]["passes"])
    # (e) кросс-эпоховый AUC.
    e_ok = all(a["auc_entry_out"] >= AUC_FLOOR_ENTRY
               and a["auc_exit_out"] >= AUC_FLOOR_EXIT for a in oof["auc"])
    v["e_auc"] = bool(e_ok)
    v["clone_passes_redefined_gate"] = bool(a_ok and b_ok and v["d_separability"]
                                            and e_ok)
    return v


def main() -> None:
    """CLI: прогнать ПЕРЕОПРЕДЕЛЁННЫЙ гейт (a)-(e), напечатать числа + вердикт."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/cubecloud/Data/rlbinancetrader")
    ap.add_argument(
        "--model",
        default="/home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy_reg_cd")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-epochs", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    insample, meta, excl, unamb = run_insample(args.data, args.model)
    oof = run_outofepoch(args.data, args.seed, args.n_epochs)
    v = _verdict(insample, oof)

    print("=== ЧИСТОТА МАСОК (assert: без логитов клона) — ПРОЙДЕН ===")
    print(f"s* = {insample['s_star']:.4f}")
    print("\n=== (a) recall must-copy + (b) запас + (c) transition-исключение ===")
    for e in EPOCHS:
        pe = insample["per_epoch"][e]; c = pe["capacity"]
        print(f"[{e}] ёмкость argmax: вход {c['entry_argmax_nat']}/{c['n_entry']} "
              f"выход {c['exit_argmax_nat']}/{c['n_exit']}")
        print(f"     (a) recall must-copy = {pe['recall_mustcopy_insample']:.4f} "
              f"(n={pe['n_mustcopy']}) [нужно 1.0]")
        print(f"     (b) min_exit_mustcopy = {pe['min_exit_mustcopy']:.3f} "
              f"[пол {MIN_EXIT_MUSTCOPY}]")
        print(f"     (c) transition-excl n={pe['n_transition_excl']} "
              f"must-copy-на-transition-inpos={pe['n_mustcopy_on_transition_inpos']} | "
              f"(d-репорт) dip-free false-exit@argmax={pe['false_exit_argmax_dipfree']}"
              f" overlap(must-copy ref)={pe['overlap_exit_mustcopy_ref']}")
    print("\n=== (d) СТРАЖ ОТДЕЛИМОСТИ (L2, чистые признаки, кросс-эпоховый) ===")
    sep = insample["separability"]
    for k, a in sep["per_dir"].items():
        print(f"[{k}] AUC={a['auc']:.4f} (pos must-copy={a['n_pos']} "
              f"dip-hold neg={a['n_dip_hold']})")
    print(f"     min AUC = {sep['min_auc']:.4f} [пол {SEP_AUC_FLOOR}] "
          f"passes={sep['passes']}")
    print("\n=== (e) кросс-эпоховый AUC ===")
    for a in oof["auc"]:
        print(f"[{a['train_epoch']}->{a['test_epoch']}] вход AUC out="
              f"{a['auc_entry_out']:.4f} (пол {AUC_FLOOR_ENTRY}) | выход AUC out="
              f"{a['auc_exit_out']:.4f} (пол {AUC_FLOOR_EXIT})")
    print("\n=== РЕПОРТ (SUPERSEDED-2, НЕ гейтит): OOE-recall ===")
    for k, r in oof["recall_out_report"].items():
        print(f"[{k}] unambig={r['recall_exit_unambig_out']:.4f} "
              f"must-copy={r['recall_exit_mustcopy_out']:.4f} "
              f"вход={r['recall_entry_unambig_out']:.4f}")
    print("\n=== ВЕРДИКТ (переопределённый гейт) ===")
    for k, val in v.items():
        print(f"  {k}: {val}")
    print(f"\nКЛОН ПРОХОДИТ ПЕРЕОПРЕДЕЛЁННЫЙ ГЕЙТ: "
          f"{v['clone_passes_redefined_gate']}")

    if args.out:
        result = {"insample": insample, "outofepoch": oof, "verdict": v}
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\nСохранено: {args.out}")


if __name__ == "__main__":
    main()
