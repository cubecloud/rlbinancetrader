"""Коллизионный тест ВЫХОДА (ШАГ 1 pre-reg BC) — отделимость выходов.

Зачем. Разрешение «A=0 на выходе» (argmax обязан воспроизвести все сигнальные
выходы бар-в-бар) держится на доказательстве отделимости: не существует STAY-бара
В ПОЗИЦИИ, неотличимого по наблюдению от FLIP-выхода.

ЧЕСТНОСТЬ МЕТОДА (важно). Полный 38-вектор наблюдения непригоден как ключ: в obs
есть непрерывные признаки (m_ret_close, a_unreal_pnl, m_buy_margin ...), поэтому
две строки НИКОГДА не совпадут побитово — точный конфликт вернёт ВАКУУМНЫЙ 0 вне
зависимости от отделимости. Наоборот, ОДИНОЧНЫЙ булев ключ (m_exit_flag/m_leg_dn)
срабатывает на десятках тысяч STAY-баров, потому что v7 выходит по КОНЪЮНКЦИИ
условий, а не по одному флагу — одиночный ключ даёт грубо-завышенную верхнюю
границу, НЕ k. Поэтому:
  * НЕвакуумный, содержательный тест — только tp-полоса: TP v7 — чистый порог на
    a_unreal_pnl per pos_tag. L = min(a_unreal_pnl по tp-выходам данного pos_tag);
    коллизия tp = STAY-in-position того же pos_tag с a_unreal>=L. Это единственный
    непрерывно-пороговый механизм → единственная реальная «полоса неотделимости».
  * точный конфликт наблюдений (буквальное определение pre-reg) репортится как
    справка, но помечен ВАКУУМНЫМ (все in-pos наблюдения уникальны);
  * одиночные ключи signal/b2b/legflip репортятся как ГРУБАЯ верхняя граница, НЕ k.
Окончательное finite-capacity подтверждение отделимости сигнал/legflip-выходов —
эмпирический argmax capacity precheck (ШАГ 3 часть A): argmax-клон обязан
воспроизвести ВСЕ выходы бар-в-бар. Здесь доказывается только tp-полоса.

k (связывающий, = k_tp) одновременно ограничивает потолок recall выхода
(коллизионный бар: клон выдаёт одинаковую P(FLIP) на STAY и FLIP → сдвиг под FPR
топит и FLIP-выход). Допуски (recall_flip>=99.5% per-epoch):
  * 2024 выход 150 баров → нужно >=149.25 → k=0 обязателен (k=1 = 99.33% < 99.5%);
  * 2021 выход 224 бара  → нужно >=222.88 → k<=1 проходит (223/224 = 99.55%).

Тест ЧИСТО по датасету (наблюдения записаны под экспертной траекторией — ровно то,
на что кондиционируется argmax-клон). Среда не нужна.

Run (env rlbinancetrader):
  python -m spotrl.bc.exit_collision --data /home/cubecloud/Data/rlbinancetrader
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Допуск числа промахов выхода на эпоху при пороге recall_flip>=99.5%.
EXIT_RECALL_FLOOR = 0.995


@dataclass
class ExitCollisionReport:
    """Результат коллизионного теста выхода одной эпохи."""

    epoch: str
    n_in_position: int = 0
    n_stay_in_pos: int = 0
    n_flip_exit: int = 0
    # k (связывающий) = k_tp — единственный НЕвакуумный тест (непрерывный порог).
    k_tp: int = 0
    # Ниже — НЕ k: грубые верхние границы по одиночному булеву ключу (v7 выходит
    # по конъюнкции, флаг сам по себе не решение). Репортятся только как справка.
    upper_signal_b2b: int = 0
    upper_legflip: int = 0
    exact_obs_conflict: int = 0     # ВАКУУМНО ~0 (все in-pos наблюдения уникальны)
    exact_is_vacuous: bool = True
    tp_thresholds: Dict[str, float] = field(default_factory=dict)
    per_pos_tag_tp: Dict[str, int] = field(default_factory=dict)
    max_recall_exit: float = 1.0    # потолок recall выхода при данном k
    k_allowed: int = 0              # допустимое число промахов (floor 99.5%)
    verdict: str = ""

    def describe(self) -> dict:
        """Сериализуемая сводка для JSON-отчёта."""
        return {k: (int(v) if isinstance(v, (np.integer,)) else v)
                for k, v in self.__dict__.items()}


def run_epoch(df: pd.DataFrame, epoch: str) -> ExitCollisionReport:
    """Коллизионный тест выхода на датасете одной эпохи."""
    rep = ExitCollisionReport(epoch=epoch)
    in_pos = df["a_in_position"].to_numpy() >= 0.5
    is_exit = df["is_flip_exit"].to_numpy().astype(bool)
    stay = df["bc_action"].to_numpy() == 0
    stay_in_pos = in_pos & stay

    rep.n_in_position = int(in_pos.sum())
    rep.n_stay_in_pos = int(stay_in_pos.sum())
    rep.n_flip_exit = int(is_exit.sum())

    exit_flag = df["m_exit_flag"].to_numpy() >= 0.5
    leg_dn = df["m_leg_dn"].to_numpy() >= 0.5
    unreal = df["a_unreal_pnl"].to_numpy().astype(float)
    pos_tag = df["pos_tag"].to_numpy().astype(str)
    reason = df["exit_reason"].to_numpy().astype(str)

    # Грубые верхние границы по одиночному булеву ключу (НЕ k — справка).
    rep.upper_signal_b2b = int(np.count_nonzero(stay_in_pos & exit_flag))
    rep.upper_legflip = int(np.count_nonzero(stay_in_pos & leg_dn))

    # Точный конфликт наблюдений (буквальное определение pre-reg, ВАКУУМНО ~0).
    obs_cols = [c for c in df.columns if c.startswith(("m_", "a_", "w_", "rule_"))]
    sub = df.loc[stay_in_pos | is_exit, obs_cols].to_numpy(np.float32)
    lab = df.loc[stay_in_pos | is_exit, "bc_action"].to_numpy()
    seen: Dict[int, int] = {}
    conflict = 0
    for r, y in zip(sub, lab):
        h = hash(r.tobytes())
        if h in seen and seen[h] != int(y):
            conflict += 1
        else:
            seen[h] = int(y)
    rep.exact_obs_conflict = conflict
    rep.exact_is_vacuous = (len(seen) == sub.shape[0])

    # --- tp (ЕДИНСТВЕННЫЙ содержательный тест): per pos_tag порог ---
    tp_exit = is_exit & (reason == "tp")
    k_tp_total = 0
    for tag in ("dip", "transition"):
        tag_tp = tp_exit & (pos_tag == tag)
        if not tag_tp.any():
            continue
        thr = float(unreal[tag_tp].min())
        rep.tp_thresholds[tag] = thr
        hit = stay_in_pos & (pos_tag == tag) & (unreal >= thr)
        rep.per_pos_tag_tp[tag] = int(hit.sum())
        k_tp_total += int(hit.sum())
    rep.k_tp = k_tp_total

    # Потолок recall выхода и допуск — по СВЯЗЫВАЮЩЕМУ k = k_tp.
    k = rep.k_tp
    rep.max_recall_exit = 1.0 - k / rep.n_flip_exit if rep.n_flip_exit else 1.0
    rep.k_allowed = int(np.floor(rep.n_flip_exit * (1.0 - EXIT_RECALL_FLOOR)))
    if k == 0:
        rep.verdict = "A=0 на выходе разрешён (0 коллизий по семантическим ключам)"
    elif rep.max_recall_exit >= EXIT_RECALL_FLOOR:
        rep.verdict = (f"полоса k={k}: recall выхода потолок "
                       f"{rep.max_recall_exit:.4f} >= 0.995 → A<=k допустим")
    else:
        rep.verdict = (f"ПРОВАЛ: k={k} → recall выхода потолок "
                       f"{rep.max_recall_exit:.4f} < 0.995, гейт непроходим")
    return rep


def _print(rep: ExitCollisionReport) -> None:
    print(f"[{rep.epoch}] в позиции={rep.n_in_position:,} "
          f"(STAY={rep.n_stay_in_pos:,}, FLIP-выходов={rep.n_flip_exit})")
    print(f"  [справка, НЕ k] верхняя граница signal/b2b(m_exit_flag) = {rep.upper_signal_b2b}")
    print(f"  [справка, НЕ k] верхняя граница legflip(m_leg_dn)       = {rep.upper_legflip}")
    print(f"  [справка] точный конфликт наблюдений = {rep.exact_obs_conflict} "
          f"(вакуумно={rep.exact_is_vacuous})")
    print(f"  k_tp (СВЯЗЫВАЮЩИЙ, unreal>=порог) = {rep.k_tp}  пороги={rep.tp_thresholds} "
          f"по pos_tag={rep.per_pos_tag_tp}")
    print(f"  потолок recall выхода = {rep.max_recall_exit:.5f}; "
          f"допуск k_allowed(99.5%) = {rep.k_allowed}")
    print(f"  ВЕРДИКТ: {rep.verdict}\n")


def main() -> None:
    """CLI: прогнать коллизионный тест выхода по обеим эпохам, вывести k."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/cubecloud/Data/rlbinancetrader")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base = Path(args.data)
    reports: List[ExitCollisionReport] = []
    for epoch in ("2024", "2021"):
        df = pd.read_parquet(base / f"bc_clone_v7_{epoch}.parquet")
        rep = run_epoch(df, epoch)
        _print(rep)
        reports.append(rep)
    if args.out:
        Path(args.out).write_text(json.dumps(
            {r.epoch: r.describe() for r in reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
