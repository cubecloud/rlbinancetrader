"""Гейт копирования BC под ПРИНЯТЫЕ пороги (ШАГ 3, редакция 2026-07-27).

Две РАЗНЫЕ оценки (не смешивать):

(A) TEACHER-FORCED per-decision FPR/recall — ГЛАВНЫЙ go/no-go. Все записанные
    наблюдения (под экспертной траекторией) прогоняются через политику ОДИН раз →
    d=logit_FLIP−logit_STAY. Затем сэмплирование ≥100 сидами: Bernoulli(sigmoid(
    d+s*)). Знаменатель FPR = записанные STAY-бары; recall — раздельно вход/выход.
    Почему teacher-forced: в closed-loop, как только клон спонтанно флипнул,
    экспертный бар выхода в его траектории не наступает → recall на конкретных
    барах не определён. Плюс предпроверка ЁМКОСТИ (argmax при пороге s*): все
    сделки бар-в-бар.

(B) CLOSED-LOOP — доля чистых прогонов + |Δкомпаунд| (ДИАГНОСТИКА, не go/no-go).
    Действие клона идёт в env.step, среда пересчитывает agent-state obs из своей
    книги. Ложный FLIP меняет САМ набор сделок — только так осмысленны
    воспроизведение и компаунд. Записанные obs для (B) НЕПРИГОДНЫ.

Run:
  python -m spotrl.bc.eval_gate --data /home/cubecloud/Data/rlbinancetrader \
      --model /home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy --seeds 100
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

from spotrl.bc.train_clone import apply_scaler, head0_logits, load_pooled, obs_columns

FPR_TARGET = 3e-7
RECALL_FLOOR = 0.995
# v7 эталонный компаунд (pre-reg §3, judge_trades_v7).
V7_COMPOUND = {"2024": 3.6811, "2021": 1.9842}
V7_NTRADES = {"2024": 162, "2021": 272}
FEE_SIDE = 0.001


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


@dataclass
class TeacherForcedResult:
    """Результат teacher-forced гейта одной эпохи (главный go/no-go)."""

    epoch: str
    n_stay: int = 0
    n_entry: int = 0
    n_exit: int = 0
    gap_nat: float = 0.0
    # сэмплированные (среднее по сидам) + анти-вырожденный минимум.
    fpr_mean: float = 0.0
    fpr_max: float = 0.0
    recall_entry_mean: float = 0.0
    recall_exit_mean: float = 0.0
    recall_entry_min: float = 0.0
    recall_exit_min: float = 0.0
    # argmax-предпроверка ёмкости (при пороге s*): все сделки бар-в-бар.
    argmax_entry: int = 0
    argmax_exit: int = 0
    argmax_false_flip: int = 0
    passed: bool = False

    def describe(self) -> dict:
        """Сериализуемая сводка для JSON-отчёта."""
        return {k: (float(v) if isinstance(v, np.floating) else v)
                for k, v in self.__dict__.items()}


def teacher_forced(policy, data_dir: str, s_star: float, mu, sd,
                   seeds: int) -> List[TeacherForcedResult]:
    """Главный go/no-go: сэмплированные FPR/recall per-epoch + argmax-предпроверка."""
    X_raw, y, w, meta = load_pooled(data_dir)
    X = apply_scaler(X_raw, mu, sd)
    d_all = head0_logits(policy, X)
    d = (d_all[:, 1] - d_all[:, 0]).astype(np.float64)
    out = []
    for epoch, m in meta["per_epoch"].items():
        di = d[m["idx"]]
        stay = di[m["is_stay"]]
        entry = di[m["is_entry"]]
        exit_ = di[m["is_exit"]]
        r = TeacherForcedResult(epoch=epoch, n_stay=len(stay),
                                n_entry=len(entry), n_exit=len(exit_))
        flip_all = np.concatenate([entry, exit_])
        r.gap_nat = float(flip_all.min() - stay.max())
        p_stay = _sig(stay + s_star)
        p_entry = _sig(entry + s_star)
        p_exit = _sig(exit_ + s_star)
        rng = np.random.default_rng(0)
        fprs, re, rx = [], [], []
        for _ in range(seeds):
            fprs.append(rng.random(len(p_stay)).__lt__(p_stay).mean())
            re.append(rng.random(len(p_entry)).__lt__(p_entry).mean())
            rx.append(rng.random(len(p_exit)).__lt__(p_exit).mean())
        r.fpr_mean = float(np.mean(fprs)); r.fpr_max = float(np.max(fprs))
        r.recall_entry_mean = float(np.mean(re)); r.recall_entry_min = float(np.min(re))
        r.recall_exit_mean = float(np.mean(rx)); r.recall_exit_min = float(np.min(rx))
        # argmax при пороге s*: FLIP ⇔ d + s* > 0.
        r.argmax_entry = int((entry + s_star > 0).sum())
        r.argmax_exit = int((exit_ + s_star > 0).sum())
        r.argmax_false_flip = int((stay + s_star > 0).sum())
        r.passed = (r.fpr_max <= FPR_TARGET
                    and r.recall_entry_min >= RECALL_FLOOR
                    and r.recall_exit_min >= RECALL_FLOOR
                    and r.recall_entry_min > 0 and r.recall_exit_min > 0)
        out.append(r)
    return out


def _print_tf(r: TeacherForcedResult) -> None:
    print(f"[{r.epoch}] TEACHER-FORCED (STAY={r.n_stay:,} вход={r.n_entry} выход={r.n_exit})")
    print(f"  зазор d = {r.gap_nat:.2f} nat (нужно >= 20.31)")
    print(f"  FPR: mean={r.fpr_mean:.3e} max={r.fpr_max:.3e} (порог {FPR_TARGET:.1e})")
    print(f"  recall вход:  mean={r.recall_entry_mean:.5f} min={r.recall_entry_min:.5f}")
    print(f"  recall выход: mean={r.recall_exit_mean:.5f} min={r.recall_exit_min:.5f}")
    print(f"  argmax@s*: вход={r.argmax_entry}/{r.n_entry} выход={r.argmax_exit}/{r.n_exit} "
          f"ложных_FLIP={r.argmax_false_flip}")
    print(f"  ПРОШЁЛ: {r.passed}")


def main() -> None:
    """CLI: прогнать гейт (teacher-forced + опц. closed-loop) и вынести вердикт."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/cubecloud/Data/rlbinancetrader")
    ap.add_argument("--model", default="/home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--closed-loop", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    mf = json.loads(Path(args.model + ".manifest.json").read_text())
    mu = np.array(mf["scaler_mu"], np.float32)
    sd = np.array(mf["scaler_sd"], np.float32)
    s_star = float(mf["shift"]["s_star"]) if mf["shift"]["feasible"] else 0.0
    print(f"s* = {s_star} (feasible={mf['shift']['feasible']}, "
          f"gap_train={mf['shift']['gap_nat']:.2f})")
    model = PPO.load(args.model, device="cpu")

    tf = teacher_forced(model.policy, args.data, s_star, mu, sd, args.seeds)
    for r in tf:
        _print_tf(r)

    result = {"s_star": s_star, "seeds": args.seeds,
              "teacher_forced": {r.epoch: r.describe() for r in tf}}

    if args.closed_loop:
        from spotrl.bc.closed_loop import run_closed_loop
        cl = run_closed_loop(args.data, model.policy, s_star, mu, sd, args.seeds)
        result["closed_loop"] = cl
        for epoch, c in cl.items():
            print(f"[{epoch}] CLOSED-LOOP: чистых {c['clean_frac']:.3f} "
                  f"| residual min/med/max = {c['resid_min']:.4f}/"
                  f"{c['resid_med']:.4f}/{c['resid_max']:.4f}")

    verdict = all(r.passed for r in tf)
    result["verdict_clone_ready"] = verdict
    print(f"\nВЕРДИКТ КЛОН ГОТОВ (главный сэмплированный гейт): {verdict}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
