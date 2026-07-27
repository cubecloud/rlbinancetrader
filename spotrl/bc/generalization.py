"""Кросс-эпоховая проверка ОБОБЩЕНИЯ клона (разводит «выучил» и «запомнил»).

Зачем (критично). Обучающий объектив клона (worst-case logsumefp-маржа) напрямую
максимизирует ровно ту величину `min_FLIP(d) − max_STAY(d)`, которую проверяет
главный гейт, НА ТЕХ ЖЕ барах. Поэтому in-sample прохождение гейта ТАВТОЛОГИЧНО:
объектив = метрика на одних данных. Большой in-sample зазор (наблюдалось 87 nat
при требуемых 20.3) может быть подписью МЕМОИЗАЦИИ по шумовым признакам (ценовые/
объёмные оси в 38-мерном obs), а не выученного правила v7.

Правило v7 идентично в обеих эпохах (те же булевы сигналы, те же TP-константы
per pos_tag). Значит клон, выучивший ПРАВИЛО, обязан дать зазор >= 20.3 nat и ВНЕ
своей эпохи; клон, запомнивший бары, — развалится (зазор → минус, min_FLIP уходит
ниже max_STAY out-of-epoch). Эта проверка — go/no-go для вердикта «готов к RL».
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from spotrl.bc.train_clone import (apply_scaler, build_policy, fit_scaler,
                                    head0_logits, load_pooled, train)

GAP_NEEDED = 20.31  # logit(0.995) − logit(3e-7)


@dataclass
class CrossEpochResult:
    """Зазор клона, обученного на train_epoch, замеренный in- и out-of-epoch."""

    train_epoch: str
    test_epoch: str
    gap_in: float
    gap_out: float
    min_flip_out: float
    max_stay_out: float
    generalizes: bool

    def describe(self) -> dict:
        """Сериализуемая сводка."""
        return {k: (float(v) if isinstance(v, np.floating) else v)
                for k, v in self.__dict__.items()}


def _gap(policy, Xn, meta, epoch):
    """min_FLIP(d) − max_STAY(d) на срезе одной эпохи."""
    m = meta["per_epoch"][epoch]
    lg = head0_logits(policy, Xn[m["idx"]])
    d = lg[:, 1] - lg[:, 0]
    fa = d[m["is_entry"] | m["is_exit"]]
    st = d[m["is_stay"]]
    return float(fa.min() - st.max()), float(fa.min()), float(st.max())


def run_cross_epoch(data_dir: str, seed: int = 0,
                    n_epochs: int = 25) -> List[CrossEpochResult]:
    """Обучить на КАЖДОЙ эпохе отдельно, замерить зазор на другой. Обе стороны."""
    X_raw, y, w, meta = load_pooled(data_dir)
    mu, sd = fit_scaler(X_raw)
    Xn = apply_scaler(X_raw, mu, sd)
    out: List[CrossEpochResult] = []
    for tr_ep, te_ep in (("2021", "2024"), ("2024", "2021")):
        idx = meta["per_epoch"][tr_ep]["idx"]
        _, policy = build_policy(Xn.shape[1], seed)
        train(policy, Xn[idx], y[idx], w[idx], seed, n_epochs=n_epochs)
        gin, _, _ = _gap(policy, Xn, meta, tr_ep)
        gout, mnf, mxs = _gap(policy, Xn, meta, te_ep)
        out.append(CrossEpochResult(
            train_epoch=tr_ep, test_epoch=te_ep, gap_in=gin, gap_out=gout,
            min_flip_out=mnf, max_stay_out=mxs, generalizes=(gout >= GAP_NEEDED)))
    return out


def main() -> None:
    """CLI: прогнать кросс-эпоховую проверку обобщения, вывести вердикт."""
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/cubecloud/Data/rlbinancetrader")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run_cross_epoch(args.data)
    for r in res:
        print(f"train={r.train_epoch}: IN gap={r.gap_in:.1f} | "
              f"OUT({r.test_epoch}) gap={r.gap_out:.1f} "
              f"(minFLIP={r.min_flip_out:.1f} maxSTAY={r.max_stay_out:.1f}) "
              f"обобщает={r.generalizes}")
    ok = all(r.generalizes for r in res)
    print(f"ОБОБЩЕНИЕ (зазор>= {GAP_NEEDED} обе стороны): {ok}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            [r.describe() for r in res], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
