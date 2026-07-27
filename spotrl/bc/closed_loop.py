"""Closed-loop диагностика клона: доля ЧИСТЫХ прогонов + |Δкомпаунд| (не go/no-go).

Ключевое наблюдение (делает диагностику дешёвой и ТОЧНОЙ): пока клон на каждом
баре сэмплирует ровно экспертное действие, траектория среды СОВПАДАЕТ с экспертной
→ наблюдение на каждом баре = записанному obs, а его d уже посчитан teacher-forced
проходом. «Чистый прогон» = НОЛЬ отклонений от экспертного действия по ВСЕМ барам
(любое отклонение = ложный FLIP на STAY или пропуск FLIP → другой набор сделок,
прогон грязный). Поэтому доля чистых = доля сидов, где сэмпл совпал с экспертом на
всех барах — считается сэмплированием Bernoulli(sigmoid(d+s*)) на предвычисленном
массиве d, без пошагового прогона среды (124M шагов нереальны).

Компаунд ЧИСТОГО прогона = экспертный компаунд под средой (клон = эксперт бар-в-бар)
→ residual = валидированное pre-reg §3 значение (2024: 3.04e-4, 2021: 2.72e-4,
измерено на самом эксперте). Для грязных прогонов набор сделок иной (репортим число
лишних/недостающих сделок как справку). Эта функция НЕ go/no-go (§0б pre-reg).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from spotrl.bc.train_clone import apply_scaler, head0_logits, load_pooled

# residual чистого прогона = валидированный на эксперте (pre-reg §3, таблица).
VALIDATED_CLEAN_RESIDUAL = {"2024": 0.000304, "2021": 0.000272}


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def run_closed_loop(data_dir: str, policy, s_star: float, mu, sd,
                    seeds: int) -> Dict[str, dict]:
    """Доля чистых прогонов + компаунд-диагностика per-epoch (сэмплирование ≥seeds).

    Чистый прогон = на всех барах эпохи сэмплированное действие == экспертному.
    p(correct) = p(FLIP) на FLIP-барах, 1−p(FLIP) на STAY-барах.
    """
    X_raw, y, w, meta = load_pooled(data_dir)
    X = apply_scaler(X_raw, mu, sd)
    d_all = head0_logits(policy, X)
    d = (d_all[:, 1] - d_all[:, 0]).astype(np.float64)
    p_flip = _sig(d + s_star)

    result = {}
    for epoch, m in meta["per_epoch"].items():
        idx = m["idx"]
        is_flip = (m["is_entry"] | m["is_exit"])
        pf = p_flip[idx]
        # вероятность правильного действия на каждом баре эпохи.
        p_correct = np.where(is_flip, pf, 1.0 - pf)
        rng = np.random.default_rng(2000)
        clean = 0
        extra_trades = []
        for _ in range(seeds):
            draw = rng.random(len(p_correct))
            correct = draw < p_correct
            deviations = int((~correct).sum())
            if deviations == 0:
                clean += 1
            # ложный FLIP на STAY (лишняя сделка) минус пропуск FLIP.
            false_flip = int(((~correct) & (~is_flip)).sum())
            extra_trades.append(false_flip)
        # аналитическая доля чистых для сверки (∏ p_correct).
        log_clean = float(np.log(np.clip(p_correct, 1e-300, 1.0)).sum())
        analytic_clean = float(np.exp(log_clean))
        result[epoch] = {
            "seeds": seeds,
            "clean_runs": clean,
            "clean_frac": clean / seeds,
            "clean_frac_analytic": analytic_clean,
            "false_flip_per_run_max": int(max(extra_trades)),
            "resid_min": VALIDATED_CLEAN_RESIDUAL[epoch],
            "resid_med": VALIDATED_CLEAN_RESIDUAL[epoch],
            "resid_max": VALIDATED_CLEAN_RESIDUAL[epoch],
            "resid_note": ("чистый прогон = эксперт бар-в-бар → residual = "
                           "валидированное pre-reg §3 значение; грязные прогоны "
                           "имеют иной набор сделок (см. false_flip_per_run_max)"),
        }
    return result
