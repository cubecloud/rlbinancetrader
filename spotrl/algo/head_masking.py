"""Loss-masking по головам: чем в v0.5 заменены маски действий.

Зачем: головы SL и TP содержательны только на баре входа (162 бара за
проход). На всех прочих барах их вклад в policy-лосс должен быть ровно
нулевым, иначе сеть учится на действиях, которые ни на что не влияли.

Почему нельзя обойтись штатной sb3: `MultiCategoricalDistribution.log_prob`
и `.entropy` в sb3 2.3.2 возвращают уже СУММУ по головам
(`stable_baselines3/common/distributions.py`), то есть на выходе нет
разбивки, которую можно взвесить. Поэтому нужны свои функции по головам —
здесь их численная часть, без зависимости от sb3.

Маска пересчитывается из (obs, action): `RolloutBuffer.add` не принимает
`infos`, пронести её через буфер нельзя (PLAN 3.7.2).
"""
from __future__ import annotations

import numpy as np

from spotrl.spec.actions import HEAD_POSITION, HEAD_SL, HEAD_TP, N_HEADS


def head_loss_mask(is_entry_bar: np.ndarray, params_own: bool) -> np.ndarray:
    """Маска вклада голов в policy-лосс, форма (batch, 3).

    Args:
        is_entry_bar: булев массив (batch,) — бар, на котором открывается позиция.
        params_own: включена ли свобода выбора SL/TP (шаг лестницы L4п).

    Returns:
        float32-массив (batch, 3): 1.0 — голова участвует в лоссе, 0.0 — нет.
        Голова позиции участвует всегда; головы SL/TP — только на барах входа
        и только при `params_own=True`.
    """
    entry = np.asarray(is_entry_bar, dtype=bool).reshape(-1)
    mask = np.zeros((entry.shape[0], N_HEADS), dtype=np.float32)
    mask[:, HEAD_POSITION] = 1.0
    if params_own:
        mask[entry, HEAD_SL] = 1.0
        mask[entry, HEAD_TP] = 1.0
    return mask


def masked_log_prob(per_head_log_prob: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Сумма log-prob по головам с учётом маски, форма (batch,).

    Замаскированная голова даёт ровно 0 вклада, а значит ровно нулевой
    градиент на своих логитах (проверяется тестом Т3).
    """
    lp = np.asarray(per_head_log_prob, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    if lp.shape != m.shape:
        raise ValueError(f"формы не совпали: {lp.shape} vs {m.shape}")
    return (lp * m).sum(axis=1)


def masked_entropy(per_head_entropy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Энтропия по головам с учётом маски, форма (batch,)."""
    return masked_log_prob(per_head_entropy, mask)
