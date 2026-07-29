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

# Имена наблюдаемых столбцов, по которым определяется «бар под контролем агента»
# при ПЕРВОЙ СВОБОДЕ (exit_own, entry_own=False). Индексы берутся из obs_cols
# манифеста семени — предикат обязан быть ЧИСТОЙ функцией obs (RolloutBuffer.add
# не несёт infos, см. модульный docstring).
DIP_IN_POSITION_COLS = ("a_in_position", "a_pos_tag_dip")


def position_head_active_dip(obs_std: np.ndarray, mu: np.ndarray, sd: np.ndarray,
                             col_index: dict, min_hold_days: float = 0.0
                             ) -> np.ndarray:
    """Булева маска (batch,): бар, на котором СЭМПЛ головы 0 реально исполняется.

    ПЕРВАЯ СВОБОДА = ранний выход только на dip-in-position (v7hook: агентский
    выход honored лишь когда позиция открыта И это dip-сделка). На flat-барах,
    на transition-in-position, в cooldown/после SL вход/выход исполняет эксперт —
    сэмпл головы 0 на переход НЕ влияет, значит его нельзя кормить policy-градиентом
    (иначе градиент коррелирует с решением эксперта, а не головы; при >99% таких
    баров это топит сигнал dip-выходов).

    Предикат — чистая функция obs. Наблюдения в rollout СТАНДАРТИЗОВАНЫ, поэтому
    де-стандартизуем нужные столбцы (`raw = obs*sd + mu`) и режем `raw > 0.5`
    (столбцы бинарные: a_in_position, a_pos_tag_dip).

    Args:
        obs_std: стандартизованные наблюдения (batch, obs_dim).
        mu, sd: параметры скейлера семени (obs_dim,).
        col_index: {имя_столбца: индекс} из obs_cols манифеста семени.

    Returns:
        Булев массив (batch,): True — голова 0 под контролем агента (dip-in-pos).
    """
    obs_std = np.asarray(obs_std, dtype=np.float64)
    active = np.ones(obs_std.shape[0], dtype=bool)
    for name in DIP_IN_POSITION_COLS:
        j = col_index[name]
        raw = obs_std[:, j] * float(sd[j]) + float(mu[j])
        active &= raw > 0.5
    if min_hold_days > 0.0:
        # мин-холд (amend3 §1): не кредитуем dip-бары возраста < min_hold_days,
        # где агентский ранний выход запрещён (executed=STAY форсирован драйвером).
        # Гейт на ТОМ ЖЕ obs-признаке a_days_in_trade, что и в драйвере → маска и
        # исполнение согласованы бит-в-бит, on-policy корректность сохранена.
        j = col_index["a_days_in_trade"]
        age = obs_std[:, j] * float(sd[j]) + float(mu[j])
        active &= age >= min_hold_days
    return active


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


def first_freedom_head_mask(obs_std: np.ndarray, mu: np.ndarray, sd: np.ndarray,
                            col_index: dict, params_own: bool = False,
                            min_hold_days: float = 0.0) -> np.ndarray:
    """Полная маска вклада голов в policy-лосс для ПЕРВОЙ СВОБОДЫ, форма (batch, 3).

    Голова 0 (position) активна ТОЛЬКО на dip-in-position барах
    (`position_head_active_dip`); головы SL/TP — как в `head_loss_mask`
    (при `params_own=False` всегда 0, при True — на баре входа). Вход определяется
    как переход flat→dip (a_in_position>0.5 и предыдущий бар flat) — но так как
    entry_own=False на первой свободе, SL/TP-головы всё равно замаскированы.
    """
    dip = position_head_active_dip(obs_std, mu, sd, col_index, min_hold_days)
    mask = np.zeros((dip.shape[0], N_HEADS), dtype=np.float32)
    mask[dip, HEAD_POSITION] = 1.0
    if params_own:
        entry = dip & ~np.concatenate(([False], dip[:-1]))
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
