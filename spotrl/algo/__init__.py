"""algo/ — надстройки над PPO: per-head loss-masking и своя дистрибуция.

Чего здесь НЕТ: знания о конкретной стратегии и о рынке.
Каркас: сейчас реализована только численная часть маскирования голов
(head_masking.py). Наследник PPO и torch-дистрибуция — следующий этап
(PLAN v0.5, 3.7.3).
"""
from __future__ import annotations

from spotrl.algo.head_masking import head_loss_mask, masked_entropy, masked_log_prob

__all__ = ["head_loss_mask", "masked_log_prob", "masked_entropy"]
