"""bridge/ — мост к движку sunday и parity-проверка.

Чего здесь НЕТ: ничего про обучение (PLAN v0.5, 1.5.1).
"""
from __future__ import annotations

from spotrl.bridge.v7parity import ParityResult, check_parity, run_v7

__all__ = ["run_v7", "check_parity", "ParityResult"]
