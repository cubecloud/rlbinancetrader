"""data/ — загрузка состояний и предвычисленные массивы.

Чего здесь НЕТ: ничего про агента, сделки и признаки наблюдения
(PLAN v0.5, 1.5.1).
"""
from __future__ import annotations

from spotrl.data.state_dataset import (REQUIRED_COLUMNS, StateDataset,
                                       from_frame, load_state)
from spotrl.data.synthetic import make_synthetic_frame, make_synthetic_state

__all__ = ["StateDataset", "load_state", "from_frame", "REQUIRED_COLUMNS",
           "make_synthetic_frame", "make_synthetic_state"]
