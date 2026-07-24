"""spec/ — версионированные спецификации действий и наблюдения.

Здесь только описания и константы. Ничего вычислительного, никаких
обращений к данным и к среде (PLAN v0.5, 1.5.1).
"""
from __future__ import annotations

from spotrl.spec.actions import (ActionSpec, DecodedAction, FLIP, HEAD_NAMES,
                                 HEAD_POSITION, HEAD_SL, HEAD_TP, N_HEADS,
                                 N_POSITION_ACTIONS, POSITION_ACTION_NAMES, STAY)
from spotrl.spec.observation import (ObservationSpec, RESERVED_SLOTS,
                                     RESERVED_SLOT_VALUE)

__all__ = ["ActionSpec", "DecodedAction", "STAY", "FLIP", "N_POSITION_ACTIONS",
           "POSITION_ACTION_NAMES", "HEAD_POSITION", "HEAD_SL", "HEAD_TP",
           "HEAD_NAMES", "N_HEADS", "ObservationSpec", "RESERVED_SLOTS",
           "RESERVED_SLOT_VALUE"]
