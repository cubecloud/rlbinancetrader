"""features/ — сборка наблюдения и скейлер.

Чего здесь НЕТ: мутации состояния и обращений к будущим барам
(PLAN v0.5, 1.5.1 и тест Т2).
"""
from __future__ import annotations

from spotrl.features.builder import (AgentState, ObservationLayout, WorldState,
                                     build_observation, precompute_market_features,
                                     write_observation)
from spotrl.features.scaler import ObsScaler

__all__ = ["AgentState", "WorldState", "ObservationLayout", "build_observation",
           "write_observation", "precompute_market_features", "ObsScaler"]
