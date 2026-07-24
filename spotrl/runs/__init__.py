"""runs/ — манифесты прогонов, сиды, оркестрация.

Чего здесь НЕТ: бизнес-логики стратегии (PLAN v0.5, 1.5.1).
"""
from __future__ import annotations

from spotrl.runs.manifest import RunManifest, compare_world_version

__all__ = ["RunManifest", "compare_world_version"]
