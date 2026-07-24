"""Манифест прогона: всё, что влияет на результат, записано числом.

Требование PLAN v0.5, 1.5.1 (п.3): сид, окно данных, версия спецификации
действий, версия наблюдения, `world_version`, хэш скейлера и версия движка
попадают в манифест. Любое число сравнивается только с числом той же версии
мира (4.5.2).
"""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from spotrl import __version__
from spotrl.config import EnvConfig


@dataclass(frozen=True)
class RunManifest:
    """Паспорт прогона — сериализуемый и сравнимый.

    Attributes:
        run_id: имя прогона.
        seed: сид генератора.
        env_config: параметры среды (правила мира, свободы, спецификации).
        data: описание источника данных.
        scaler_sha256: хэш артефакта скейлера (пусто, если скейлера нет).
        extra: произвольные дополнительные величины прогона.
    """

    run_id: str
    seed: int
    env_config: EnvConfig
    data: Dict[str, object] = field(default_factory=dict)
    scaler_sha256: str = ""
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Полное сериализуемое представление манифеста."""
        return {"run_id": self.run_id,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "spotrl_version": __version__,
                "python": platform.python_version(),
                "seed": self.seed,
                "env": self.env_config.describe(),
                "data": self.data,
                "scaler_sha256": self.scaler_sha256,
                "extra": self.extra}

    def save(self, path: str | Path) -> Path:
        """Записать манифест в json и вернуть путь."""
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                          encoding="utf-8")
        return target


def compare_world_version(left: RunManifest, right: RunManifest) -> Optional[str]:
    """Проверить сравнимость двух прогонов по версии мира.

    Returns:
        None, если версии совпадают; иначе текст расхождения — числа таких
        прогонов сравнивать нельзя (PLAN 4.5.2).
    """
    a = left.env_config.world.world_version
    b = right.env_config.world.world_version
    if a == b:
        return None
    return f"world_version различается: {a} vs {b} — числа несравнимы"
