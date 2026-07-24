"""Скейлер наблюдения: fit только на train, артефакт с хэшем (PLAN 1.3.2).

Требование: в backtest и в live применяется БАЙТ-В-БАЙТ один и тот же
артефакт. Поэтому у скейлера есть хэш параметров, который попадает в
манифест прогона; несовпадение хэша при serve — ошибка, а не предупреждение.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ObsScaler:
    """Покомпонентная нормализация наблюдения (z-score) с хэшем артефакта.

    Attributes:
        mean: среднее по каждому признаку, посчитанное ТОЛЬКО на train.
        scale: масштаб по каждому признаку (std, ноль заменён на 1).
        version: версия артефакта.
    """

    mean: np.ndarray
    scale: np.ndarray
    version: str = "v1"

    @classmethod
    def fit(cls, observations: np.ndarray, version: str = "v1") -> "ObsScaler":
        """Обучить скейлер на массиве наблюдений train-периода (n, d)."""
        arr = np.asarray(observations, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2:
            raise ValueError("нужен массив (n, d) с n >= 2")
        mean = arr.mean(axis=0)
        scale = arr.std(axis=0)
        scale[scale < 1e-12] = 1.0
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32),
                   version=version)

    def transform(self, obs: np.ndarray) -> np.ndarray:
        """Применить нормализацию; вход не изменяется."""
        arr = np.asarray(obs, dtype=np.float32)
        if arr.shape[-1] != self.mean.shape[0]:
            raise ValueError(f"размерность {arr.shape[-1]} != {self.mean.shape[0]}")
        return ((arr - self.mean) / self.scale).astype(np.float32)

    @property
    def sha256(self) -> str:
        """Хэш артефакта: попадает в манифест и сверяется при serve."""
        digest = hashlib.sha256()
        digest.update(self.version.encode("utf-8"))
        digest.update(np.ascontiguousarray(self.mean, dtype=np.float32).tobytes())
        digest.update(np.ascontiguousarray(self.scale, dtype=np.float32).tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path) -> str:
        """Сохранить артефакт в json; возвращает его хэш."""
        payload = {"version": self.version,
                   "mean": self.mean.tolist(),
                   "scale": self.scale.tolist(),
                   "sha256": self.sha256}
        Path(path).expanduser().write_text(json.dumps(payload), encoding="utf-8")
        return payload["sha256"]

    @classmethod
    def load(cls, path: str | Path) -> "ObsScaler":
        """Загрузить артефакт и проверить его хэш."""
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        scaler = cls(mean=np.asarray(payload["mean"], dtype=np.float32),
                     scale=np.asarray(payload["scale"], dtype=np.float32),
                     version=payload["version"])
        if scaler.sha256 != payload["sha256"]:
            raise ValueError("хэш артефакта скейлера не совпал — файл изменён")
        return scaler
