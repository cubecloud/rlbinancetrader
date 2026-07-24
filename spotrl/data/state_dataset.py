"""Загрузка каузальных state-таблиц в предвычисленные массивы.

Что здесь: чтение parquet, проверка обязательных колонок, выравнивание по
времени, выдача плотных numpy-массивов (приём из старого проекта: массивы,
а не DataFrame — на этом стоит замер 754k шаг/с сырой среды).

Чего здесь НЕТ: ничего про агента, сделки, награду и признаки наблюдения.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# Колонки каузального state от sunday (спецификация from_sunday_reply_5)
REQUIRED_COLUMNS: tuple[str, ...] = (
    "open", "high", "low", "close", "volume",
    "q_buy", "q_sell", "regime_code", "leg_dn",
)


@dataclass(frozen=True)
class StateDataset:
    """Плотное представление одной state-таблицы.

    Attributes:
        index: моменты времени баров (DatetimeIndex).
        ohlcv: массив (n, 5) — open, high, low, close, volume.
        signals: массив (n, 2) — q_buy, q_sell.
        regime_code: массив (n,) int32.
        leg_dn: массив (n,) bool.
        source: путь к исходному файлу (для манифеста прогона).
    """

    index: pd.DatetimeIndex
    ohlcv: np.ndarray
    signals: np.ndarray
    regime_code: np.ndarray
    leg_dn: np.ndarray
    source: str

    def __len__(self) -> int:
        """Число баров."""
        return len(self.index)

    @property
    def close(self) -> np.ndarray:
        """Цены закрытия."""
        return self.ohlcv[:, 3]

    @property
    def open(self) -> np.ndarray:
        """Цены открытия (исполнение решения бара t идёт по open(t+1))."""
        return self.ohlcv[:, 0]

    def describe(self) -> dict:
        """Сериализуемое описание для манифеста прогона."""
        return {"source": self.source, "n_bars": len(self),
                "start": str(self.index[0]), "end": str(self.index[-1])}


def load_state(path: str | Path, columns: Sequence[str] = REQUIRED_COLUMNS) -> StateDataset:
    """Прочитать parquet каузального state и превратить в массивы.

    Args:
        path: путь к parquet-файлу состояния.
        columns: обязательные колонки; их отсутствие — ошибка, а не молчание.

    Raises:
        KeyError: если в таблице нет обязательной колонки.
        ValueError: если индекс не монотонен по времени.
    """
    frame = pd.read_parquet(Path(path).expanduser())
    return from_frame(frame, source=str(path), columns=columns)


def from_frame(frame: pd.DataFrame, source: str = "<memory>",
               columns: Sequence[str] = REQUIRED_COLUMNS) -> StateDataset:
    """Собрать StateDataset из готового DataFrame (используется в тестах)."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"в state нет обязательных колонок: {missing}")
    index = pd.DatetimeIndex(frame.index)
    if not index.is_monotonic_increasing:
        raise ValueError("индекс state должен быть строго возрастающим по времени")
    ohlcv = np.column_stack([frame[c].to_numpy(dtype=np.float64)
                             for c in ("open", "high", "low", "close", "volume")])
    signals = np.column_stack([frame["q_buy"].to_numpy(dtype=np.float64),
                               frame["q_sell"].to_numpy(dtype=np.float64)])
    return StateDataset(index=index, ohlcv=ohlcv, signals=signals,
                        regime_code=frame["regime_code"].to_numpy().astype(np.int32),
                        leg_dn=frame["leg_dn"].to_numpy().astype(bool),
                        source=source)
