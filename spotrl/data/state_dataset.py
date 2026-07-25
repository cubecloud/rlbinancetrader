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
        entry_signal: массив (n,) bool — булев сигнал входа v7 (self._entry).
        trans_entry_signal: массив (n,) bool — сигнал transition-входа v7
            (self._trans_entry).
        exit_sig: массив (n,) bool — штатный сигнал выхода v7 (self._exit).
        source: путь к исходному файлу (для манифеста прогона).

    Три булевых сигнала v7 нужны машине скрытого состояния среды: по ним на
    баре входа определяется pos_tag (dip vs transition) и отделяется штатный
    сигнальный выход. В каузальном state их нет — их выгружает из объекта v7
    `spotrl.data.dump_v7_signals` и подкладывает `attach_signals`. Если сигналы
    не приложены, поля = массивы False длины n (среда работает как без них).
    """

    index: pd.DatetimeIndex
    ohlcv: np.ndarray
    signals: np.ndarray
    regime_code: np.ndarray
    leg_dn: np.ndarray
    source: str
    entry_signal: np.ndarray = None  # type: ignore[assignment]
    trans_entry_signal: np.ndarray = None  # type: ignore[assignment]
    exit_sig: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Заполнить отсутствующие булевы сигналы массивами False длины n."""
        n = len(self.index)
        for name in ("entry_signal", "trans_entry_signal", "exit_sig"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.zeros(n, dtype=bool))
            else:
                arr = np.asarray(getattr(self, name)).astype(bool)
                if len(arr) != n:
                    raise ValueError(
                        f"{name}: длина {len(arr)} != числу баров {n}")
                object.__setattr__(self, name, arr)

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

    def _opt_bool(name: str):
        """Прочитать булев сигнал из кадра, если колонка есть, иначе None."""
        return frame[name].to_numpy().astype(bool) if name in frame.columns else None

    return StateDataset(index=index, ohlcv=ohlcv, signals=signals,
                        regime_code=frame["regime_code"].to_numpy().astype(np.int32),
                        leg_dn=frame["leg_dn"].to_numpy().astype(bool),
                        source=source,
                        entry_signal=_opt_bool("entry_signal"),
                        trans_entry_signal=_opt_bool("trans_entry_signal"),
                        exit_sig=_opt_bool("exit_sig"))


def attach_signals(dataset: StateDataset, signals_path: str | Path) -> StateDataset:
    """Приложить булевы сигналы v7 из артефакта к готовому StateDataset.

    Args:
        dataset: набор, загруженный из каузального state.
        signals_path: parquet-артефакт `dump_v7_signals` (колонки
            entry_signal, trans_entry_signal, exit_sig; индекс = индекс state).

    Returns:
        Новый StateDataset с приложенными булевыми сигналами. Индекс артефакта
        обязан совпасть с индексом набора бар-в-бар (иначе train/serve
        разъедется — тот же мастер-инвариант «один прогон»).

    Raises:
        ValueError: если индекс артефакта не совпадает с индексом набора.
    """
    sig = pd.read_parquet(Path(signals_path).expanduser())
    if len(sig) != len(dataset) or not sig.index.equals(dataset.index):
        raise ValueError("индекс артефакта сигналов не совпадает с индексом state")
    return StateDataset(
        index=dataset.index, ohlcv=dataset.ohlcv, signals=dataset.signals,
        regime_code=dataset.regime_code, leg_dn=dataset.leg_dn,
        source=dataset.source,
        entry_signal=sig["entry_signal"].to_numpy().astype(bool),
        trans_entry_signal=sig["trans_entry_signal"].to_numpy().astype(bool),
        exit_sig=sig["exit_sig"].to_numpy().astype(bool))
