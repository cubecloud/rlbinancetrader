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
        quote_volume: массив (n,) float — денежный объём бара (quote_asset_volume
            расширенного kline). None → NaN (рыночный строитель v2 деградирует к
            нейтрали).
        trades: массив (n,) float — число сделок в баре. None → NaN.
        taker_buy_base: массив (n,) float — базовый объём агрессивных покупок.
            None → NaN.
        buy_margin: массив (n,) float — непрерывная составляющая входа v7
            (q_buy − thr_buy). None → 0.0.
        sell_margin: массив (n,) float — непрерывная составляющая выхода v7. None → 0.0.
        bounce_pct: массив (n,) float — bounce эксперта v7. None → 0.0.
        leg_age: массив (n,) float — возраст ноги в часах (leg_age_h). None → 0.0.
        source: путь к исходному файлу (для манифеста прогона).

    Три булевых сигнала v7 нужны машине скрытого состояния среды: по ним на
    баре входа определяется pos_tag (dip vs transition) и отделяется штатный
    сигнальный выход. В каузальном state их нет — их выгружает из объекта v7
    `spotrl.data.dump_v7_signals` и подкладывает `attach_signals`. Если сигналы
    не приложены, поля = массивы False длины n (среда работает как без них).

    Торговые колонки (quote_volume/trades/taker_buy_base) и непрерывные
    составляющие v7 (buy_margin/sell_margin/bounce_pct/leg_age) нужны рыночному
    строителю наблюдения v2. base_volume = ohlcv[:, 4] (volume). Их отсутствие не
    ошибка — строитель v2 деградирует к задокументированной нейтрали.
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
    quote_volume: np.ndarray = None  # type: ignore[assignment]
    trades: np.ndarray = None  # type: ignore[assignment]
    taker_buy_base: np.ndarray = None  # type: ignore[assignment]
    buy_margin: np.ndarray = None  # type: ignore[assignment]
    sell_margin: np.ndarray = None  # type: ignore[assignment]
    bounce_pct: np.ndarray = None  # type: ignore[assignment]
    leg_age: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Заполнить отсутствующие булевы сигналы и рыночные колонки v2.

        Булевы сигналы v7 при отсутствии = массивы False длины n. Торговые
        колонки при отсутствии = NaN (строитель v2 деградирует к нейтрали, не
        придумывая объём). Непрерывные составляющие v7 при отсутствии = 0.0.
        """
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
        nan_defaults = ("quote_volume", "trades", "taker_buy_base")
        zero_defaults = ("buy_margin", "sell_margin", "bounce_pct", "leg_age")
        for name in nan_defaults + zero_defaults:
            val = getattr(self, name)
            fill = np.nan if name in nan_defaults else 0.0
            if val is None:
                object.__setattr__(self, name, np.full(n, fill, dtype=np.float64))
            else:
                arr = np.asarray(val, dtype=np.float64)
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

    @property
    def base_volume(self) -> np.ndarray:
        """Базовый объём бара = volume (5-й столбец ohlcv)."""
        return self.ohlcv[:, 4]

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

    def _opt_float(*names: str):
        """Прочитать float-колонку по первому найденному имени, иначе None.

        Имён несколько, потому что расширенный kline и снимок state зовут одну
        величину по-разному (quote_asset_volume vs quote_volume, leg_age_h vs
        leg_age).
        """
        for name in names:
            if name in frame.columns:
                return frame[name].to_numpy(dtype=np.float64)
        return None

    return StateDataset(index=index, ohlcv=ohlcv, signals=signals,
                        regime_code=frame["regime_code"].to_numpy().astype(np.int32),
                        leg_dn=frame["leg_dn"].to_numpy().astype(bool),
                        source=source,
                        entry_signal=_opt_bool("entry_signal"),
                        trans_entry_signal=_opt_bool("trans_entry_signal"),
                        exit_sig=_opt_bool("exit_sig"),
                        quote_volume=_opt_float("quote_volume", "quote_asset_volume"),
                        trades=_opt_float("trades"),
                        taker_buy_base=_opt_float("taker_buy_base"),
                        buy_margin=_opt_float("buy_margin"),
                        sell_margin=_opt_float("sell_margin"),
                        bounce_pct=_opt_float("bounce_pct"),
                        leg_age=_opt_float("leg_age", "leg_age_h"))


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
        exit_sig=sig["exit_sig"].to_numpy().astype(bool),
        quote_volume=dataset.quote_volume, trades=dataset.trades,
        taker_buy_base=dataset.taker_buy_base, buy_margin=dataset.buy_margin,
        sell_margin=dataset.sell_margin, bounce_pct=dataset.bounce_pct,
        leg_age=dataset.leg_age)
