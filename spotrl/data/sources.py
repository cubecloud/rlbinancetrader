"""Единый интерфейс к данным: parquet-снимок и база PostgreSQL.

Два источника за одним интерфейсом :class:`DataSource` отдают ОДИНАКОВЫЙ
плотный numpy-результат :class:`WindowData` на окне ``[start, end]``.

Ключевая идея (уточнение пользователя 2026-07-25): набор колонок — ПАРАМЕТР
источника, а не хардкод. База содержит все нужные СЫРЫЕ колонки; их надо просто
запросить. Производные признаки стратегии v7 (q_buy/q_sell/regime_code/leg_dn) —
НЕ сырьё, их в базе нет и не должно быть: они нужны только на этапе копирования
v7 и приходят из parquet-снимка.

* :class:`PgSource` — база через ``dbbinance-storage``. По умолчанию отдаёт
  расширенный набор Binance-kline (OHLCV + поток ордеров) через
  ``use_extended_cols=True``; можно задать явный ``columns``. Импорт
  ``dbbinance`` — ЛЕНИВЫЙ, ключи PG грузятся из ``sunday/*.env`` перед импортом,
  чтобы ``secureapikey`` не требовал интерактивного SALT-ввода (headless).
* :class:`ParquetSource` — обёртка над снимком: отдаёт любые колонки, что в нём
  есть (включая v7-производные). Не импортирует ``dbbinance``.

Гейт эквивалентности :func:`bitwise_gate` сравнивает ПЕРЕСЕЧЕНИЕ колонок двух
источников по имени, побитово (порог — ноль различающихся элементов).
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from spotrl.data.state_dataset import StateDataset

# Параметры чтения OHLCV, ТОЧНО как в билдере снимка
# (sunday/tooling/audit/state_v0_builder.py:79-85). Любое расхождение здесь
# ломает побитовую эквивалентность, поэтому значения зафиксированы.
PG_TABLE = "spot_data_btcusdt_1m"
PG_TIMEFRAME = "1m"
PG_ORIGIN = "start"

# Каталог с *.env-ключами PostgreSQL (проект sunday). Значения ключей НИКОГДА не
# логируются. Путь переопределяется параметром PgSource или $SPOTRL_PG_ENV_DIR.
DEFAULT_PG_ENV_DIR = "~/Python/projects/sunday"
PG_ENV_FILES = ("PSGSQL_KEY.env", "PSGSQLKEYS.env")

# Наборы колонок.
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
# Расширенный kline из базы (Constants.binance_extended_cols без open_time):
# OHLCV + поток ордеров / микроструктура. Это то, что реально есть в PG.
PG_EXTENDED_COLUMNS = OHLCV_COLUMNS + (
    "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote")
# Производные признаки стратегии v7 (только в снимке, в сырой базе их НЕТ).
V7_FEATURE_COLUMNS = ("q_buy", "q_sell", "regime_code", "leg_dn")
# Полный state-набор снимка sunday (нужен пути копирования v7 и StateDataset).
SUNDAY_STATE_COLUMNS = OHLCV_COLUMNS + V7_FEATURE_COLUMNS

# Идентификатор билдера признаков снимка. У билдера нет __version__; фиксируем
# строкой — попадает в манифест прогона (требование 4).
FEATURE_BUILDER_VERSION = "state_v0"


@dataclass(frozen=True)
class WindowData:
    """Плотный результат чтения окна одним источником — именованная матрица.

    Индекс — tz-naive UTC (как в снимке: билдер делает
    ``tz_convert("UTC").tz_localize(None)``). Колонки хранятся по именам; форма
    источника (sunday vs сырьё) не зашита в поля.

    Attributes:
        index: моменты баров, tz-naive UTC, строго возрастающие.
        columns: имена колонок в порядке столбцов ``data``.
        data: массив (n, k) float64 — значения колонок.
        source: человекочитаемое имя источника (для манифеста).
        meta: параметры чтения (таблица, окно, версии) для манифеста.
    """

    index: pd.DatetimeIndex
    columns: tuple
    data: np.ndarray
    source: str
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        """Число баров в окне."""
        return len(self.index)

    def has(self, name: str) -> bool:
        """Есть ли колонка ``name``."""
        return name in self.columns

    def column(self, name: str) -> np.ndarray:
        """Одна колонка по имени, массив (n,).

        Raises:
            KeyError: если колонки нет в источнике.
        """
        if name not in self.columns:
            raise KeyError(f"нет колонки {name!r}; есть: {self.columns}")
        return self.data[:, self.columns.index(name)]

    def select(self, names: Sequence[str]) -> np.ndarray:
        """Подматрица (n, len(names)) в порядке ``names``."""
        idx = [self.columns.index(n) for n in names]
        return self.data[:, idx]

    @property
    def ohlcv(self) -> np.ndarray:
        """Удобный срез (n, 5): open, high, low, close, volume."""
        return self.select(OHLCV_COLUMNS)

    def has_features(self) -> bool:
        """Есть ли ВСЕ производные признаки v7 (True только у полного снимка)."""
        return all(self.has(c) for c in V7_FEATURE_COLUMNS)

    def data_hash(self) -> str:
        """sha256 матрицы данных — для манифеста и быстрого сравнения окон."""
        return hashlib.sha256(np.ascontiguousarray(self.data)).hexdigest()

    def to_state_dataset(self) -> StateDataset:
        """Собрать :class:`StateDataset` для среды (путь копирования v7).

        Требует полный state-набор снимка (OHLCV + v7-признаки).

        Raises:
            ValueError: если производных признаков нет (сырой PG).
        """
        if not self.has_features():
            raise ValueError(
                "источник без производных признаков v7 "
                "(q_buy/q_sell/regime_code/leg_dn); StateDataset для среды "
                "строится только из снимка state_v0, а не из сырой базы")
        signals = np.column_stack([self.column("q_buy"), self.column("q_sell")])

        def _opt(*names: str):
            """Первая присутствующая колонка окна как float, иначе None."""
            for name in names:
                if self.has(name):
                    return self.column(name).astype(np.float64)
            return None

        return StateDataset(
            index=self.index, ohlcv=self.ohlcv, signals=signals,
            regime_code=self.column("regime_code").astype(np.int32),
            leg_dn=self.column("leg_dn").astype(bool), source=self.source,
            quote_volume=_opt("quote_volume", "quote_asset_volume"),
            trades=_opt("trades"), taker_buy_base=_opt("taker_buy_base"),
            buy_margin=_opt("buy_margin"), sell_margin=_opt("sell_margin"),
            bounce_pct=_opt("bounce_pct"), leg_age=_opt("leg_age", "leg_age_h"))


class DataSource(ABC):
    """Общий интерфейс двух источников данных."""

    name: str = "source"

    @abstractmethod
    def load_window(self, start, end) -> WindowData:
        """Прочитать окно ``[start, end]`` (границы включительно) в плотный вид.

        Args:
            start: левая граница окна (Timestamp/строка/datetime).
            end: правая граница окна (включительно).
        """


def _as_naive_utc(ts) -> pd.Timestamp:
    """Привести момент к tz-naive UTC (как индекс снимка)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


class ParquetSource(DataSource):
    """Источник поверх parquet-снимка (обучение, воспроизводимость).

    Args:
        path: путь к parquet каузального state (снимок билдера).
        columns: какие колонки отдавать; по умолчанию полный state-набор
            снимка (OHLCV + производные v7).
    """

    name = "parquet"

    def __init__(self, path: str | Path,
                 columns: Sequence[str] = SUNDAY_STATE_COLUMNS):
        """См. docstring класса."""
        self._path = str(Path(path).expanduser())
        self._columns = tuple(columns)
        self._frame: Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        if self._frame is None:
            frame = pd.read_parquet(self._path)
            idx = pd.DatetimeIndex(frame.index)
            if not idx.is_monotonic_increasing:
                raise ValueError("индекс снимка должен возрастать по времени")
            missing = [c for c in self._columns if c not in frame.columns]
            if missing:
                raise KeyError(f"в снимке нет запрошенных колонок: {missing}")
            self._frame = frame
        return self._frame

    def cache_key_fields(self) -> dict:
        """Идентифицирующие параметры источника для ключа кэша (без окна).

        Для снимка добавляем ``st_mtime``+``size`` файла: если снимок
        пересоберут по тому же ``path`` без смены версии билдера, ключ всё
        равно изменится и устаревший кэш промахнётся (закрытие «тихой дыры»
        ручной версии билдера).
        """
        import os
        try:
            stat = os.stat(self._path)
            file_sig = {"st_size": int(stat.st_size),
                        "st_mtime_ns": int(stat.st_mtime_ns)}
        except OSError:
            file_sig = {"st_size": None, "st_mtime_ns": None}
        return {"source_kind": "parquet", "path": self._path,
                "columns": list(self._columns), "timeframe": None,
                "origin": None, "extended": None, "last_full_bar": None,
                **file_sig}

    def load_window(self, start, end) -> WindowData:
        """Срез снимка на окне; отдаются запрошенные колонки."""
        frame = self._load()
        lo, hi = _as_naive_utc(start), _as_naive_utc(end)
        idx = pd.DatetimeIndex(frame.index)
        pos = np.flatnonzero((idx >= lo) & (idx <= hi))
        data = np.column_stack([
            frame[c].to_numpy(dtype=np.float64)[pos] for c in self._columns])
        meta = {"source_kind": "parquet", "path": self._path,
                "columns": list(self._columns), "window_start": str(lo),
                "window_end": str(hi), "n_bars": int(pos.size)}
        return WindowData(index=idx[pos], columns=self._columns, data=data,
                          source=self._path, meta=meta)


class PgSource(DataSource):
    """Источник поверх PostgreSQL через ``dbbinance-storage`` (paper/live).

    По умолчанию отдаёт расширенный kline (OHLCV + поток ордеров) через
    ``use_extended_cols=True``. Импорт ``dbbinance`` ленивый; ключи PG грузятся
    из ``sunday/*.env`` до импорта, поэтому интерактивного SALT-ввода нет.

    Args:
        table_name: имя таблицы 1m-баров в базе.
        timeframe: целевой таймфрейм ресемпла (снимок собран на ``1m``).
        columns: явный набор колонок; если задан — ``use_extended_cols=False``
            и запрашиваются именно эти колонки. Если ``None`` — берётся
            расширенный набор (:data:`PG_EXTENDED_COLUMNS`).
        fetcher: готовый DataFetcher (для тестов/инъекции); если ``None`` —
            создаётся лениво через ``dbbinance`` при первом чтении.
        env_dir: каталог с ``*.env``-ключами PG.
    """

    name = "pg"

    def __init__(self, table_name: str = PG_TABLE, timeframe: str = PG_TIMEFRAME,
                 columns: Optional[Sequence[str]] = None, fetcher=None,
                 env_dir: Optional[str] = None):
        """См. docstring класса."""
        import os
        self._table = table_name
        self._timeframe = timeframe
        self._extended = columns is None
        self._columns = (PG_EXTENDED_COLUMNS if columns is None
                         else tuple(columns))
        self._fetcher = fetcher
        self._injected = fetcher is not None
        self._env_dir = (env_dir or os.getenv("SPOTRL_PG_ENV_DIR")
                         or DEFAULT_PG_ENV_DIR)

    def cache_key_fields(self) -> dict:
        """Идентифицирующие параметры источника для ключа кэша (без окна).

        ``last_full_bar`` НЕ входит сюда — он параметр вызова ``load_window`` и
        добавляется в ключ кэшем отдельно (меняет набор баров). Версия данных
        базы аппроксимируется ``dbbinance_version()`` (единственный доступный
        прокси; задокументированное ограничение — ревизия строк в самой базе
        без смены версии пакета кэшем не отлавливается).
        """
        return {"source_kind": "pg", "table": self._table,
                "columns": list(self._columns), "timeframe": self._timeframe,
                "origin": PG_ORIGIN, "extended": bool(self._extended),
                "path": None}

    def _load_secrets(self) -> None:
        """Загрузить ключи PG из ``*.env`` в окружение ДО импорта dbbinance.

        Значения ключей не логируются. Отсутствие файлов не ошибка — тогда
        сработает штатный путь dbbinance (env/интерактив).
        """
        from dotenv import load_dotenv
        base = Path(self._env_dir).expanduser()
        for fname in PG_ENV_FILES:
            path = base / fname
            if path.exists():
                load_dotenv(str(path), override=False)

    def _get_fetcher(self):
        """Лениво получить DataFetcher; ключи и импорт dbbinance только здесь."""
        if self._fetcher is None:
            self._load_secrets()
            from dbbinance.fetcher.getfetcher import get_datafetcher
            self._fetcher = get_datafetcher()
        return self._fetcher

    def _fetch_raw(self, start, end, last_full_bar: bool) -> pd.DataFrame:
        """Ресемпл базы теми же параметрами, что и билдер снимка."""
        fetcher = self._get_fetcher()
        kwargs = dict(table_name=self._table,
                      start=_as_naive_utc(start).to_pydatetime(),
                      end=_as_naive_utc(end).to_pydatetime(),
                      to_timeframe=self._timeframe, origin=PG_ORIGIN,
                      open_time_index=True, last_full_bar=last_full_bar)
        if self._extended:
            # На реальном пути указываем extended-набор колонок базы.
            if not self._injected:
                kwargs.update(use_extended_cols=True)
        else:
            kwargs.update(use_cols=("open_time",) + self._columns)
        return fetcher.pg_resample_to_timeframe(**kwargs)

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Дедуп и tz-naive UTC индекс — точно как в билдере снимка."""
        df = df[~df.index.duplicated(keep="last")].sort_index()
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        df = df.copy()
        df.index = idx
        return df

    def load_window(self, start, end, last_full_bar: bool = True) -> WindowData:
        """Прочитать окно ``[start, end]``; отдаются выбранные колонки.

        ``last_full_bar=True`` (по умолчанию) отбрасывает неполный последний бар
        живого края — гарантия «последнего ЗАКРЫТОГО бара» (П7, NaT-guard).
        """
        df = self._normalize(self._fetch_raw(start, end, last_full_bar))
        lo, hi = _as_naive_utc(start), _as_naive_utc(end)
        df = df[(df.index >= lo) & (df.index <= hi)]
        cols = [c for c in self._columns if c in df.columns]
        if not cols:
            raise ValueError(
                f"база не вернула ни одной запрошенной колонки: {self._columns}")
        block = df[cols]
        if block.isna().to_numpy().any():
            raise ValueError("PgSource: NaN в данных окна (пропуск баров в базе)")
        data = np.column_stack([block[c].to_numpy(dtype=np.float64)
                                for c in cols])
        meta = {"source_kind": "pg", "table": self._table,
                "timeframe": self._timeframe, "origin": PG_ORIGIN,
                "use_extended_cols": self._extended, "columns": cols,
                "last_full_bar": last_full_bar, "window_start": str(lo),
                "window_end": str(hi), "n_bars": int(len(df))}
        return WindowData(index=pd.DatetimeIndex(df.index),
                          columns=tuple(cols), data=data,
                          source=f"pg://{self._table}", meta=meta)

    def load_latest_closed(self, now, lookback_bars: int,
                           freq: str = "1min") -> WindowData:
        """Онлайн: последние ``lookback_bars`` ЗАКРЫТЫХ баров на момент ``now``.

        Неполный текущий бар в выборку не попадает. Проверяется assert'ом:
        метка последнего бара + интервал <= ``now`` (бар полностью закрыт).
        """
        now = _as_naive_utc(now)
        step = pd.Timedelta(freq)
        start = now - step * (lookback_bars + 2)
        win = self.load_window(start, now, last_full_bar=True)
        if len(win) == 0:
            return win
        last_label = pd.Timestamp(win.index[-1])
        assert last_label + step <= now, (
            f"живой край: последний бар {last_label} не закрыт к {now} "
            f"(метка+{freq} > now)")
        keep = min(lookback_bars, len(win))
        sl = slice(len(win) - keep, len(win))
        return WindowData(index=win.index[sl], columns=win.columns,
                          data=win.data[sl], source=win.source,
                          meta={**win.meta, "n_bars": keep})


@dataclass(frozen=True)
class GateResult:
    """Результат побитового гейта двух источников по общим колонкам.

    Attributes:
        n_bars: число сравнённых баров (пересечение меток).
        index_equal: совпал ли индекс полностью (строгая сверка).
        columns_compared: колонки, сравнённые по имени (пересечение).
        columns_only_a: колонки только у первого источника.
        columns_only_b: колонки только у второго источника.
        diff_elements: число различающихся элементов (гейт: 0).
        max_abs_diff: максимум |a-b| по общим колонкам (диагностика dtype).
        per_column: число различий по каждой сравнённой колонке.
        first_mismatch: (row, column, a, b) первого расхождения либо ``None``.
        n_only_a: бары только у первого источника (эффект last_full_bar).
        n_only_b: бары только у второго источника.
    """

    n_bars: int
    index_equal: bool
    columns_compared: tuple
    columns_only_a: tuple
    columns_only_b: tuple
    diff_elements: int
    max_abs_diff: float
    per_column: dict
    first_mismatch: Optional[tuple]
    n_only_a: int = 0
    n_only_b: int = 0

    @property
    def passed(self) -> bool:
        """Гейт пройден: есть общие бары И колонки И ноль различий значений.

        Граничные бары, уникальные для одного источника (эффект
        ``last_full_bar``), гейт не проваливают — они в ``n_only_*``.
        """
        return (self.n_bars > 0 and len(self.columns_compared) > 0
                and self.diff_elements == 0)


def bitwise_gate(a: WindowData, b: WindowData) -> GateResult:
    """Сравнить два окна ПОБИТОВО на ПЕРЕСЕЧЕНИИ колонок (train/serve-гейт).

    Сравниваются только колонки с общим именем и только на пересечении меток:
    рассинхрон края (``last_full_bar``) или разный набор колонок не должны
    маскироваться под расхождение значений. Порог по значениям — ноль различий.
    """
    ia, ib = pd.DatetimeIndex(a.index), pd.DatetimeIndex(b.index)
    index_equal = ia.equals(ib)
    common_cols = tuple(c for c in a.columns if c in b.columns)
    only_a_cols = tuple(c for c in a.columns if c not in b.columns)
    only_b_cols = tuple(c for c in b.columns if c not in a.columns)

    common_idx = ia.intersection(ib)
    pa = ia.get_indexer(common_idx)
    pb = ib.get_indexer(common_idx)
    n = len(common_idx)

    per_col: dict = {}
    diff_elements = 0
    max_abs = 0.0
    first = None
    for name in common_cols:
        xa = a.data[pa, a.columns.index(name)]
        xb = b.data[pb, b.columns.index(name)]
        neq = xa != xb
        c = int(neq.sum())
        per_col[name] = c
        diff_elements += c
        if n:
            max_abs = max(max_abs, float(np.max(np.abs(xa - xb))))
        if c and first is None:
            r = int(np.argmax(neq))
            first = (r, name, float(xa[r]), float(xb[r]))
    return GateResult(
        n_bars=n, index_equal=index_equal, columns_compared=common_cols,
        columns_only_a=only_a_cols, columns_only_b=only_b_cols,
        diff_elements=diff_elements, max_abs_diff=max_abs, per_column=per_col,
        first_mismatch=first, n_only_a=int(len(ia.difference(ib))),
        n_only_b=int(len(ib.difference(ia))))


def dbbinance_version() -> str:
    """Версия установленного ``dbbinance-storage`` (для манифеста).

    У пакета нет ``__version__``; берётся из метаданных установки.
    Возвращает ``"unavailable"``, если пакет не установлен.
    """
    try:
        from importlib.metadata import version
        return version("dbbinance-storage")
    except Exception:
        return "unavailable"


def source_manifest(win: WindowData) -> dict:
    """Паспорт источника окна для манифеста прогона.

    Включает версию dbbinance, версию билдера признаков, список выбранных
    колонок и их источник, параметры чтения, границы окна и хэш данных.
    """
    return {"source": win.source, "columns": list(win.columns),
            "has_v7_features": win.has_features(), "n_bars": len(win),
            "data_sha256": win.data_hash(),
            "dbbinance_version": dbbinance_version(),
            "feature_builder_version": FEATURE_BUILDER_VERSION, **win.meta}
