"""Единый интерфейс к данным: parquet-снимок и база PostgreSQL.

Два источника за одним интерфейсом :class:`DataSource` отдают ОДИНАКОВЫЙ
плотный numpy-результат :class:`WindowData` на окне ``[start, end]``.

Разделение источников (принцип П1: источник истины — база, parquet — ускоритель):

* :class:`ParquetSource` — обёртка над :mod:`spotrl.data.state_dataset`.
  Отдаёт OHLCV И предвычисленные признаки (q_buy/q_sell/regime_code/leg_dn),
  потому что снимок собран билдером sunday (``state_v0_builder.py``).
  Не импортирует ``dbbinance`` — работает вообще без базы.
* :class:`PgSource` — база через ``dbbinance-storage``. Отдаёт ТОЛЬКО сырой
  OHLCV: сигнальные признаки в сырой базе отсутствуют, их считает билдер
  стратегии. Импорт ``dbbinance`` — ЛЕНИВЫЙ, внутри метода, чтобы импорт
  этого модуля не дёргал ``secureapikey`` (интерактивный SALT-ввод повесил бы
  headless-прогон, см. handoff/for_sunday_dbbinance_1_0_10).

Гейт эквивалентности :func:`ohlcv_bitwise_gate` сравнивает OHLCV двух
источников на общем окне ПОБИТОВО (порог — ноль различающихся элементов).
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from spotrl.data.state_dataset import (REQUIRED_COLUMNS, StateDataset,
                                       from_frame, load_state)

# Параметры чтения OHLCV, ТОЧНО как в билдере снимка
# (sunday/tooling/audit/state_v0_builder.py:79-85). Любое расхождение здесь
# ломает побитовую эквивалентность OHLCV, поэтому значения зафиксированы.
PG_TABLE = "spot_data_btcusdt_1m"
PG_TIMEFRAME = "1m"
PG_ORIGIN = "start"

# Каталог с *.env-ключами PostgreSQL (проект sunday). Внутри — SALT-фраза и
# зашифрованные логин/пароль; их значения НИКОГДА не логируются. Путь можно
# переопределить переменной окружения SPOTRL_PG_ENV_DIR или параметром PgSource.
DEFAULT_PG_ENV_DIR = "~/Python/projects/sunday"
PG_ENV_FILES = ("PSGSQL_KEY.env", "PSGSQLKEYS.env")

_OHLCV_NAMES = ("open", "high", "low", "close", "volume")

# Идентификатор билдера признаков снимка (sunday state_v0_builder). У билдера нет
# __version__; фиксируем строкой — попадает в манифест прогона (требование 4).
FEATURE_BUILDER_VERSION = "state_v0"


@dataclass(frozen=True)
class WindowData:
    """Плотный результат чтения окна одним источником.

    Индекс — tz-naive UTC (как в снимке: билдер делает
    ``tz_convert("UTC").tz_localize(None)``). OHLCV присутствует всегда;
    признаки могут отсутствовать (``None``), если источник их не отдаёт.

    Attributes:
        index: моменты баров, tz-naive UTC, строго возрастающие.
        ohlcv: массив (n, 5) float64 — open, high, low, close, volume.
        signals: массив (n, 2) float64 (q_buy, q_sell) либо ``None``.
        regime_code: массив (n,) int32 либо ``None``.
        leg_dn: массив (n,) bool либо ``None``.
        source: человекочитаемое имя источника (для манифеста).
        meta: параметры чтения (таблица, окно, версии) для манифеста.
    """

    index: pd.DatetimeIndex
    ohlcv: np.ndarray
    signals: Optional[np.ndarray]
    regime_code: Optional[np.ndarray]
    leg_dn: Optional[np.ndarray]
    source: str
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        """Число баров в окне."""
        return len(self.index)

    def has_features(self) -> bool:
        """Есть ли сигнальные признаки (True только у полного снимка)."""
        return self.signals is not None

    def ohlcv_hash(self) -> str:
        """sha256 массива OHLCV — для манифеста и быстрого сравнения окон."""
        return hashlib.sha256(np.ascontiguousarray(self.ohlcv)).hexdigest()

    def to_state_dataset(self) -> StateDataset:
        """Собрать :class:`StateDataset` для среды.

        Raises:
            ValueError: если признаки отсутствуют (сырой PG без билдера).
        """
        if not self.has_features():
            raise ValueError(
                "источник отдал только OHLCV без сигнальных признаков "
                "(q_buy/q_sell/regime_code/leg_dn); для среды нужен снимок "
                "или прогон билдера признаков поверх OHLCV")
        return StateDataset(index=self.index, ohlcv=self.ohlcv,
                            signals=self.signals, regime_code=self.regime_code,
                            leg_dn=self.leg_dn, source=self.source)


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
    """Источник поверх parquet-снимка (обучение, воспроизводимость)."""

    name = "parquet"

    def __init__(self, path: str | Path):
        """Args: path — путь к parquet каузального state (снимок билдера)."""
        self._path = str(Path(path).expanduser())
        self._state: Optional[StateDataset] = None

    def _load(self) -> StateDataset:
        if self._state is None:
            self._state = load_state(self._path)
        return self._state

    def load_window(self, start, end) -> WindowData:
        """Срез снимка на окне; признаки отдаются полностью."""
        st = self._load()
        lo, hi = _as_naive_utc(start), _as_naive_utc(end)
        idx = pd.DatetimeIndex(st.index)
        mask = (idx >= lo) & (idx <= hi)
        pos = np.flatnonzero(mask)
        meta = {"source_kind": "parquet", "path": self._path,
                "window_start": str(lo), "window_end": str(hi),
                "n_bars": int(pos.size)}
        return WindowData(
            index=idx[pos], ohlcv=st.ohlcv[pos], signals=st.signals[pos],
            regime_code=st.regime_code[pos], leg_dn=st.leg_dn[pos],
            source=self._path, meta=meta)


class PgSource(DataSource):
    """Источник поверх PostgreSQL через ``dbbinance-storage`` (paper/live).

    Отдаёт ТОЛЬКО сырой OHLCV. Импорт ``dbbinance`` ленивый: он тянет
    ``secureapikey`` (интерактивный SALT-ввод), поэтому происходит внутри
    :meth:`_get_fetcher`, а не на уровне модуля.
    """

    name = "pg"

    def __init__(self, table_name: str = PG_TABLE, timeframe: str = PG_TIMEFRAME,
                 fetcher=None, env_dir: Optional[str] = None):
        """Args:

        table_name: имя таблицы 1m-баров в базе.
        timeframe: целевой таймфрейм ресемпла (снимок собран на ``1m``).
        fetcher: уже готовый объект DataFetcher (для тестов/инъекции); если
            ``None`` — создаётся лениво через ``dbbinance`` при первом чтении.
        env_dir: каталог с ``*.env``-ключами PG; по умолчанию
            ``$SPOTRL_PG_ENV_DIR`` или :data:`DEFAULT_PG_ENV_DIR`.
        """
        import os
        self._table = table_name
        self._timeframe = timeframe
        self._fetcher = fetcher
        self._injected = fetcher is not None
        self._env_dir = (env_dir or os.getenv("SPOTRL_PG_ENV_DIR")
                         or DEFAULT_PG_ENV_DIR)

    def _load_secrets(self) -> None:
        """Загрузить ключи PG из ``*.env`` в окружение ДО импорта dbbinance.

        Так ``secureapikey`` расшифровывает доступ без интерактивного SALT-ввода.
        Значения ключей не логируются. Отсутствие файлов не ошибка — тогда
        сработает штатный путь dbbinance (env/интерактив).
        """
        import os
        from dotenv import load_dotenv
        base = Path(self._env_dir).expanduser()
        for name in PG_ENV_FILES:
            path = base / name
            if path.exists():
                load_dotenv(str(path), override=False)

    def _get_fetcher(self):
        """Лениво получить DataFetcher; ключи и импорт dbbinance только здесь."""
        if self._fetcher is None:
            self._load_secrets()
            from dbbinance.fetcher.getfetcher import get_datafetcher
            self._fetcher = get_datafetcher()
        return self._fetcher

    def _use_cols(self):
        """Колонки/типы билдера; импорт констант ленивый."""
        from dbbinance.fetcher.constants import Constants
        return Constants.binance_extended_cols, Constants.binance_extended_dtypes

    def _fetch_raw(self, start, end, last_full_bar: bool) -> pd.DataFrame:
        """Ресемпл базы теми же параметрами, что и билдер снимка."""
        fetcher = self._get_fetcher()
        # Константы dbbinance берём ТОЛЬКО на реальном пути: их импорт тянет
        # пакет dbbinance.fetcher (SALT-ввод). При инъекции fetcher (тесты)
        # не импортируем ничего из dbbinance.
        if self._injected:
            use_cols, use_dtypes = None, None
        else:
            use_cols, use_dtypes = self._use_cols()
        kwargs = dict(table_name=self._table,
                      start=_as_naive_utc(start).to_pydatetime(),
                      end=_as_naive_utc(end).to_pydatetime(),
                      to_timeframe=self._timeframe, origin=PG_ORIGIN,
                      open_time_index=True, last_full_bar=last_full_bar)
        if use_cols is not None:
            kwargs.update(use_cols=use_cols, use_dtypes=use_dtypes)
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
        """Прочитать окно ``[start, end]``; отдаётся только OHLCV.

        ``last_full_bar=True`` (по умолчанию) отбрасывает неполный последний бар
        живого края — гарантия «последнего ЗАКРЫТОГО бара» (П7, NaT-guard).
        """
        df = self._normalize(self._fetch_raw(start, end, last_full_bar))
        lo, hi = _as_naive_utc(start), _as_naive_utc(end)
        df = df[(df.index >= lo) & (df.index <= hi)]
        if df[list(_OHLCV_NAMES)].isna().to_numpy().any():
            raise ValueError("PgSource: NaN в OHLCV окна (пропуск баров в базе)")
        ohlcv = np.column_stack([df[c].to_numpy(dtype=np.float64)
                                 for c in _OHLCV_NAMES])
        meta = {"source_kind": "pg", "table": self._table,
                "timeframe": self._timeframe, "origin": PG_ORIGIN,
                "last_full_bar": last_full_bar, "window_start": str(lo),
                "window_end": str(hi), "n_bars": int(len(df))}
        return WindowData(index=pd.DatetimeIndex(df.index), ohlcv=ohlcv,
                          signals=None, regime_code=None, leg_dn=None,
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
        return WindowData(index=win.index[sl], ohlcv=win.ohlcv[sl],
                          signals=None, regime_code=None, leg_dn=None,
                          source=win.source, meta={**win.meta, "n_bars": keep})


@dataclass(frozen=True)
class GateResult:
    """Результат побитового гейта OHLCV двух источников.

    Attributes:
        n_bars: число сравнённых баров (после выравнивания индекса).
        index_equal: совпал ли индекс побитово.
        diff_elements: число различающихся элементов OHLCV (гейт: 0).
        max_abs_diff: максимум |a-b| по OHLCV (диагностика dtype).
        per_column: число различий по каждой колонке.
        first_mismatch: (row, col, a, b) первого расхождения либо ``None``.
        features_reproducible: список признаков, воспроизводимых из сырого PG.
        features_missing: список признаков, которых в сыром PG нет.
    """

    n_bars: int
    index_equal: bool
    diff_elements: int
    max_abs_diff: float
    per_column: dict
    first_mismatch: Optional[tuple]
    features_reproducible: tuple
    features_missing: tuple
    n_only_a: int = 0
    n_only_b: int = 0

    @property
    def passed(self) -> bool:
        """Гейт пройден: есть общие бары и ноль различий OHLCV на пересечении.

        Граничные бары, присутствующие только у одного источника (эффект
        ``last_full_bar`` на живом крае), гейт OHLCV не проваливают — они
        учтены в ``n_only_a``/``n_only_b`` как диагностика, а не как расхождение
        значений.
        """
        return self.n_bars > 0 and self.diff_elements == 0


# Признаки снимка: OHLCV воспроизводимы из сырого PG побитово; сигнальные —
# нет (их считает билдер стратегии, в сырой базе их не существует).
_FEATURES_REPRODUCIBLE = _OHLCV_NAMES
_FEATURES_MISSING = tuple(c for c in REQUIRED_COLUMNS if c not in _OHLCV_NAMES)


def ohlcv_bitwise_gate(a: WindowData, b: WindowData) -> GateResult:
    """Сравнить OHLCV двух окон ПОБИТОВО (гейт train/serve-эквивалентности).

    Сначала выравнивается индекс (assert равенства как первая линия обороны):
    рассинхрон окна/таймзоны должен падать громко, а не выглядеть «шумом».
    Затем OHLCV сравнивается элемент-в-элемент; порог — ноль различий.
    """
    ia, ib = pd.DatetimeIndex(a.index), pd.DatetimeIndex(b.index)
    index_equal = ia.equals(ib)
    # Сравниваем на ПЕРЕСЕЧЕНИИ меток: рассинхрон края (last_full_bar) не должен
    # маскироваться под расхождение значений. Бары, уникальные для одной
    # стороны, идут в n_only_* как диагностика.
    common = ia.intersection(ib)
    pa = ia.get_indexer(common)
    pb = ib.get_indexer(common)
    n = len(common)
    xa, xb = a.ohlcv[pa], b.ohlcv[pb]
    neq = xa != xb
    diff_elements = int(neq.sum())
    max_abs = float(np.max(np.abs(xa - xb))) if n else 0.0
    per_col = {name: int(neq[:, j].sum()) for j, name in enumerate(_OHLCV_NAMES)}
    first = None
    if diff_elements:
        r, c = (int(x) for x in np.argwhere(neq)[0])
        first = (r, _OHLCV_NAMES[c], float(xa[r, c]), float(xb[r, c]))
    return GateResult(
        n_bars=n, index_equal=index_equal, diff_elements=diff_elements,
        max_abs_diff=max_abs, per_column=per_col, first_mismatch=first,
        features_reproducible=_FEATURES_REPRODUCIBLE,
        features_missing=_FEATURES_MISSING,
        n_only_a=int(len(ia.difference(ib))),
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

    Включает версию dbbinance, параметры чтения, границы окна и хэш OHLCV.
    """
    return {"source": win.source, "has_features": win.has_features(),
            "n_bars": len(win), "ohlcv_sha256": win.ohlcv_hash(),
            "dbbinance_version": dbbinance_version(),
            "feature_builder_version": FEATURE_BUILDER_VERSION,
            "features_missing_in_raw_pg": list(_FEATURES_MISSING),
            **win.meta}
