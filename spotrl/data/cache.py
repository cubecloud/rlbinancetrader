"""Кэш материализованных окон :class:`WindowData` (П1: база — истина, кэш —
ускоритель с манифестом).

Идея (образец приёмов — ``strategy_rl/datalayer.py``, ``StateCache``, но код не
импортируется): результат :meth:`DataSource.load_window` материализуется в
parquet рядом с json-манифестом. Ключ кэша — sha256 от ВСЕХ компонентов,
влияющих на данные: тип источника, таблица/путь, набор колонок, границы окна,
timeframe, origin, extended, ``last_full_bar``, версия ``dbbinance``, версия
билдера признаков и версия формата кэша. Любое изменение любого компонента даёт
другой ключ → промах → данные читаются заново. Никаких «тихих» попаданий на
устаревших данных.

Живой край (решение по семантике :meth:`PgSource.load_latest_closed`,
``sources.py:320``): последний ЗАКРЫТЫЙ бар может быть переписан дозагрузкой
базы (ревизия ``volume``/``trades``), поэтому «закрыт» ≠ «окончателен». Окно,
чей фактический последний бар не старше ``now - live_margin``, НЕ кэшируется
целиком. Вариант «кэшировать без последнего бара» отвергнут сознательно: он
вернул бы на тёплом чтении на один бар МЕНЬШЕ, чем на холодном, для того же
вызова ``load_window(start, now)`` — молчаливое изменение поведения и провал
побитового ``cold == warm``. Онлайн-путь ``load_latest_closed`` целится в
``now`` и проксируется сквозь обёртку БЕЗ кэша.

Потокобезопасность: запись атомарна (временный файл в той же директории +
``os.replace``); параллельные среды не видят недописанный файл. Случайные
``*.tmp`` читателем игнорируются.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from spotrl.data.sources import (FEATURE_BUILDER_VERSION, DataSource,
                                 WindowData, _as_naive_utc, dbbinance_version)

# Версия формата самого кэша: бамп ломает все старые ключи (осознанная
# глобальная инвалидация при смене раскладки parquet/манифеста).
CACHE_FORMAT_VERSION = "wc1"

# Директория кэша по умолчанию (требование 5: параметр, не хардкод в коде).
# Переопределяется параметром конструктора или переменной окружения.
DEFAULT_CACHE_DIR = "~/Data/rlbinancetrader/window_cache"
CACHE_DIR_ENV = "SPOTRL_CACHE_DIR"

# Запас у живого края: сколько шагов таймфрейма от ``now`` окно считается
# «незакрытым/неокончательным» и не кэшируется.
DEFAULT_LIVE_MARGIN_BARS = 2


def _canon(obj):
    """Канонизировать значение для стабильного json-хэша ключа."""
    if isinstance(obj, dict):
        return {k: _canon(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_canon(x) for x in obj]
    return obj


def _hash_key(fields: dict) -> str:
    """sha256 от канонизированного словаря компонентов ключа."""
    blob = json.dumps(_canon(fields), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _sha256_file(path: str, chunk: int = 1 << 22) -> str:
    """sha256 файла (для проверки целостности parquet)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class WindowCache:
    """Обёртка над :class:`DataSource`: кэширует закрытые исторические окна.

    Args:
        inner: базовый источник (истина). Должен реализовать
            ``cache_key_fields()`` и ``load_window``.
        cache_dir: директория кэша. По умолчанию — ``$SPOTRL_CACHE_DIR`` либо
            :data:`DEFAULT_CACHE_DIR`.
        now: момент «сейчас» для проверки живого края (инъекция в тестах);
            по умолчанию ``pd.Timestamp.utcnow()`` без tz.
        live_margin_bars: запас у живого края в шагах таймфрейма.
    """

    def __init__(self, inner: DataSource, cache_dir: Optional[str] = None,
                 now=None, live_margin_bars: int = DEFAULT_LIVE_MARGIN_BARS):
        """См. docstring класса."""
        if not hasattr(inner, "cache_key_fields"):
            raise TypeError(
                "источник обязан реализовать cache_key_fields() для кэша")
        self._inner = inner
        base = cache_dir or os.getenv(CACHE_DIR_ENV) or DEFAULT_CACHE_DIR
        self._dir = Path(base).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._now = now
        self._live_margin = int(live_margin_bars)

    @property
    def name(self) -> str:
        """Имя источника (проксируется для манифеста прогона)."""
        return getattr(self._inner, "name", "cached")

    # ---------- ключ и пути ----------

    def _key_fields(self, lo: pd.Timestamp, hi: pd.Timestamp,
                    last_full_bar: Optional[bool]) -> dict:
        """Полный набор компонентов ключа (окно + источник + версии)."""
        fields = dict(self._inner.cache_key_fields())
        fields.update({
            "window_start_ns": int(pd.Timestamp(lo).value),
            "window_end_ns": int(pd.Timestamp(hi).value),
            "last_full_bar": last_full_bar,
            "dbbinance_version": dbbinance_version(),
            "feature_builder_version": FEATURE_BUILDER_VERSION,
            "cache_format_version": CACHE_FORMAT_VERSION,
        })
        return fields

    def _paths(self, key: str):
        return (self._dir / f"{key}.parquet",
                self._dir / f"{key}.manifest.json")

    def _now_ts(self) -> pd.Timestamp:
        return _as_naive_utc(self._now) if self._now is not None \
            else pd.Timestamp.utcnow().tz_localize(None)

    # ---------- живой край ----------

    @staticmethod
    def _step(win: WindowData, key_fields: dict) -> pd.Timedelta:
        """Шаг таймфрейма: из индекса, иначе из timeframe ключа, иначе 1min."""
        idx = pd.DatetimeIndex(win.index)
        if len(idx) >= 2:
            return pd.Timestamp(idx[1]) - pd.Timestamp(idx[0])
        tf = key_fields.get("timeframe")
        if tf:
            return pd.Timedelta(str(tf).replace("m", "min"))
        return pd.Timedelta("1min")

    def _is_cacheable(self, win: WindowData, key_fields: dict) -> bool:
        """Окно кэшируется только если полностью в прошлом (не у живого края).

        Правило: ``last_label + step <= now - live_margin * step``. Пустое окно
        не кэшируется (нечего фиксировать).
        """
        if len(win) == 0:
            return False
        step = self._step(win, key_fields)
        last_label = pd.Timestamp(pd.DatetimeIndex(win.index)[-1])
        cutoff = self._now_ts() - step * self._live_margin
        return last_label + step <= cutoff

    # ---------- чтение/запись ----------

    def _read_cache(self, key: str) -> Optional[WindowData]:
        """Тёплый хит: восстановить WindowData из кэша или None (промах)."""
        parquet, manifest = self._paths(key)
        if not (parquet.exists() and manifest.exists()):
            return None
        with open(manifest) as f:
            man = json.load(f)
        if _sha256_file(str(parquet)) != man["parquet_sha256"]:
            # Испорченный/недописанный parquet — трактуем как промах.
            return None
        frame = pd.read_parquet(parquet)
        columns = tuple(man["columns"])
        data = np.column_stack([
            frame[c].to_numpy(dtype=np.float64) for c in columns]) \
            if columns else np.empty((len(frame), 0), dtype=np.float64)
        data = np.ascontiguousarray(data, dtype=np.float64)
        return WindowData(index=pd.DatetimeIndex(frame.index), columns=columns,
                          data=data, source=man["source"], meta=man["meta"])

    def _atomic_write(self, target: Path, writer) -> None:
        """Записать ``target`` атомарно: УНИКАЛЬНЫЙ temp в той же директории +
        ``os.replace``.

        Уникальное имя temp (``mkstemp``) обязательно: при параллельном
        холодном старте несколько сред пишут ОДИН ключ; детерминированное имя
        temp дало бы гонку (писатель заменяет недописанный файл другого). С
        inode худший исход конкуренции — лишний пересчёт, не порча.
        """
        fd, tmp = tempfile.mkstemp(dir=str(self._dir),
                                   prefix=target.stem + ".",
                                   suffix=target.suffix + ".tmp")
        os.close(fd)
        try:
            writer(tmp)
            os.replace(tmp, target)  # атомарно в пределах одной ФС
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _write_cache(self, key: str, win: WindowData) -> None:
        """Атомарно записать окно: parquet + манифест (temp + os.replace)."""
        parquet, manifest = self._paths(key)
        frame = pd.DataFrame(
            {c: win.data[:, i] for i, c in enumerate(win.columns)},
            index=pd.DatetimeIndex(win.index))
        self._atomic_write(parquet, frame.to_parquet)

        man = {
            "key": key,
            "columns": list(win.columns),
            "source": win.source,
            "meta": win.meta,
            "parquet_sha256": _sha256_file(str(parquet)),
            "data_sha256": win.data_hash(),
            "n_bars": len(win),
            "cache_format_version": CACHE_FORMAT_VERSION,
        }

        def write_manifest(path: str) -> None:
            """Сериализовать манифест в json по временному пути."""
            with open(path, "w") as f:
                json.dump(man, f, ensure_ascii=False, indent=1)

        self._atomic_write(manifest, write_manifest)

    # ---------- публичный интерфейс DataSource ----------

    def load_window(self, start, end, **kwargs) -> WindowData:
        """Прочитать окно через кэш; закрытые исторические окна кэшируются.

        Дополнительные ``kwargs`` (например ``last_full_bar`` у PgSource)
        пробрасываются во ``inner.load_window`` и входят в ключ кэша.
        """
        lo, hi = _as_naive_utc(start), _as_naive_utc(end)
        last_full_bar = kwargs.get("last_full_bar")
        key_fields = self._key_fields(lo, hi, last_full_bar)
        key = _hash_key(key_fields)

        cached = self._read_cache(key)
        if cached is not None:
            return cached  # тёплый хит: inner (и его _get_fetcher) не зовём

        win = self._inner.load_window(start, end, **kwargs)
        if self._is_cacheable(win, key_fields):
            self._write_cache(key, win)
        return win

    def load_latest_closed(self, now, lookback_bars: int, freq: str = "1min"):
        """Онлайн-путь: всегда мимо кэша (целится в живой край)."""
        return self._inner.load_latest_closed(now, lookback_bars, freq=freq)
