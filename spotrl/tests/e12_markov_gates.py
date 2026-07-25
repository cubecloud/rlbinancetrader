"""Гейты немарковости и чистоты наблюдения Э1.2 на BC-разметке v7.

Что здесь: три гейта [Ч] из PLAN v0.5 (раздел 4, задача Э1.2), считаемые на
экспертной траектории v7 (наблюдение + действие эксперта). Модуль ЧИТАЕТ
готовые артефакты strategy_rl (expert-лог, BC-метки, trades) и state-таблицу;
он ничего не правит в strategy_rl и не реализует механику v7 заново (П3).

  ГЕЙТ 1.2а-выход: среди баров с leg_dn=True внутри dip-позиции метки
      EXIT/HOLD обязаны разделяться признаком entered_on_up и ТОЛЬКО им.
      Доля неразделённых пар = 0. Печатается число групп и медианный размер
      группы: если он равен 1, гейт ничего не мерит (тест негоден).
  ГЕЙТ 1.2а-вход: среди баров с entry_signal=True вне позиции метки
      ENTER/WAIT разделяются {остаток cooldown, cb_active, день halt} и только
      ими. Доля неразделённых = 0. entry_signal берётся из ОБЪЕКТА v7
      (stats._strategy._entry) — это чтение готового массива, а не вторая
      реализация входа.
  ГЕЙТ 1.2б: чистота — проверяется отдельно на среде (test_t2_*), здесь не
      дублируется.

Термин «неразделённая пара»: две строки популяции с одинаковыми значениями
разделителей, но разными метками. Если такие есть — разделителей мало,
марковость не выполнена.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GateResult:
    """Результат одного гейта немарковости.

    Attributes:
        population: число баров в популяции гейта.
        n_groups: число групп по значениям разделителей.
        median_group_size: медианный размер группы (страж вырожденности).
        unseparated: число баров в «смешанных» группах (обе метки при равных
            разделителях) — обязано быть 0.
        label_counts: распределение меток в популяции.
    """

    population: int
    n_groups: int
    median_group_size: float
    unseparated: int
    label_counts: dict

    @property
    def passed(self) -> bool:
        """Гейт пройден: нет неразделённых баров и группировка не вырождена."""
        return self.unseparated == 0 and self.median_group_size > 1.0

    def describe(self) -> dict:
        """Сериализуемое описание для отчёта."""
        return {"population": self.population, "n_groups": self.n_groups,
                "median_group_size": self.median_group_size,
                "unseparated": self.unseparated,
                "label_counts": self.label_counts, "passed": self.passed}


def _mixed_group_bars(frame: pd.DataFrame, keys: Sequence[str],
                      label: str) -> GateResult:
    """Сгруппировать по `keys` и посчитать неразделённые бары.

    Неразделённый бар — тот, чья группа (одинаковые значения `keys`) содержит
    более одной метки. Такой бар доказывает немарковость: разделителей не
    хватает, чтобы предсказать действие.
    """
    grp = frame.groupby(list(keys), observed=True)[label]
    sizes = grp.size()
    n_labels = grp.nunique()
    mixed_mask = n_labels > 1
    mixed_keys = sizes.index[mixed_mask.reindex(sizes.index).to_numpy()]
    unseparated = int(sizes[mixed_mask.reindex(sizes.index).to_numpy()].sum()) \
        if len(mixed_keys) else 0
    return GateResult(
        population=int(len(frame)),
        n_groups=int(len(sizes)),
        median_group_size=float(np.median(sizes.to_numpy())) if len(sizes) else 0.0,
        unseparated=unseparated,
        label_counts={str(k): int(v) for k, v in
                      frame[label].value_counts().to_dict().items()})


def exit_gate(expert: pd.DataFrame, trades: pd.DataFrame,
              leg_dn: np.ndarray) -> GateResult:
    """ГЕЙТ 1.2а-выход на ЕДИНОМ прогоне v7 (мастер-инвариант источника).

    Args:
        expert: экспертный лог v7 (bar, in_pos_before, pos_tag, mech_exit,
            agent_exit, exit_sig) ОТ ТОГО ЖЕ прогона, что и `trades`/`leg_dn`.
        trades: сделки v7 того же прогона (entry_bar, exit_bar).
        leg_dn: массив `_leg_dn` того же прогона (бит-в-бит равен колонке state).

    Разделитель — entered_on_up (не leg_dn на баре РЕШЕНИЯ о входе = entry_bar-1;
    off-by-one: вход эксперта исполняется по open(entry_bar), а сигнал считан на
    entry_bar-1, regimeb_bt_strategy.py:317).

    Разделители — ПАРА {entered_on_up, exit_sig}: сигнальный выход объясняет
    ДРУГОЙ наблюдаемый признак `_exit[i]` (раздел 4 PLAN прямо требует его в
    наблюдении наравне с entered_on_up), а не entered_on_up. Поэтому оба
    признака входят в разделители; исключать сигнальные бары нельзя — это
    спрятало бы обязательный признак `_exit[i]` из проверки.

    ВАЖНО (мастер-инвариант): expert, trades и leg_dn обязаны быть из ОДНОГО
    прогона v7. Смешение источников (например BC-метки из causal-таблицы и
    trades из non-causal) сдвигает границы сделок и ломает присвоение
    entered_on_up — гейт тогда ложно проваливается.
    """
    bar = expert["bar"].to_numpy()
    entered_on_up = np.full(len(expert), -1, dtype=np.int64)
    for t in trades.sort_values("entry_bar").itertuples():
        lo, hi = int(t.entry_bar), int(t.exit_bar)
        entered_on_up[(bar >= lo) & (bar <= hi)] = int(not bool(leg_dn[lo - 1]))
    in_pos = expert["in_pos_before"].to_numpy().astype(bool)
    dip = (expert["pos_tag"].to_numpy().astype(str) == "dip") & in_pos
    is_exit = (expert["mech_exit"].to_numpy().astype(bool)
               | expert["agent_exit"].to_numpy().astype(bool))
    signal_exit = expert["exit_sig"].to_numpy().astype(bool)
    label = np.where(is_exit, "EXIT", "HOLD")
    pop = dip & leg_dn[bar] & (entered_on_up >= 0)
    frame = pd.DataFrame({"entered_on_up": entered_on_up[pop],
                          "exit_sig": signal_exit[pop], "label": label[pop]})
    return _mixed_group_bars(frame, ["entered_on_up", "exit_sig"], "label")


def entry_gate(expert: pd.DataFrame, entry_signal: np.ndarray) -> GateResult:
    """ГЕЙТ 1.2а-вход на экспертной траектории.

    Args:
        expert: экспертный лог v7 (bar, in_pos_before, entry_placed,
            cooldown_active, cb_active, time).
        entry_signal: булев массив self._entry по всем барам (из объекта v7).

    Разделители — {cooldown_active, cb_active, день halt}. День halt берётся
    как календарная дата бара при активном cb (снятие CB — по смене
    календарного дня, regimeb_bt_strategy.py:229-236); при неактивном cb дата
    не влияет, поэтому кодируется как пустая строка.
    """
    bar = expert["bar"].to_numpy()
    sig = entry_signal[bar]
    flat = ~expert["in_pos_before"].to_numpy().astype(bool)
    pop = sig & flat
    placed = expert["entry_placed"].to_numpy().astype(bool)
    cb = expert["cb_active"].to_numpy().astype(bool)
    day = pd.to_datetime(expert["time"]).dt.date.astype(str).to_numpy()
    halt_day = np.where(cb, day, "")
    frame = pd.DataFrame({
        "cooldown": expert["cooldown_active"].to_numpy().astype(bool)[pop],
        "cb": cb[pop],
        "halt_day": halt_day[pop],
        "label": np.where(placed[pop], "ENTER", "WAIT")})
    return _mixed_group_bars(frame, ["cooldown", "cb", "halt_day"], "label")
