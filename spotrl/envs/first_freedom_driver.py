"""ЛЕСА КОПИРОВАНИЯ: драйвер ПЕРВОЙ СВОБОДЫ (экспертный override на не-dip барах).

Зачем. При `entry_own=False` `SpotFlipEnv` вход сам НЕ открывает
(`spot_env.step`, ветка входа только под `elif self._entry_own`), поэтому в
чистом PPO-роллауте позиций 0 → dip-in-position баров 0 → маска головы 0 всегда
0 → градиент свободы ровно 0. Первая свобода (ранний выход dip) требует, чтобы
среда на НЕ-свободных барах исполняла решение эксперта v7 (входы, transition-
холды, сигнальные выходы), а на dip-in-position барах honored сэмпл агента.

Механизм (v7hook). Среда работает в конфиге КОПИРОВАНИЯ (`entry_own=True`,
`exit_own=True`, `params_own=False`, TP-sentinel 0.20, V7-константы, CB v7_ledger
— как `build_clone_dataset._make_env`), а этот Wrapper ПОДМЕНЯЕТ действие,
приходящее в `step`:
  * не-dip бар → экспертное действие `bc_action[bar]` (+ директива
    `_exit_is_signal` для сигнальных выходов, как build_clone_dataset);
  * dip-in-position бар → сэмпл агента головы 0 (honored). Агентский выход
    НЕсигнальный (`_exit_is_signal=False`, взводит cooldown, v7hook).

ВАЖНО про PPO-буфер. В on-policy буфер PPO кладётся СЭМПЛ агента (а не
исполненное действие). На не-dip барах исполненное≠сэмпл, но dip-only loss-маска
головы 0 (`first_freedom_head_mask`) даёт там РОВНО нулевой policy-градиент —
рассинхрон действий на маскированных барах в лосс не входит. Value-таргеты и
KL-якорь считаются по ИСПОЛНЕННОЙ траектории (reward/obs), что и требуется.

ЛЕСА КОПИРОВАНИЯ (реестр `scaffolding_registry`). Выключается флагом
`honor_dip=False` (тогда эксперт исполняется ВЕЗДЕ — режим чистой копии, для
паритета) или удаляется целиком при полной свободе (`entry_own=True` без
клонирования): тогда действия агента идут в среду напрямую, драйвер не нужен.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from spotrl.spec.actions import FLIP, STAY, HEAD_POSITION
from spotrl.spec.observation import ObservationSpec


class FirstFreedomDriver(gym.Wrapper):
    """ЛЕСА КОПИРОВАНИЯ: экспертный override действия на не-dip барах.

    Args:
        env: `SpotFlipEnv` в конфиге КОПИРОВАНИЯ (entry_own=True, exit_own=True,
            params_own=False, V7-константы, cb_equity_mode='v7_ledger').
        expert_action: int-массив длины n_bars, действие головы 0 эксперта v7
            по АБСОЛЮТНОМУ бару (0=STAY, 1=FLIP; `bc_action`).
        is_signal_exit: bool-массив длины n_bars — бар сигнального выхода v7
            (директива `_exit_is_signal`, cooldown НЕ взводится).
        honor_dip: если False — эксперт исполняется ВЕЗДЕ (режим чистой копии,
            для паритета драйвера). По умолчанию True (первая свобода).
    """

    _V2_NAMES = tuple(ObservationSpec.v2().names)
    _I_INPOS = _V2_NAMES.index("a_in_position")
    _I_DIP = _V2_NAMES.index("a_pos_tag_dip")
    _I_AGE = _V2_NAMES.index("a_days_in_trade")

    is_copy_scaffolding = True  # реестр лесов копирования

    def __init__(self, env, expert_action: np.ndarray, is_signal_exit: np.ndarray,
                 honor_dip: bool = True, enforce_v7_exit: bool = True,
                 min_hold_days: float = 0.0):
        """Задать экспертные массивы по бару, флаги honored-dip и enforce-v7-exit.

        Args:
            enforce_v7_exit: на dip-баре агент может выйти РАНЬШЕ, но НЕ пропустить
                штатный выход v7: exec_FLIP = agent_FLIP ИЛИ (v7-выход на баре).
                Гарантирует must-copy recall=1.0 (amend1 §1). По умолчанию True.
            min_hold_days: мин-холд (amend3 §1) — агентский ранний выход запрещён,
                пока НАБЛЮДАЕМЫЙ возраст позиции a_days_in_trade < min_hold_days
                (гейт на том же obs-признаке, что видит агент → MDP-безопасно и
                одинаково в train/serve). enforce_v7_exit имеет ПРИОРИТЕТ над
                мин-холдом (выход v7 не подавляется). 0.0 — выкл (поведение amend1).
                60 баров ≈ 0.0245 (a_days_in_trade растёт ~4.08e-4/бар).
        """
        super().__init__(env)
        self._expert = np.asarray(expert_action, dtype=np.int64)
        self._is_sig = np.asarray(is_signal_exit, dtype=bool)
        self._honor_dip = bool(honor_dip)
        self._enforce_v7_exit = bool(enforce_v7_exit)
        self._min_hold_days = float(min_hold_days)
        self._age = -1                 # возраст позиции в барах (диагностика лога)
        self.dip_honored_count = 0     # диагностика: сколько раз honored сэмпл
        self.expert_forced_count = 0
        self.early_exit_count = 0      # агентские выходы РАНЬШЕ v7
        self.minhold_blocked = 0       # сколько раз мин-холд заблокировал выход
        self.log_bars = False          # диагностика (K4-предзапуск): лог env._t/age
        self.step_bars = []            # (bar, age, dip, exec_pos) по шагам rollout
        self.early_exit_log = []       # (bar, age) агентских ранних выходов

    def _dip_in_position(self, raw_obs: np.ndarray) -> bool:
        """dip-in-position по СЫРОМУ наблюдению (a_in_position & a_pos_tag_dip)."""
        return bool(raw_obs[self._I_INPOS] > 0.5 and raw_obs[self._I_DIP] > 0.5)

    def step(self, agent_action):
        """Override: не-dip → эксперт v7; dip → агент (v7-выход и мин-холд учтены)."""
        env = self.env
        t = int(env._t)
        raw = env.observe()                     # 38-мерное СЫРОЕ наблюдение
        in_pos = raw[self._I_INPOS] > 0.5
        self._age = (self._age + 1) if in_pos and self._age >= 0 else \
            (0 if in_pos else -1)
        dip = self._dip_in_position(raw)

        exec_pos = None
        if self._honor_dip and dip:
            agent_pos = int(np.asarray(agent_action).reshape(-1)[HEAD_POSITION])
            v7_exit = self._enforce_v7_exit and int(self._expert[t]) == FLIP
            if v7_exit:
                # штатный выход v7 жёстко срабатывает (ПРИОРИТЕТ над мин-холдом)
                exec_action = np.array([FLIP, 0, 0], dtype=np.int64)
                env._exit_is_signal = bool(self._is_sig[t])
            elif (self._min_hold_days > 0.0
                  and float(raw[self._I_AGE]) < self._min_hold_days):
                # мин-холд: ранний выход запрещён — форсируем STAY (не кредитуется)
                exec_action = np.array([STAY, 0, 0], dtype=np.int64)
                env._exit_is_signal = False
                if agent_pos == FLIP:
                    self.minhold_blocked += 1
            else:
                # свобода: агент может выйти РАНЬШЕ (агентский выход НЕсигнальный)
                exec_action = np.array([agent_pos, 0, 0], dtype=np.int64)
                env._exit_is_signal = False
                if agent_pos == FLIP:
                    self.early_exit_count += 1
                    self.early_exit_log.append((t, self._age))
            self.dip_honored_count += 1
            exec_pos = int(exec_action[HEAD_POSITION])
        else:
            exec_action = np.array([int(self._expert[t]), 0, 0], dtype=np.int64)
            env._exit_is_signal = bool(self._is_sig[t])
            self.expert_forced_count += 1

        if self.log_bars:
            self.step_bars.append((t, self._age, int(dip),
                                   -1 if exec_pos is None else exec_pos))
        return env.step(exec_action)


def load_expert_arrays(dataset_df, n_bars: int):
    """Собрать (expert_action, is_signal_exit) длины n_bars по АБСОЛЮТНОМУ бару.

    Датасет `bc_clone_v7_{epoch}.parquet` — полный проход с бара 0
    (`build_clone_dataset`): столбец `bar` = абсолютный бар решения, `bc_action`
    — действие эксперта головы 0, `is_flip_exit` — сигнальный выход v7.

    Returns:
        (expert_action[int64, n_bars], is_signal_exit[bool, n_bars]).
    """
    expert = np.zeros(n_bars, dtype=np.int64)
    is_sig = np.zeros(n_bars, dtype=bool)
    bar = dataset_df["bar"].to_numpy()
    expert[bar] = dataset_df["bc_action"].to_numpy().astype(np.int64)
    is_sig[bar] = dataset_df["is_flip_exit"].to_numpy().astype(bool)
    return expert, is_sig
