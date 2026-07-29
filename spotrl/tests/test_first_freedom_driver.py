"""Юнит-тесты драйвера первой свободы (override действия на не-dip барах)."""
from __future__ import annotations

import numpy as np

from spotrl.envs.first_freedom_driver import FirstFreedomDriver, load_expert_arrays
from spotrl.spec.observation import ObservationSpec

_NAMES = list(ObservationSpec.v2().names)
_I_INPOS = _NAMES.index("a_in_position")
_I_DIP = _NAMES.index("a_pos_tag_dip")
_DIM = len(_NAMES)


class _StubEnv:
    """Мини-стаб SpotFlipEnv: фиксированный obs на баре t, запоминает действие."""

    def __init__(self, obs_by_bar):
        """obs_by_bar: список 38-мерных наблюдений по бару."""
        self._obs_by_bar = obs_by_bar
        self._t = 0
        self._exit_is_signal = None
        self.executed = []
        self.observation_space = None
        self.action_space = None

    def observe(self):
        """Наблюдение текущего бара."""
        return self._obs_by_bar[self._t]

    def step(self, action):
        """Записать исполненное действие и продвинуть бар."""
        self.executed.append(np.asarray(action).copy())
        self._t += 1
        done = self._t >= len(self._obs_by_bar)
        return np.zeros(_DIM, np.float32), 0.0, False, done, {}


def _mk_obs(in_pos: bool, dip: bool) -> np.ndarray:
    o = np.zeros(_DIM, np.float32)
    o[_I_INPOS] = 1.0 if in_pos else 0.0
    o[_I_DIP] = 1.0 if dip else 0.0
    return o


def test_load_expert_arrays_aligns_by_bar():
    """expert/is_sig раскладываются по абсолютному бару."""
    import pandas as pd
    df = pd.DataFrame({"bar": [0, 2, 3], "bc_action": [1, 0, 1],
                       "is_flip_exit": [False, False, True]})
    exp, sig = load_expert_arrays(df, n_bars=5)
    assert list(exp) == [1, 0, 0, 1, 0]
    assert list(sig.astype(int)) == [0, 0, 0, 1, 0]


def test_non_dip_bar_forces_expert_ignoring_agent():
    """На не-dip баре исполняется экспертное действие, сэмпл агента игнорируется."""
    # бар0: flat (не dip) → эксперт FLIP; бар1: in-pos+dip → honored агент
    env = _StubEnv([_mk_obs(False, False), _mk_obs(True, True)])
    expert = np.array([1, 0], np.int64)          # эксперт: FLIP, STAY
    is_sig = np.array([True, False])
    drv = FirstFreedomDriver(env, expert, is_sig, honor_dip=True)
    drv.step(np.array([0, 0, 0]))                # агент STAY на flat-баре
    assert list(env.executed[-1]) == [1, 0, 0]   # исполнен эксперт FLIP
    assert env._exit_is_signal is True           # сигнальный выход помечен


def test_dip_bar_honors_agent_sample():
    """На dip-in-position баре исполняется сэмпл агента (не эксперт)."""
    env = _StubEnv([_mk_obs(True, True), _mk_obs(True, True)])
    expert = np.array([0, 0], np.int64)          # эксперт STAY
    is_sig = np.array([False, False])
    drv = FirstFreedomDriver(env, expert, is_sig, honor_dip=True)
    drv.step(np.array([1, 0, 0]))                # агент FLIP → honored
    assert list(env.executed[-1]) == [1, 0, 0]
    assert env._exit_is_signal is False          # агентский выход НЕсигнальный


def test_honor_off_forces_expert_on_dip():
    """honor_dip=False → эксперт исполняется и на dip баре (режим чистой копии)."""
    env = _StubEnv([_mk_obs(True, True)])
    drv = FirstFreedomDriver(env, np.array([0], np.int64), np.array([False]),
                             honor_dip=False)
    drv.step(np.array([1, 0, 0]))                # агент FLIP игнорируется
    assert list(env.executed[-1]) == [0, 0, 0]   # исполнен эксперт STAY


def test_enforce_v7_exit_on_dip_agent_cannot_skip():
    """На dip баре v7-выход жёстко срабатывает, даже если агент STAY (amend1 §1)."""
    env = _StubEnv([_mk_obs(True, True)])
    expert = np.array([1], np.int64)             # v7 выходит (FLIP) на этом баре
    is_sig = np.array([True])
    drv = FirstFreedomDriver(env, expert, is_sig, honor_dip=True,
                             enforce_v7_exit=True)
    drv.step(np.array([0, 0, 0]))                # агент STAY → НЕ пропустит выход v7
    assert list(env.executed[-1]) == [1, 0, 0]   # исполнен FLIP (v7-выход)
    assert env._exit_is_signal is True           # это сигнальный выход v7
    assert drv.early_exit_count == 0             # это НЕ ранний выход агента


def test_early_exit_counted_when_agent_flips_before_v7():
    """Агентский FLIP на dip, где v7 держит (STAY), считается ранним выходом."""
    env = _StubEnv([_mk_obs(True, True)])
    drv = FirstFreedomDriver(env, np.array([0], np.int64), np.array([False]),
                             honor_dip=True, enforce_v7_exit=True)
    drv.step(np.array([1, 0, 0]))                # агент FLIP раньше v7
    assert list(env.executed[-1]) == [1, 0, 0]
    assert env._exit_is_signal is False          # агентский выход НЕсигнальный
    assert drv.early_exit_count == 1


def test_marked_as_copy_scaffolding():
    """Драйвер помечен как леса копирования (реестр)."""
    assert FirstFreedomDriver.is_copy_scaffolding is True
