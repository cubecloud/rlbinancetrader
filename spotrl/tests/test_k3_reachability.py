"""Тесты утилит К3-пробы (без внешних данных): интервальная разметка + CI AUC."""
import numpy as np
import pandas as pd

from spotrl.analysis.k3_reachability import _assign_trades, _boot_auc_ci


def test_assign_trades_intervals_and_labels():
    """Бары раскладываются по dip-сделкам, метка SL берётся из exit_reason."""
    trades = pd.DataFrame({
        "entry_bar": [10, 100, 200],
        "exit_bar": [20, 110, 210],
        "pos_tag": ["dip", "dip", "transition"],   # третья — не dip
        "exit_reason": ["sl", "signal", "sl"],
    })
    dip_bars = np.array([10, 15, 20, 100, 105, 205])
    tid, y, n_dip = _assign_trades(dip_bars, trades)
    assert n_dip == 2                              # только dip-сделки
    # бары первой сделки (SL) -> y=1; второй (signal) -> y=0
    assert list(y[:3]) == [1, 1, 1]
    assert list(y[3:5]) == [0, 0]
    assert tid[0] == tid[1] == tid[2]              # одна сделка
    # бар 205 принадлежит transition-интервалу, которого нет в dip -> не назначен
    assert tid[5] == -1


def test_boot_auc_ci_separable():
    """Идеально разделимый сигнал -> CI AUC около 1.0; шум -> около 0.5."""
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200), np.ones(200)].astype(int)
    g = np.r_[np.arange(200), np.arange(200)]      # группы = «сделки»
    p_sep = np.r_[rng.uniform(0, 0.4, 200), rng.uniform(0.6, 1.0, 200)]
    lo, hi, mean = _boot_auc_ci(y, p_sep, g)
    assert lo > 0.9
    p_noise = rng.uniform(0, 1, 400)
    _lo2, _hi2, mean2 = _boot_auc_ci(y, p_noise, g)
    assert 0.4 < mean2 < 0.6          # шум -> AUC около 0.5
