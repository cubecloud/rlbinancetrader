"""Датасет клонирования поведения v7 (BC) — наблюдение v2 + метка головы 0.

Что делает. Прогоняет эксперта v7 через СРЕДУ (`SpotFlipEnv`) в режиме
копирования (`cb_equity_mode="v7_ledger"`, паритет 6/6) полным детерминированным
проходом от бара 0. На КАЖДОМ баре решения t записывает:
  * наблюдение v2 — ровно то, что вернёт `env.observe()` НА баре t ДО `step`
    (единственный путь упаковки; на этом кондиционируется политика);
  * метку головы 0 (STAY/FLIP) — экспертное действие v7;
  * метаданные: pos_tag, exit_reason, bc_weight, epoch.

СЕМАНТИКА МЕТКИ ГОЛОВЫ 0 (важно, отличается от «2×число сделок» в тексте задачи).
FLIP ставится ТОЛЬКО там, где решение принимает политика:
  * бар входа (v7 разместил вход): decision_bar = entry_bar − 1 → FLIP;
  * бар СИГНАЛЬНОГО выхода (tp/legflip/signal/b2b/agent): exit_bar − 1 → FLIP;
  * SL и forced_eod исполняет СРЕДА (брокерский стоп внутрибарно по Low / конец
    данных) при метке STAY. Пометить их FLIP НЕЛЬЗЯ: агентский FLIP-close в
    `spot_env.step` закрывается по open(t+1) с exit_reason="agent" ДО проверки SL
    (spot_env.py:252-257 против :289-291), то есть по ДРУГОЙ цене и причине —
    паритет бар-в-бар с v7 ломается. Поэтому:
        число FLIP = число входов + число сигнальных выходов
                   = 2×число_сделок − (n_SL + n_forced_eod).
Сами сделки (вход/выход/причина) среда воспроизводит бар-в-бар ВСЕ (SL/forced
закрывает среда) — это отдельный, уже зелёный гейт `parity_env_expert` (6/6).

Реестр лесов копирования: cb_equity_mode="v7_ledger" (см. spot_env, WorldConfig).
Обучение НЕ запускается здесь (П8): сначала pre-reg порогов, потом обучение.

Run (env rlbinancetrader):
  python -m spotrl.bc.build_clone_dataset --data /home/cubecloud/Data
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from spotrl.config import EnvConfig, FreedomConfig, WorldConfig
from spotrl.data.state_dataset import attach_signals, load_state
from spotrl.envs.spot_env import SpotFlipEnv
from spotrl.spec.actions import ActionSpec, FLIP, STAY
from spotrl.spec.observation import ObservationSpec

# Причины выхода, которые в v7 закрывают сделку СИГНАЛЬНО (решение политики) —
# на баре решения (exit_bar−1) ставится FLIP. SL/forced_eod закрывает среда.
SIGNAL_EXIT_REASONS = frozenset({"tp", "legflip", "signal", "b2b", "agent"})
ENV_DRIVEN_REASONS = frozenset({"sl", "forced_eod", "end"})

# v7 (P7) параметры мира — среда обязана прогоняться на них (иначе тайминги
# закрытий/cooldown разъедутся). Совпадают с гейтом паритета.
V7_SL = 0.07
V7_COOLDOWN_BARS = 423
V7_CB_DD = 0.15


@dataclass
class BuildReport:
    """Самоконтроль сборки одной эпохи (числа для отчёта и манифеста)."""

    epoch: str
    n_bars: int = 0
    n_rows: int = 0
    n_flip: int = 0
    n_flip_entry: int = 0
    n_flip_exit: int = 0
    n_stay: int = 0
    n_trades_env: int = 0
    n_trades_csv: int = 0
    n_signal_exits: int = 0
    n_sl: int = 0
    n_forced_eod: int = 0
    entry_bars_match: bool = False
    exit_bars_match: bool = False
    trade_tuples_match: bool = False
    obs_equiv_checked: int = 0
    obs_equiv_mismatch: int = 0
    entry_signal_coverage: int = 0   # FLIP-входов с булевым сигналом v7 в obs
    exit_keyed_coverage: int = 0     # FLIP-выходов с наблюдаемым ключом причины
    entry_collisions: int = 0        # flat+сигнал, STAY, НЕ блок cb/cooldown
    exit_reason_counts: Dict[str, int] = field(default_factory=dict)
    pos_tag_counts: Dict[str, int] = field(default_factory=dict)

    def describe(self) -> dict:
        """Сериализуемая сводка для манифеста."""
        return {k: (v if not isinstance(v, (np.integer,)) else int(v))
                for k, v in self.__dict__.items()}


def _file_id(path: str) -> dict:
    """Идентификатор версии входного файла для манифеста (без чтения содержимого).

    Хэш 129МБ parquet дорог; для версионирования достаточно устойчивого
    идентификатора: размер в байтах + mtime. Попадает в манифест прогона
    (PLAN 1.5.1, п.3: любая величина, влияющая на результат, — в манифест).
    """
    p = Path(path).expanduser()
    st = p.stat()
    return {"path": str(p), "bytes": st.st_size, "mtime": int(st.st_mtime)}


def _make_env(state_path: str, signals_path: str) -> SpotFlipEnv:
    """Собрать среду копирования на v7-параметрах (как гейт паритета 6/6)."""
    dataset = attach_signals(load_state(state_path), signals_path)
    n = len(dataset)
    world = WorldConfig(stop_loss_frac=V7_SL, cooldown_bars=V7_COOLDOWN_BARS,
                        breaker_drawdown_frac=V7_CB_DD, close_at_end=True,
                        cb_equity_mode="v7_ledger")
    freedom = FreedomConfig(exit_own=True, entry_own=True, params_own=False)
    # TP-корзина копирования = COPY_TP_SENTINEL (НЕ 1e9). Все НЕ-SL выходы ведёт
    # драйвер экспертным FLIP; чтобы среда не срабатывала своим TP раньше v7,
    # порог должен быть выше максимального валового unreal по всем барам в позиции
    # (замер: 0.1222 на 2024, 0.1423 на 2021). 0.20 это условие выполняет с
    # запасом — паритет сделок 162/162 и 272/272 сохранён (проверено:
    # pairs_match и driven_match True на обеих эпохах). ЗАЧЕМ не 1e9: наблюдение
    # a_dist_to_tp = tp_frac − unreal; при 1e9 это float32-константа ~1e9 на всех
    # ~53% баров в позиции — токсичный вход в сеть (доминирует первый слой) и
    # разрыв train/serve (в самостоятельном режиме признак O(0.1)). При 0.20
    # dist_to_tp = 0.20 − unreal, масштаба O(0.1), как реальные корзины serve
    # (PLAN 3.5.4: TP {8,12,16%}). Это КОПИРОВОЧНЫЙ sentinel, а не реальный TP v7
    # (12%/6%): реальный per-pos_tag TP одной корзиной невыразим и он ниже макс.
    # unreal → среда опередила бы драйвер и сломала паритет. Для клона головы 0
    # это не мешает: TP-выходы ключуются по a_unreal_pnl + pos_tag (см. MDP-чек).
    COPY_TP_SENTINEL = 0.20
    action_spec = ActionSpec(tp_buckets=(COPY_TP_SENTINEL,), expert_tp_index=0)
    config = EnvConfig(world=world, freedom=freedom, action_spec=action_spec,
                       episode_len=n + 10)   # старт с бара 0, полный проход
    return SpotFlipEnv(dataset, config)


def build_epoch(epoch: str, state_path: str, signals_path: str,
                expert_path: str, trades_path: str,
                equiv_stride: int = 500) -> tuple[pd.DataFrame, BuildReport, dict]:
    """Собрать датасет клонирования одной эпохи + самоконтроль.

    Args:
        epoch: метка эпохи ("2021"/"2024").
        state_path: parquet каузального state.
        signals_path: артефакт булевых сигналов v7 (dump_v7_signals).
        expert_path: parquet побарного экспертного лога v7 (entry_placed и т.п.).
        trades_path: CSV размеченных сделок v7 (exit_reason/pos_tag/bc_weight).
        equiv_stride: шаг сэмплирования сверки observe()==obs из step (0=выкл).

    Returns:
        (df, report, manifest): датасет, самоконтроль, манифест.
    """
    env = _make_env(state_path, signals_path)
    n = env._n_bars
    spec = env.config.obs_spec

    expert = pd.read_parquet(expert_path)
    trades = pd.read_csv(trades_path)
    rep = BuildReport(epoch=epoch, n_bars=n)
    rep.n_trades_csv = len(trades)
    rep.exit_reason_counts = trades["exit_reason"].value_counts().to_dict()
    rep.pos_tag_counts = trades["pos_tag"].value_counts().to_dict()
    rep.n_sl = int((trades["exit_reason"] == "sl").sum())
    rep.n_forced_eod = int((trades["exit_reason"] == "forced_eod").sum())
    rep.n_signal_exits = int(trades["exit_reason"].isin(SIGNAL_EXIT_REASONS).sum())

    # Экспертные факты по бару решения (как в parity_env_expert).
    b = expert["bar"].to_numpy()
    entry_placed = np.zeros(n, dtype=bool)
    entry_placed[b] = expert["entry_placed"].to_numpy().astype(bool)
    # Бар решения о СИГНАЛЬНОМ выходе (exit_bar−1) и его причина.
    signal_exit_reason: Dict[int, str] = {}
    for tr in trades.itertuples():
        if str(tr.exit_reason) in SIGNAL_EXIT_REASONS:
            signal_exit_reason[int(tr.exit_bar) - 1] = str(tr.exit_reason)
    # forced_eod: бар решения о (env-driven) выходе → bc_weight=0 на этом STAY-баре
    # (его цена выхода анти-каузальна; label_exits:89). Вход forced_eod-сделки —
    # каузально валиден, остаётся весом 1.0 (реш. дизайна, в манифесте).
    forced_exit_bars = {int(tr.exit_bar) - 1 for tr in trades.itertuples()
                        if str(tr.exit_reason) == "forced_eod"}
    # Эталонные множества баров решения для самоконтроля.
    ref_entry_bars = {int(tr.entry_bar) - 1 for tr in trades.itertuples()}
    ref_exit_bars = set(signal_exit_reason.keys())
    # Индексы наблюдаемых ключей (MDP-достаточность копирования).
    i_entry = spec.index_of("m_entry_signal")
    i_trans = spec.index_of("m_trans_entry_signal")
    i_exit_flag = spec.index_of("m_exit_flag")
    i_leg_dn = spec.index_of("m_leg_dn")
    i_dist_tp = spec.index_of("a_dist_to_tp")
    i_unreal = spec.index_of("a_unreal_pnl")

    # Предвыделенные буферы.
    obs_buf = np.zeros((n, spec.size), dtype=np.float32)
    bar_col = np.zeros(n, dtype=np.int64)
    act_col = np.full(n, STAY, dtype=np.int8)
    is_entry = np.zeros(n, dtype=bool)
    is_exit = np.zeros(n, dtype=bool)
    postag_col = np.empty(n, dtype=object)
    reason_col = np.empty(n, dtype=object)
    weight_col = np.ones(n, dtype=np.float32)

    stay = np.array([STAY, 0, 0])
    flip = np.array([FLIP, 0, 0])

    env.reset(seed=0)
    assert env._start == 0, f"ожидался старт с бара 0, получен {env._start}"
    row = 0
    got_entry_bars: set = set()
    got_exit_bars: set = set()

    while True:
        t = env._t
        trade = env.book.open_trade
        # Наблюдение НА баре решения t, ДО step — единственный путь упаковки.
        obs_t = env.observe()
        obs_buf[row] = obs_t
        bar_col[row] = t

        # Экспертное действие головы 0 (SL/forced_eod закрывает среда при STAY).
        action = stay
        env._exit_is_signal = False
        reason = ""
        pos_tag = trade.pos_tag if trade is not None else "none"
        if trade is None:
            if entry_placed[t]:
                action = flip
                act_col[row] = FLIP
                is_entry[row] = True
                got_entry_bars.add(t)
                rep.n_flip_entry += 1
                # pos_tag входимой сделки определяется как в среде.
                pos_tag = ("dip" if bool(env._entry_sig[t])
                           else "transition" if bool(env._trans_sig[t]) else "dip")
                # MDP-достаточность: булев сигнал v7 обязан быть виден в obs.
                if obs_t[i_entry] >= 0.5 or obs_t[i_trans] >= 0.5:
                    rep.entry_signal_coverage += 1
        else:
            if t in signal_exit_reason:
                action = flip
                act_col[row] = FLIP
                is_exit[row] = True
                got_exit_bars.add(t)
                rep.n_flip_exit += 1
                env._exit_is_signal = True
                reason = signal_exit_reason[t]
                # Наблюдаемый ключ причины выхода в obs (диагностика pre-reg):
                #   signal/b2b -> m_exit_flag; legflip -> m_leg_dn;
                #   tp -> a_unreal_pnl у порога TP (в среде копирования dist_to_tp
                #   отключён корзиной 1e9, поэтому TP кодируется чистым PnL +
                #   pos_tag: dip 12%≈0.118, transition 6%≈0.058 чистыми).
                keyed = (obs_t[i_exit_flag] >= 0.5 or obs_t[i_leg_dn] >= 0.5
                         or float(obs_t[i_unreal]) >= 0.05
                         or abs(float(obs_t[i_dist_tp])) <= 0.02)
                if keyed:
                    rep.exit_keyed_coverage += 1
            # SL/forced_eod — STAY, закроет среда.

        postag_col[row] = pos_tag
        reason_col[row] = reason
        if t in forced_exit_bars:
            weight_col[row] = 0.0

        obs_step, _r, _term, trunc, _info = env.step(action)

        # Сверка путей упаковки: obs из step (для бара nxt) == observe() на nxt.
        if equiv_stride and (row % equiv_stride == 0) and not trunc:
            ref = env.observe()
            rep.obs_equiv_checked += 1
            if not np.array_equal(obs_step, ref):
                rep.obs_equiv_mismatch += 1

        row += 1
        if trunc:
            break

    rep.n_rows = row
    rep.n_flip = rep.n_flip_entry + rep.n_flip_exit
    rep.n_stay = row - rep.n_flip
    rep.n_trades_env = len(env.book.closed)
    rep.entry_bars_match = (got_entry_bars == ref_entry_bars)
    rep.exit_bars_match = (got_exit_bars == ref_exit_bars)

    # Trade-level паритет. Пары (entry_bar, exit_bar) обязаны совпасть 1:1 с
    # judge_trades_v7. Метку ПРИЧИНЫ сигнального выхода среда пишет "agent" (их
    # инициирует драйвер FLIP-close), поэтому по причине сверяем только
    # env-управляемые выходы (sl, forced_eod/end) — их среда определяет сама.
    # Истинная причина v7 сигнального выхода сохранена в колонке exit_reason.
    env_pairs = {(c.entry_bar, c.exit_bar) for c in env.book.closed}
    csv_pairs = {(int(tr.entry_bar), int(tr.exit_bar)) for tr in trades.itertuples()}
    env_driven = {(c.entry_bar, c.exit_bar,
                   ("forced_eod" if c.exit_reason == "end" else c.exit_reason))
                  for c in env.book.closed if c.exit_reason in ("sl", "end")}
    csv_driven = {(int(tr.entry_bar), int(tr.exit_bar), str(tr.exit_reason))
                  for tr in trades.itertuples()
                  if str(tr.exit_reason) in ("sl", "forced_eod")}
    rep.trade_tuples_match = (env_pairs == csv_pairs and env_driven == csv_driven)

    # ОТДЕЛИМОСТЬ ВХОДА (не только recall). Решение v7 о входе обязано быть
    # чистой функцией наблюдения. Проверяем отсутствие коллизии: flat-бар, где
    # булев сигнал входа v7 включён, но действие = STAY (v7 НЕ разместил вход),
    # и при этом бар НЕ заблокирован ни cb, ни cooldown — такой бар был бы
    # неотличим от FLIP-входа теми же признаками. Их должно быть 0: каждый
    # STAY-при-сигнале блокируется наблюдаемым cb_active/cooldown_remain.
    obs = obs_buf[:row]
    act = act_col[:row]
    i_inpos = spec.index_of("a_in_position")
    i_cb = spec.index_of("a_cb_active")
    i_cd = spec.index_of("a_cooldown_remain")
    flat = obs[:, i_inpos] < 0.5
    has_sig = (obs[:, i_entry] >= 0.5) | (obs[:, i_trans] >= 0.5)
    stay = act == STAY
    unblocked = (obs[:, i_cb] < 0.5) & (obs[:, i_cd] <= 0.0)
    rep.entry_collisions = int(np.count_nonzero(flat & has_sig & stay & unblocked))

    # ---- ЖЁСТКИЕ САМОКОНТРОЛИ (падают ДО записи артефакта) ----
    # Баров решения n−1: у последнего бара нет open(t+1) для исполнения (trunc).
    assert rep.n_rows == n - 1, f"строк {rep.n_rows} != баров решения {n - 1}"
    assert rep.n_flip == len(ref_entry_bars) + len(ref_exit_bars), \
        f"FLIP {rep.n_flip} != входы+сигнальные выходы"
    assert rep.n_flip == 2 * rep.n_trades_csv - (rep.n_sl + rep.n_forced_eod), \
        "FLIP != 2×trades − (n_SL + n_forced_eod)"
    assert rep.entry_bars_match, "бары входа FLIP != {entry_bar−1} из trades"
    assert rep.exit_bars_match, "бары сигнального выхода FLIP != {exit_bar−1}"
    assert rep.n_trades_env == rep.n_trades_csv, \
        f"сделок среды {rep.n_trades_env} != CSV {rep.n_trades_csv}"
    assert rep.trade_tuples_match, "состав сделок среды != judge_trades_v7"
    assert rep.obs_equiv_mismatch == 0, \
        f"расхождений упаковки observe()!=step: {rep.obs_equiv_mismatch}"
    assert rep.entry_signal_coverage == rep.n_flip_entry, \
        "не на всех барах входа виден булев сигнал v7 в obs (MDP-недостаточность)"
    assert rep.entry_collisions == 0, \
        (f"коллизии входа: {rep.entry_collisions} flat+сигнал STAY-баров не "
         "блокированы cb/cooldown — вход неотделим по наблюдению")

    # ---- Датасет (обрезка до заполненных n−1 строк решения) ----
    df = pd.DataFrame(obs_buf[:row], columns=list(spec.names))
    df.insert(0, "bar", bar_col[:row])
    df["bc_action"] = act_col[:row]
    df["is_flip_entry"] = is_entry[:row]
    df["is_flip_exit"] = is_exit[:row]
    df["pos_tag"] = postag_col[:row].astype(str)
    df["exit_reason"] = reason_col[:row].astype(str)
    df["bc_weight"] = weight_col[:row]
    df["epoch"] = epoch

    manifest = {
        "epoch": epoch,
        "obs_spec_version": spec.version,
        "obs_spec_hash": spec.spec_hash(),
        "obs_size": spec.size,
        "obs_names": list(spec.names),
        "action_spec": env.config.action_spec.describe(),
        "world": env.config.world.describe(),
        "freedom": env.config.freedom.describe(),
        "scaffolding": {"cb_equity_mode": env.config.world.cb_equity_mode},
        "state_source": state_path,
        "signals_source": signals_path,
        "trades_source": trades_path,
        "data_version": {name: _file_id(p) for name, p in
                         (("state", state_path), ("signals", signals_path),
                          ("trades", trades_path))},
        "data_start": str(env.state.index[0]),
        "data_end": str(env.state.index[-1]),
        "flip_label_semantics":
            "FLIP = входы + сигнальные выходы; SL/forced_eod исполняет среда "
            "при STAY (2×trades НЕ держится — см. docstring)",
        "bc_weight_policy":
            "forced_eod выходной STAY-бар = 0.0; вход forced_eod-сделки = 1.0",
        "selfcheck": rep.describe(),
    }
    return df, rep, manifest


def _print_report(rep: BuildReport) -> None:
    """Печать самоконтроля эпохи."""
    print(f"[{rep.epoch}] баров={rep.n_bars:,} строк={rep.n_rows:,}")
    print(f"  сделок: env={rep.n_trades_env} csv={rep.n_trades_csv} "
          f"tuple_match={rep.trade_tuples_match}")
    print(f"  причины выходов: {rep.exit_reason_counts}")
    print(f"  FLIP={rep.n_flip} (входы={rep.n_flip_entry} + сигн.выходы={rep.n_flip_exit}); "
          f"STAY={rep.n_stay:,}")
    print(f"  контроль FLIP: 2×trades−(SL+forced) = "
          f"{2 * rep.n_trades_csv - (rep.n_sl + rep.n_forced_eod)} "
          f"(SL={rep.n_sl}, forced={rep.n_forced_eod}); 2×trades={2 * rep.n_trades_csv}")
    print(f"  bar-set: entry_match={rep.entry_bars_match} exit_match={rep.exit_bars_match}")
    print(f"  obs упаковка: сверено={rep.obs_equiv_checked} расхождений={rep.obs_equiv_mismatch}")
    print(f"  MDP: entry_signal_cov={rep.entry_signal_coverage}/{rep.n_flip_entry}; "
          f"exit_keyed_cov={rep.exit_keyed_coverage}/{rep.n_flip_exit}; "
          f"entry_collisions={rep.entry_collisions}")
    flip_share = rep.n_flip / rep.n_rows
    print(f"  доля FLIP={flip_share:.3e} (входы={rep.n_flip_entry / rep.n_rows:.3e}, "
          f"выходы={rep.n_flip_exit / rep.n_rows:.3e})")


def main() -> None:
    """CLI: собрать датасет клонирования по обеим эпохам раздельными артефактами."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/cubecloud/Data")
    ap.add_argument("--out", default="/home/cubecloud/Data/rlbinancetrader")
    ap.add_argument("--equiv-stride", type=int, default=500)
    args = ap.parse_args()
    base, out = args.data, Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    epochs = [
        ("2021", f"{base}/sunday_tests/state_v0/state_v3_causal_2021-01_2024-03.parquet",
         f"{base}/rlbinancetrader/v7signals_2021.parquet",
         f"{base}/rlbinancetrader/expert_v7_2021-01_2024-03.parquet",
         f"{base}/rlbinancetrader/trades_v7_labeled_2021.csv"),
        ("2024", f"{base}/sunday_tests/state_v0/state_v3_causal_2024-03_2026-07.parquet",
         f"{base}/rlbinancetrader/v7signals_2024.parquet",
         f"{base}/rlbinancetrader/expert_v7_2024-03_2026-07.parquet",
         f"{base}/rlbinancetrader/trades_v7_labeled_2024.csv"),
    ]
    for epoch, state, signals, expert, trades in epochs:
        t0 = time.time()
        df, rep, manifest = build_epoch(epoch, state, signals, expert, trades,
                                        equiv_stride=args.equiv_stride)
        ds_path = out / f"bc_clone_v7_{epoch}.parquet"
        mf_path = out / f"bc_clone_v7_{epoch}.manifest.json"
        df.to_parquet(ds_path)
        mf_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        _print_report(rep)
        print(f"  СОХРАНЕНО: {ds_path} + манифест ({time.time() - t0:.1f} c)\n")


if __name__ == "__main__":
    main()
