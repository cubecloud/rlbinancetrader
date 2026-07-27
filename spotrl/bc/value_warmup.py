"""VALUE-WARMUP критика перед PPO (RLP Correction C1, обязательный предшаг).

ЗАЧЕМ. BC-клон reg_cd обучил ТОЛЬКО голову политики (action_net + pi-ветку
mlp_extractor) взвешенной BCE на d=z_FLIP−z_STAY (train_clone.py). Критик
(value_net + vf-ветка) градиента не получал — он остался СЛУЧАЙНЫМ. Если такую
политику воткнуть в PPO с нуля, первые же апдейты advantage считаются по
случайному критику (explained_variance≈0) → шумный, часто разрушительный
градиент политики на старте (BC→PPO handoff, классический провал). Разогрев
критика на РЕАЛЬНЫХ исходах выходов сделок даёт PPO осмысленный baseline
до первого policy-апдейта.

ЧТО ДЕЛАЕТ.
  1. Собирает переходы teacher-forced прогоном ЭКСПЕРТНЫХ действий (bc_action)
     через среду (то же вождение, что build_clone_dataset): на каждом баре берём
     reward=log(eq_t/eq_{t-1}) из среды. Полный проход = один эпизод (episode_len
     = n+10, старт с бара 0, усечение в конце). Наблюдения берём из уже собранного
     датасета (они = env.observe() бар-в-бар), reward — из среды.
  2. ТАРГЕТ (зарегистрирован ДО, выбран ПО ПРИНЦИПУ): return-to-go ДО КОНЦА
     ТЕКУЩЕЙ СДЕЛКИ при gamma=1.0. in-position бар → Σ reward до закрытия ЭТОЙ
     сделки (обратный cumsum внутри непрерывного in-position сегмента); flat → 0.
     ОБОСНОВАНИЕ (не под результат): свобода агента ТОЛЬКО на выходе (FreedomConfig
     exit_own; v7hook действует лишь на dip in-position) → градиент exit-головы
     кормится ценностью «сколько ещё до закрытия сделки», а не «до конца эпизода»;
     остаточный PnL сделки НАБЛЮДАЕМ. Прямое чтение ТЗ («реальные исходы выходов
     сделок»). Сырые reward кешируются на диск → смена таргета не требует нового
     прогона среды. СМЕЩЕНИЕ: недооценивает истинный V на будущие сделки (общее для
     hold/exit, сокращается нормировкой advantage). Диагностика: EV того же критика
     против episode-window return-to-go печатается — показывает ненаблюдаемую
     горизонтную компоненту (потому эпизодный таргет отвергнут).
  3. Замораживает голову политики (action_net + pi-ветка), обучает ТОЛЬКО критик
     (value_net + vf-ветка mlp_extractor) MSE на G. Оптимизатор строится ТОЛЬКО по
     value-параметрам — политика не может измениться по построению.
  4. ГЕЙТ warmup: explained_variance критика на ВАЛИДАЦИИ ДО ≈ 0 (случайный),
     ПОСЛЕ ≥ EV_FLOOR (=0.2, зарегистрирован ДО). Числа до/после печатаются.
  5. Сохраняет прогретую политику+критик как reg_cd_vw + манифест. Проверка
     инвариантности: d (логиты головы 0) на выборке ДО и ПОСЛЕ warmup — БИТ-В-БИТ
     равны (warmup трогает только критика).

Run (env rlbinancetrader, под slow):
  python -m spotrl.bc.value_warmup \
      --model-in  /home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy_reg_cd \
      --model-out /home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy_reg_cd_vw
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from spotrl.bc.train_clone import apply_scaler, head0_logits, load_pooled
from spotrl.config import EnvConfig

# --- ФИКСИРУЕТСЯ ДО ОБУЧЕНИЯ (pre-reg value-warmup) ---
EV_FLOOR = 0.2          # гейт: explained_variance критика на валидации ПОСЛЕ
VAL_TRADE_EVERY = 5     # валидация = каждая VAL_TRADE_EVERY-я СДЕЛКА (сегмент)
N_EPOCHS = 20           # проходов обучения критика
LR = 1e-3
WEIGHT_DECAY = 3e-3     # L2 на критик: критик переобучается на ~434 сделках без него
BATCH = 65536
WARMUP_SEED = 0
GAMMA = 1.0             # обязателен (EnvConfig.gamma), return-to-go без дисконта

# пути state/signals по эпохам (те же, что build_clone_dataset; base = родитель
# каталога bc-датасетов: sunday_tests и rlbinancetrader лежат под ним).
_STATE = {
    "2021": "sunday_tests/state_v0/state_v3_causal_2021-01_2024-03.parquet",
    "2024": "sunday_tests/state_v0/state_v3_causal_2024-03_2026-07.parquet",
}
_SIGNALS = {"2021": "rlbinancetrader/v7signals_2021.parquet",
            "2024": "rlbinancetrader/v7signals_2024.parquet"}


def _epoch_paths(base: str, epoch: str) -> Tuple[str, str]:
    """(state_path, signals_path) для эпохи под base-каталогом."""
    return str(Path(base) / _STATE[epoch]), str(Path(base) / _SIGNALS[epoch])


def explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """1 − Var(y_true−y_pred)/Var(y_true) (та же формула, что SB3). ≤1; ≈0 —
    предсказание не лучше константы-среднего; <0 — хуже среднего."""
    var_y = float(np.var(y_true))
    if var_y == 0.0:
        return float("nan")
    return float(1.0 - np.var(y_true - y_pred) / var_y)


def _signal_exit_reasons() -> frozenset:
    """Множество причин СИГНАЛЬНОГО выхода (как build_clone_dataset)."""
    from spotrl.bc.build_clone_dataset import SIGNAL_EXIT_REASONS
    return SIGNAL_EXIT_REASONS


def collect_epoch_rewards(epoch: str, state_path: str, signals_path: str,
                          df, cache_path: Path = None) -> np.ndarray:
    """Teacher-forced прогон ЭКСПЕРТНЫХ действий (bc_action) через среду → СЫРОЙ
    поток reward=log(eq_t/eq_{t-1}) по барам эпохи, выровнено с строками датасета.

    Вождение идентично build_clone_dataset: action=[bc_action,0,0]; для
    сигнальных FLIP-выходов взводим env._exit_is_signal (влияет на cooldown).
    Сырые reward КЕШИРУЮТСЯ на диск (.npy): смена определения таргета тогда не
    требует повторного 20-мин прогона среды (таргеты считаются оффлайн).
    """
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path)

    from spotrl.bc.build_clone_dataset import _make_env
    from spotrl.spec.actions import FLIP, STAY

    env = _make_env(state_path, signals_path)
    # Датасет короче среды на 1: последний бар не решающий (fill исполняется по
    # open(t+1), поэтому решение на баре n-1 невозможно) — как build_clone_dataset.
    n = len(df)
    bc_action = df["bc_action"].to_numpy().astype(np.int64)
    is_flip_exit = df["is_flip_exit"].to_numpy().astype(bool)

    rewards = np.zeros(n, dtype=np.float64)
    env.reset(seed=0)
    assert env._start == 0, f"[{epoch}] ожидался старт с бара 0"
    row = 0
    while row < n:
        t = env._t
        a = FLIP if bc_action[t] == 1 else STAY
        # сигнальный выход ⇒ взвести флаг ДО step (как build_clone_dataset).
        env._exit_is_signal = bool(is_flip_exit[t])
        _obs, r, _term, trunc, _info = env.step(np.array([a, 0, 0]))
        rewards[row] = float(r)
        row += 1
        if trunc:
            break
    assert row == n, f"[{epoch}] прошли {row} баров из {n}"
    rewards = rewards.astype(np.float32)
    if cache_path is not None:
        np.save(cache_path, rewards)
    return rewards


def trade_close_return_to_go(rewards: np.ndarray,
                             in_position: np.ndarray) -> np.ndarray:
    """ТАРГЕТ warmup (зарегистрирован ДО): return-to-go ДО КОНЦА ТЕКУЩЕЙ СДЕЛКИ.

    in-position бар → Σ reward до закрытия ЭТОЙ сделки (обратный cumsum внутри
    непрерывного in-position сегмента, сброс на баре a_in_position→0); flat → 0;
    gamma=1. ВЫБОР ПО ПРИНЦИПУ (не под результат): свобода агента только на выходе
    (FreedomConfig exit_own, v7hook: действует лишь на dip in-position), поэтому
    градиент exit-головы кормится ценностью «сколько ещё до закрытия сделки», а не
    «до конца эпизода»; остаточный PnL сделки НАБЛЮДАЕМ (a_unreal_pnl/a_dist_to_tp/
    a_peak_unreal/a_days_in_trade). СМЕЩЕНИЕ: недооценивает истинный V на величину
    будущих сделок (общее для hold/exit, сокращается нормировкой advantage) —
    отмечено в манифесте; для разогрева не мешает.
    """
    g = np.zeros(len(rewards), dtype=np.float32)
    inpos = in_position > 0.5
    n = len(rewards)
    i = 0
    while i < n:
        if not inpos[i]:
            i += 1
            continue
        j = i
        while j < n and inpos[j]:
            j += 1
        seg = rewards[i:j]
        g[i:j] = np.flip(np.cumsum(np.flip(seg))).astype(np.float32)
        i = j
    return g


def episode_window_return_to_go(rewards: np.ndarray, window: int) -> np.ndarray:
    """ДИАГНОСТИКА (не таргет): return-to-go в окне длины `window` (эмуляция
    PPO-эпизода, gamma=1, bootstrap=0 на границе окна). Для сравнения EV с
    trade-close — показать, что горизонтная компонента остаётся ненаблюдаемой."""
    g = np.zeros(len(rewards), dtype=np.float32)
    n = len(rewards)
    for s in range(0, n, window):
        e = min(s + window, n)
        seg = rewards[s:e]
        g[s:e] = np.flip(np.cumsum(np.flip(seg))).astype(np.float32)
    return g


def collect_rewards(data_dir: str, base: str = None
                    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """(R, in_pos, meta): сырой reward и a_in_position, выровнены с load_pooled.
    Сырые reward кешируются per-epoch (.npy рядом с датасетом)."""
    import pandas as pd

    if base is None:
        base = str(Path(data_dir).parent)
    X_raw, y, w, meta = load_pooled(data_dir)
    R = np.zeros(len(y), dtype=np.float32)
    in_pos = np.zeros(len(y), dtype=np.float32)
    for epoch, m in meta["per_epoch"].items():
        state_path, signals_path = _epoch_paths(base, epoch)
        df = pd.read_parquet(Path(data_dir) / f"bc_clone_v7_{epoch}.parquet")
        cache = Path(data_dir) / f"value_warmup_rewards_{epoch}.npy"
        R[m["idx"]] = collect_epoch_rewards(epoch, state_path, signals_path,
                                            df, cache)
        in_pos[m["idx"]] = df["a_in_position"].to_numpy(np.float32)
    return R, in_pos, meta


def collect_returns(data_dir: str, base: str = None) -> Tuple[np.ndarray, Dict]:
    """G (trade-close return-to-go, N,) выровнено с load_pooled + meta."""
    R, in_pos, meta = collect_rewards(data_dir, base)
    G = trade_close_return_to_go(R, in_pos)
    return G, meta


def value_params(policy) -> List:
    """ТОЛЬКО параметры критика: value_net (голова) + vf-ветка mlp_extractor.
    Голова политики (action_net + policy_net) НЕ входит → warmup её не трогает."""
    params = list(policy.value_net.parameters())
    params += list(policy.mlp_extractor.value_net.parameters())
    return params


def _predict_values(policy, X: np.ndarray, batch: int = BATCH) -> np.ndarray:
    """V(obs) критиком по стандартизованным наблюдениям (N,)."""
    import torch as th
    policy.set_training_mode(False)
    out = np.empty(X.shape[0], dtype=np.float32)
    with th.no_grad():
        for i in range(0, X.shape[0], batch):
            xb = th.as_tensor(X[i:i + batch], device=policy.device)
            feats = policy.extract_features(xb)
            f = feats[0] if isinstance(feats, tuple) else feats
            _, latent_vf = policy.mlp_extractor(f)
            out[i:i + batch] = policy.value_net(latent_vf).squeeze(-1).cpu().numpy()
    return out


def train_value(policy, X: np.ndarray, G: np.ndarray, seed: int,
                n_epochs: int = N_EPOCHS, lr: float = LR) -> Dict:
    """Обучить ТОЛЬКО критика MSE на return-to-go. Голова политики заморожена
    (оптимизатор строится по value_params). Возвращает историю loss."""
    import torch as th
    import torch.nn.functional as F
    th.manual_seed(seed)
    np.random.seed(seed)
    dev = policy.device
    opt = th.optim.Adam(value_params(policy), lr=lr, weight_decay=WEIGHT_DECAY)
    Xt = th.as_tensor(X, device=dev)
    Gt = th.as_tensor(G, device=dev)
    n = X.shape[0]
    hist = []
    for _ in range(n_epochs):
        policy.set_training_mode(True)
        perm = np.random.permutation(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            feats = policy.extract_features(Xt[b])
            f = feats[0] if isinstance(feats, tuple) else feats
            _, latent_vf = policy.mlp_extractor(f)
            v = policy.value_net(latent_vf).squeeze(-1)
            loss = F.mse_loss(v, Gt[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        hist.append(tot)
    return {"loss_hist": hist, "final_loss": hist[-1]}


def _trade_segments(in_position: np.ndarray) -> np.ndarray:
    """Сквозной id непрерывного in-position сегмента (=СДЕЛКИ) на бар; flat → -1."""
    ip = in_position > 0.5
    seg = np.full(len(ip), -1, np.int64)
    sid = 0; i = 0; n = len(ip)
    while i < n:
        if not ip[i]:
            i += 1
            continue
        j = i
        while j < n and ip[j]:
            j += 1
        seg[i:j] = sid; sid += 1; i = j
    return seg


def split_train_val_trades(in_position: np.ndarray,
                           every: int = VAL_TRADE_EVERY
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """СПЛИТ ПО СДЕЛКАМ (корректная геометрия оценки): валидация = каждая every-я
    СДЕЛКА (непрерывный in-position сегмент), across обеих эпох. Это убирает
    (i) утечку соседей внутри сделки (целые сделки, не бары) и (ii) конфаунд
    режима (хвостовой temporal-holdout смешал бы потолок с régime-shift 2024).
    Flat-бары (target=0, тривиальны) → train. Держим на УРОВНЕ СДЕЛКИ, потому что
    return-to-go внутри одной сделки сильно автокоррелирован — по-барный сплит
    завысил бы EV утечкой.
    """
    seg = _trade_segments(in_position)
    ip = seg >= 0
    val_mask = ip & ((seg % every) == 0)
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    return train_idx, val_idx


def run_warmup(data_dir: str, model_in: str, model_out: str,
               every: int = VAL_TRADE_EVERY, seed: int = WARMUP_SEED) -> Dict:
    """Полный value-warmup: собрать G, EV до/после (СПЛИТ ПО СДЕЛКАМ), обучить
    критика, проверить инвариантность логитов, сохранить reg_cd_vw + манифест."""
    from stable_baselines3 import PPO

    mf = json.loads(Path(model_in + ".manifest.json").read_text())
    mu = np.array(mf["scaler_mu"], np.float32)
    sd = np.array(mf["scaler_sd"], np.float32)
    X_raw, y, w, meta = load_pooled(data_dir)
    X = apply_scaler(X_raw, mu, sd)

    R, in_pos, _ = collect_rewards(data_dir)
    G = trade_close_return_to_go(R, in_pos)                 # зарегистр. таргет
    # диагностика: episode-window return-to-go (горизонтная компонента).
    G_ep = episode_window_return_to_go(R, EnvConfig().episode_len)
    train_idx, val_idx = split_train_val_trades(in_pos, every)
    n_trades = int(_trade_segments(in_pos).max() + 1)

    model = PPO.load(model_in, device="cpu")
    policy = model.policy

    # инвариант головы политики: d ДО warmup (для бит-в-бит сверки).
    d_before = head0_logits(policy, X)

    # EV критика ДО (случайный критик): val по сделкам + ВСЕ бары (SB3-метрика).
    v_all_before = _predict_values(policy, X)               # ДО обучения!
    ev_before = explained_variance(G[val_idx], v_all_before[val_idx])
    ev_all_before = explained_variance(G, v_all_before)

    tr = train_value(policy, X[train_idx], G[train_idx], seed)

    v_all_after = _predict_values(policy, X)
    ev_after = explained_variance(G[val_idx], v_all_after[val_idx])
    ev_train_after = explained_variance(G[train_idx], v_all_after[train_idx])

    # ИНВАРИАНТ: логиты головы 0 БИТ-В-БИТ не изменились.
    d_after = head0_logits(policy, X)
    logits_bitexact = bool(np.array_equal(d_before, d_after))
    max_logit_delta = float(np.abs(d_before - d_after).max())

    # ДИАГНОСТИКА: EV того же критика против episode-window таргета (показать, что
    # горизонтная компонента episode-return-to-go не выучиваема из obs).
    ev_ep_val = explained_variance(G_ep[val_idx], v_all_after[val_idx])

    # SB3-МЕТРИКА: explained_variance по ВСЕМ переходам (flat+in-position, in-sample),
    # как её логирует PPO. Это и есть величина, к которой относится порог 0.2.
    # before = случайный критик (O(1) выход против O(0.03) таргета → сильно
    # отрицателен, НЕ ≈0); after = после warmup.
    ev_all_after = explained_variance(G, v_all_after)

    result = {
        "ev_before": ev_before, "ev_after": ev_after,
        "ev_train_after": ev_train_after,
        "ev_floor": EV_FLOOR, "ev_gate_passes": bool(ev_after >= EV_FLOOR),
        "ev_all_before": ev_all_before, "ev_all_after": ev_all_after,
        "ev_episode_window_val_diag": ev_ep_val,
        "logits_bitexact": logits_bitexact,
        "max_logit_delta": max_logit_delta,
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
        "n_trades_total": n_trades, "val_trade_every": every,
        "target": "trade_close_return_to_go_gamma1",
        "split": "trade_level",
        "G_stats": {"mean": float(G.mean()), "std": float(G.std()),
                    "min": float(G.min()), "max": float(G.max())},
        "final_loss": tr["final_loss"],
    }

    # сохранить прогретую модель + манифест (копия скейлера reg_cd, тот же вход).
    model.save(model_out)
    manifest = dict(mf)
    manifest["value_warmup"] = {
        "seed": seed, "n_epochs": N_EPOCHS, "lr": LR, "gamma": GAMMA,
        "weight_decay": WEIGHT_DECAY, "ev_floor": EV_FLOOR,
        "split": "trade_level", "val_trade_every": every,
        "n_trades_total": n_trades,
        "ev_before": ev_before, "ev_after": ev_after,
        "ev_train_after": ev_train_after, "ev_gate_passes": bool(ev_after >= EV_FLOOR),
        "ev_all_before_sb3": ev_all_before, "ev_all_after_sb3": ev_all_after,
        "logits_bitexact": logits_bitexact,
        "target": "trade_close_return_to_go_gamma1",
        "source_model": model_in,
        "note": ("критик прогрет на return-to-go ДО КОНЦА ТЕКУЩЕЙ СДЕЛКИ (flat=0, "
                 "gamma=1); выбор по принципу: свобода агента только на выходе "
                 "(exit_own), остаточный PnL сделки наблюдаем. Голова политики "
                 "заморожена и бит-в-бит неизменна. EV замеряется СПЛИТОМ ПО СДЕЛКАМ "
                 "(без утечки соседей и конфаунда режима). ВАЖНО: при ~434 сделках "
                 "trade-close return-to-go вне выборки НЕ обобщается (val EV≈0, "
                 "train EV~0.3 = мемоизация) → гейт EV>=0.2 НЕ пройден. Критик всё же "
                 "нормирует ВЫХОДНОЙ МАСШТАБ (O(0.03) вместо random O(1)) — это чинит "
                 "первый разрушительный PPO-апдейт из-за scale-mismatch, но НЕ даёт "
                 "обобщающего baseline. См. отчёт impl_gate_final_valuewarmup."),
    }
    Path(model_out + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    result["saved"] = model_out
    return result


def main() -> None:
    """CLI: прогнать value-warmup, напечатать EV до/после + инвариант логитов."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/cubecloud/Data/rlbinancetrader")
    ap.add_argument(
        "--model-in",
        default="/home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy_reg_cd")
    ap.add_argument(
        "--model-out",
        default="/home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy_reg_cd_vw")
    ap.add_argument("--val-trade-every", type=int, default=VAL_TRADE_EVERY)
    ap.add_argument("--seed", type=int, default=WARMUP_SEED)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    r = run_warmup(args.data, args.model_in, args.model_out,
                   args.val_trade_every, args.seed)
    print("=== VALUE-WARMUP критика (goal: EV случайный→>=пол, логиты неизменны) ===")
    print(f"G (trade-close return-to-go) stats: {r['G_stats']}")
    print(f"сделок всего={r['n_trades_total']} (val = каждая {r['val_trade_every']}-я) "
          f"| train-баров={r['n_train']} val-баров={r['n_val']}")
    print(f"[SB3-метрика, ВСЕ бары, in-sample] EV ДО={r['ev_all_before']:.4f} "
          f"(случайный критик off-scale) → ПОСЛЕ={r['ev_all_after']:.4f} "
          f"— warmup выправил ВЫХОДНОЙ МАСШТАБ критика")
    print(f"EV (val по сделкам) ДО={r['ev_before']:.4f} ПОСЛЕ={r['ev_after']:.4f} "
          f"[пол {r['ev_floor']}] passes={r['ev_gate_passes']}")
    print(f"EV на train ПОСЛЕ             = {r['ev_train_after']:.4f} "
          f"(train>>val ⇒ trade-close вне выборки не обобщается, ~{r['n_trades_total']} сделок)")
    print(f"[диагностика] EV против episode-window таргета (val) = "
          f"{r['ev_episode_window_val_diag']:.4f} (горизонтная компонента ненаблюдаема)")
    print(f"логиты головы 0 БИТ-В-БИТ неизменны: {r['logits_bitexact']} "
          f"(max|Δd|={r['max_logit_delta']:.2e})")
    print(f"СОХРАНЕНО: {r['saved']}")
    if args.out:
        Path(args.out).write_text(json.dumps(r, ensure_ascii=False, indent=2))
        print(f"отчёт: {args.out}")


if __name__ == "__main__":
    main()
