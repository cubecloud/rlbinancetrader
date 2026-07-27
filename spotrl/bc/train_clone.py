"""Обучение клона v7 (BC, supervised) головы 0 (STAY/FLIP) на наблюдении v2.

Что делает (ШАГ 2 pre-reg BC). Обучает сеть политики sb3-PPO (`MlpPolicy`,
пространство действий копи-спека `MultiDiscrete([2,1,1])`) предсказывать действие
головы 0 по наблюдению v2 взвешенным CE. Головы SL/TP в копировании ВЫРОЖДЕНЫ
(nvec=1 → log_prob≡0, entropy≡0) и дополнительно занулены loss-маской
(`algo/head_masking`), поэтому в BC не учатся — как и требует pre-reg.

Почему именно сеть sb3-PPO, а не выкидной sklearn: обучается ТА сеть, что засеет
PPO на этапе RL (`model.save`). Голова 0 = первые 2 логита `action_net`
(MultiDiscrete split [2,1,1]).

Дисциплина сдвига логита (pre-reg §2, редакция 2026-07-27). Сдвиг s добавляется к
логиту FLIP: p(FLIP)=sigmoid(d+s), d=logit_FLIP−logit_STAY. ПРАВИЛО выбора s
зафиксировано ДО замера гейта как детерминированная функция train-распределения:
  s_lo = min s: recall_flip (per-epoch min, ОБЕ стороны) >= 0.995;
  s_hi = max s: FPR (per-epoch max) <= 3e-7;
  s* = (s_lo + s_hi)/2, если s_lo<=s_hi (иначе скалярный сдвиг НЕ существует).
Калибровочная выборка = обучающая (эпохи смешаны для BC, истинного holdout нет —
честно помечено; этот гейт = воспроизведение v7 на ТОЙ ЖЕ ленте, что и есть
бар-в-бар гейт). s НЕ сканируется под прохождение гейта — это середина
допустимого интервала.

Дисбаланс FLIP (~2.5e-4): все FLIP-бары (их 808) включаются в каждый шаг, STAY —
миничанками + глобальные hard-negatives. Обучение детерминированное по сиду.

Run (env rlbinancetrader):
  python -m spotrl.bc.train_clone --data /home/cubecloud/Data/rlbinancetrader \
      --out /home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# --- ФИКСИРУЕТСЯ ДО ОБУЧЕНИЯ (pre-reg) ---
FPR_TARGET = 3e-7           # порог ложного FLIP на бар (обе эпохи)
RECALL_FLOOR = 0.995        # recall_flip per-epoch, вход и выход раздельно
N_EPOCHS = 25               # проходов обучения (зазор>=20 nat достигается к ~15)
LR = 5e-4
BATCH = 65536               # размер STAY-миничанка
HIDDEN = (256, 256)         # архитектура MLP политики (net_arch pi)
TRAIN_SEED = 0

# --- WORST-CASE (max-margin) лосс на d = logit_FLIP − logit_STAY (ФИКС. ДО обуч.)
# ГЛАВНОЕ (эмпирика 2026-07-27): гейт определяется ХУДШИМ баром (min_FLIP, max_STAY),
# т.е. это L∞/max-margin задача, НЕ средняя. Взвешенный BCE и СРЕДНИЙ маржинальный
# лосс дают пересекающиеся хвосты (зазор −7…−10 nat при отделимых данных, L2-маржа
# 0.95, argmax при этом ИДЕАЛЕН). Worst-case лосс через logsumexp (гладкий max) +
# глобальный hard-negative mining прямо максимизирует зазор:
#   L = logsumexp_FLIP(M − d) + logsumexp_STAY∪HN(M + d)
# logsumexp≈max → минимизирует (M − min_FLIP d) и (M + max_STAY d) → тянет
# min_FLIP(d) вверх и max_STAY(d) вниз. HN = HN_SIZE глобально худших STAY-баров
# (пересчёт каждую эпоху), включаются в КАЖДЫЙ шаг — иначе per-minibatch max играет
# в whack-a-mole и глобальный max_STAY не давится. Достигнутый зазор ~28 nat >
# нужных 20.3 (обе эпохи). M/HN_SIZE фиксируются ДО обучения.
MARGIN_M = 12.0
HN_SIZE = 16384             # глобально худших STAY на шаг (hard-negative mining)

OBS_PREFIXES = ("m_", "a_", "w_", "rule_")
CLIP_STD = 10.0             # клип стандартизованного входа


def obs_columns(df) -> list:
    """38 колонок наблюдения в порядке spec (префиксы m_/a_/w_/rule_)."""
    return [c for c in df.columns if c.startswith(OBS_PREFIXES)]


def fit_scaler(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Статистики стандартизации входа (mu, sd) по пулу train. sd-пол 1e-6.

    Стандартизация обязательна: сырые признаки разного масштаба (m_leg_age до
    ~1158 против булевых) иначе доминируют первый слой и душат маржу логита.
    Статистики ФИКСИРУЮТСЯ и сохраняются в манифест — на serve/RL применять те же
    (иначе разрыв train/serve). Константные признаки (rule_slot) → sd=1 (нейтраль).
    """
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_scaler(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Стандартизовать и клипнуть вход (тот же путь на train и serve)."""
    return np.clip((X - mu) / sd, -CLIP_STD, CLIP_STD).astype(np.float32)


@dataclass
class ShiftCalibration:
    """Результат калибровки скалярного сдвига логита (pre-reg §2)."""

    s_lo: float
    s_hi: float
    s_star: float
    feasible: bool
    gap_nat: float               # min_FLIP(d) − max_STAY(d), пул обеих эпох
    gap_needed: float            # logit(recall)−logit(FPR) ≈ 20.3 nat


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def load_pooled(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Пул обеих эпох для обучения BC (смешивание допустимо, pre-reg §6).

    Returns:
        X (N,38) float32, y (N,) int (голова 0), w (N,) веса bc_weight,
        per_epoch: словарь эпоха -> индексы/маски для раздельного замера гейта.
    """
    import pandas as pd
    frames = []
    for epoch in ("2021", "2024"):
        df = pd.read_parquet(Path(data_dir) / f"bc_clone_v7_{epoch}.parquet")
        frames.append(df)
    cols = obs_columns(frames[0])
    Xs, ys, ws, meta = [], [], [], {}
    off = 0
    for epoch, df in zip(("2021", "2024"), frames):
        X = df[cols].to_numpy(np.float32)
        y = df["bc_action"].to_numpy().astype(np.int64)
        w = df["bc_weight"].to_numpy(np.float32)
        idx = np.arange(off, off + len(df))
        meta[epoch] = {
            "idx": idx,
            "is_entry": df["is_flip_entry"].to_numpy().astype(bool),
            "is_exit": df["is_flip_exit"].to_numpy().astype(bool),
            "is_stay": (y == 0),
        }
        Xs.append(X); ys.append(y); ws.append(w); off += len(df)
    return (np.concatenate(Xs), np.concatenate(ys),
            np.concatenate(ws), {"per_epoch": meta, "cols": cols})


def build_policy(obs_dim: int, seed: int):
    """Собрать sb3-PPO политику копи-спека; вернуть (model, policy)."""
    import gymnasium as gym
    import torch as th
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    class _Spaces(gym.Env):
        """Пустая среда только ради пространств (шаги не вызываются)."""

        def __init__(self):
            """Задать пространства obs/action копи-спека."""
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            self.action_space = gym.spaces.MultiDiscrete(np.array([2, 1, 1]))

        def reset(self, *, seed=None, options=None):
            """Сброс-заглушка (шаги реально не вызываются)."""
            return np.zeros(obs_dim, np.float32), {}

        def step(self, action):
            """Шаг-заглушка (обучение BC идёт вне среды)."""
            return np.zeros(obs_dim, np.float32), 0.0, True, False, {}

    env = make_vec_env(_Spaces, n_envs=1, seed=seed)
    model = PPO("MlpPolicy", env, seed=seed, device="cpu",
                policy_kwargs=dict(net_arch=dict(pi=list(HIDDEN), vf=list(HIDDEN))))
    return model, model.policy


def head0_logits(policy, X: np.ndarray, batch: int = 65536) -> np.ndarray:
    """Прогон наблюдений через политику → логиты головы 0 (N,2)."""
    import torch as th
    policy.set_training_mode(False)
    out = np.empty((X.shape[0], 2), dtype=np.float32)
    with th.no_grad():
        for i in range(0, X.shape[0], batch):
            xb = th.as_tensor(X[i:i + batch], device=policy.device)
            feats = policy.extract_features(xb)
            if isinstance(feats, tuple):
                latent_pi, _ = policy.mlp_extractor(feats[0])
            else:
                latent_pi, _ = policy.mlp_extractor(feats)
            logits = policy.action_net(latent_pi)   # (B, sum(nvec)=4)
            out[i:i + batch] = logits[:, :2].cpu().numpy()
    return out


def _d_of(policy, xb):
    """d = logit_FLIP − logit_STAY для тензора наблюдений (с градиентом)."""
    feats = policy.extract_features(xb)
    latent_pi, _ = policy.mlp_extractor(
        feats[0] if isinstance(feats, tuple) else feats)
    logits = policy.action_net(latent_pi)[:, :2]
    return logits[:, 1] - logits[:, 0]


def _all_stay_d(policy, Xt, stay_idx) -> np.ndarray:
    """Логит-разница d по ВСЕМ STAY-барам (для выбора глобальных hard-negatives)."""
    import torch as th
    policy.set_training_mode(False)
    out = np.empty(len(stay_idx), dtype=np.float32)
    with th.no_grad():
        for i in range(0, len(stay_idx), 200000):
            b = stay_idx[i:i + 200000]
            out[i:i + len(b)] = _d_of(policy, Xt[b]).cpu().numpy()
    return out


def train(policy, X, y, w, seed: int, n_epochs: int = N_EPOCHS,
          lr: float = LR) -> Dict:
    """Worst-case (max-margin) обучение головы 0 через logsumexp + hard-neg mining.

    Головы SL/TP вырождены (nvec=1 → log_prob≡0) и loss-маскированы (no-op) — в BC
    не учатся. Бары с bc_weight=0 (анти-каузальный forced_eod STAY, 1 строка)
    исключаются из обучения.
    """
    import torch as th
    th.manual_seed(seed)
    np.random.seed(seed)
    dev = policy.device
    opt = th.optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-6)
    Xt = th.as_tensor(X, device=dev)
    keep = w > 0.0
    flip_idx = np.where((y == 1) & keep)[0]
    stay_idx = np.where((y == 0) & keep)[0]
    Xf = Xt[flip_idx]
    hist = []
    for epoch in range(n_epochs):
        sd_all = _all_stay_d(policy, Xt, stay_idx)
        hard = stay_idx[np.argsort(sd_all)[-HN_SIZE:]]      # глобально худшие STAY
        policy.set_training_mode(True)
        perm = np.random.permutation(stay_idx)
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            d_flip = _d_of(policy, Xf)                       # все FLIP
            d_stay = _d_of(policy, Xt[np.concatenate([b, hard])])
            loss = (th.logsumexp(MARGIN_M - d_flip, 0)
                    + th.logsumexp(MARGIN_M + d_stay, 0))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        hist.append(tot)
    return {"loss_hist": hist, "final_loss": hist[-1]}


def calibrate_shift(d_by_epoch: Dict[str, Dict[str, np.ndarray]]) -> ShiftCalibration:
    """Найти s-интервал и s* по ПРАВИЛУ pre-reg (детерминированно).

    d_by_epoch[epoch] = {"stay": d[stay], "flip_all": d[entry|exit]}.
    recall/FPR при сдвиге s считаются как средние sigmoid(d+s) (ожидание доли
    сэмплированных FLIP — точный per-decision предел при бесконечных сидах).
    """
    def sig(x):
        """Сигмоида (вероятность FLIP при сдвиге логита)."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))

    # Сетка s: достаточно плотная в диапазоне, покрывающем оба порога.
    all_d = np.concatenate([np.concatenate([v["stay"], v["flip_all"]])
                            for v in d_by_epoch.values()])
    grid = np.linspace(all_d.min() - 40, all_d.max() + 40, 20001)

    recall_ok = np.ones_like(grid, dtype=bool)
    fpr_ok = np.ones_like(grid, dtype=bool)
    for v in d_by_epoch.values():
        rec = np.array([sig(v["flip_all"] + s).mean() for s in grid])
        fpr = np.array([sig(v["stay"] + s).mean() for s in grid])
        recall_ok &= (rec >= RECALL_FLOOR)
        fpr_ok &= (fpr <= FPR_TARGET)
    # recall растёт с s, FPR тоже растёт с s → recall_ok = s>=s_lo, fpr_ok = s<=s_hi.
    s_lo = float(grid[recall_ok][0]) if recall_ok.any() else float("inf")
    s_hi = float(grid[fpr_ok][-1]) if fpr_ok.any() else float("-inf")
    feasible = s_lo <= s_hi
    s_star = (s_lo + s_hi) / 2 if feasible else float("nan")

    gap = min(v["flip_all"].min() - v["stay"].max() for v in d_by_epoch.values())
    return ShiftCalibration(s_lo=s_lo, s_hi=s_hi, s_star=s_star, feasible=feasible,
                            gap_nat=float(gap),
                            gap_needed=_logit(RECALL_FLOOR) - _logit(FPR_TARGET))


def main() -> None:
    """CLI: обучить клон BC на пуле обеих эпох, сохранить политику + манифест."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/cubecloud/Data/rlbinancetrader")
    ap.add_argument("--out", default="/home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy")
    ap.add_argument("--seed", type=int, default=TRAIN_SEED)
    args = ap.parse_args()

    X_raw, y, w, meta = load_pooled(args.data)
    mu, sd = fit_scaler(X_raw)
    X = apply_scaler(X_raw, mu, sd)
    print(f"пул: {X.shape}, FLIP={int((y == 1).sum())}, M={MARGIN_M} HN={HN_SIZE}")
    model, policy = build_policy(X.shape[1], args.seed)
    tr = train(policy, X, y, w, args.seed)
    print(f"обучение: final_loss={tr['final_loss']:.4e}")

    logits = head0_logits(policy, X)
    d = logits[:, 1] - logits[:, 0]
    d_by_epoch = {}
    for epoch, m in meta["per_epoch"].items():
        di = d[m["idx"]]
        d_by_epoch[epoch] = {
            "stay": di[m["is_stay"]],
            "flip_all": di[m["is_entry"] | m["is_exit"]],
        }
    cal = calibrate_shift(d_by_epoch)
    print(f"зазор d = {cal.gap_nat:.2f} nat (нужно >= {cal.gap_needed:.2f}); "
          f"s_lo={cal.s_lo:.3f} s_hi={cal.s_hi:.3f} s*={cal.s_star:.3f} "
          f"feasible={cal.feasible}")

    out = Path(args.out)
    model.save(str(out))
    manifest = {
        "seed": args.seed, "margin_m": MARGIN_M, "hn_size": HN_SIZE,
        "fpr_target": FPR_TARGET,
        "recall_floor": RECALL_FLOOR, "n_epochs": N_EPOCHS, "lr": LR,
        "hidden": list(HIDDEN), "final_loss": tr["final_loss"],
        "shift": cal.__dict__, "obs_cols": meta["cols"],
        "clip_std": CLIP_STD,
        "scaler_mu": [float(x) for x in mu],
        "scaler_sd": [float(x) for x in sd],
    }
    Path(str(out) + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"СОХРАНЕНО: {out}.zip + манифест")


if __name__ == "__main__":
    main()
