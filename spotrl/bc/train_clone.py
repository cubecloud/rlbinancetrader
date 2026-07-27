"""Обучение клона v7 (BC, supervised) головы 0 (STAY/FLIP) на наблюдении v2.

ЗАХОД «b» (2026-07-27): наблюдение v2 ДОСТАТОЧНО (диагностика
`diag_obs_sufficiency_2026-07-26.md`: семантика правила v7 разделяет классы
out-of-epoch, LR AUC вход ~0.98, выход ~0.94). Прежний провал клона был не
MDP-недостаточностью, а ОБЪЕКТИВОМ: ±80 logsumexp/hard-negative max-margin
мемоизировал шум (зазор 87 nat при нужных 20.3, логиты ±80, FPR≈1e-23,
out-of-epoch зазор −110/−158). Этот модуль заменяет тот объектив на НОРМАЛЬНЫЙ
регуляризованный.

Что делает. Обучает сеть политики sb3-PPO (`MlpPolicy`, копи-спек
`MultiDiscrete([2,1,1])`) предсказывать действие головы 0 по наблюдению v2
ВЗВЕШЕННОЙ кросс-энтропией с label smoothing. Головы SL/TP в копировании
ВЫРОЖДЕНЫ (nvec=1 → log_prob≡0) и loss-маскированы — в BC не учатся (pre-reg §7).

Регуляризация (всё ФИКСИРУЕТСЯ ДО обучения, pre-reg):
  * взвешенная BCE-with-logits на d=z_FLIP−z_STAY: класс FLIP редок (~2.5e-4),
    вес w_pos=n_stay/n_flip выравнивает вклад классов (иначе коллапс в «всегда
    STAY»);
  * label smoothing eps=1e-5 → мягкий КАП уверенного логита |d|≈ln((1−eps)/eps)
    ≈11.5 nat. Это (i) не даёт гиперуверенности (p(ложн.FLIP)≈6e-6, а НЕ 1e-23 —
    политика годна под PPO: разведка жива, градиент не мёртв), (ii) оставляет
    сэмплированный гейт ДОСТИЖИМЫМ: 2·11.5=23 > нужных 20.3 nat, если данные
    разделимы in-sample. eps выбран по PPO-безопасности ДО замера гейта, НЕ под
    его прохождение; после обучения печатается max|d| — КАП обязан связывать;
  * реальный weight_decay=1e-4 (прежний 1e-6 был no-op → логиты уходили в ±80);
  * клип нормы градиента 5.0.

Дисциплина сдвига логита (pre-reg §2, редакция 2026-07-27, FPR-anchored). Сдвиг s
добавляется к d: p(FLIP)=sigmoid(d+s). ПРАВИЛО (детерминированное, ДО замера):
  s_hi = max s: FPR (per-epoch max, ожидание sigmoid(stay+s)) <= FPR_TARGET;
  s_lo = min s: recall (per-epoch min, вход и выход) >= RECALL_FLOOR;
  s* = s_hi  (ЯКОРЬ на цель FPR — всегда определён; recall — ИСХОД, не подгон).
  feasible = s_lo <= s_hi (существует s с ОБОИМИ порогами одновременно).
Калибровочная выборка = обучающая (эпохи смешаны для BC, истинного holdout нет —
честно; это воспроизведение v7 на ТОЙ ЖЕ ленте = операционный копировальный гейт).
s НЕ сканируется под прохождение гейта.

Run (env rlbinancetrader):
  python -m spotrl.bc.train_clone --data /home/cubecloud/Data/rlbinancetrader \
      --out /home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy_reg
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
N_EPOCHS = 40               # проходов обучения
LR = 5e-4
BATCH = 65536               # размер STAY-миничанка на шаг
HIDDEN = (256, 256)         # архитектура MLP политики (net_arch pi)
TRAIN_SEED = 0

# --- РЕГУЛЯРИЗАЦИЯ (взамен ±80 logsumexp/hard-negative; ФИКС. ДО обучения) ---
LABEL_SMOOTH = 1e-5         # eps: мягкий кап |d|≈ln((1-eps)/eps)≈11.5 nat
WEIGHT_DECAY = 1e-4         # реальный L2 (прежний 1e-6 был no-op)
GRAD_CLIP = 5.0             # клип нормы градиента
# Баланс классов = МАКРО-усреднение: loss = FLIP_WEIGHT*mean_FLIP + mean_STAY.
# Термы уже НОРМИРОВАНЫ на размер класса (mean), поэтому при FLIP_WEIGHT=1 оба
# класса дают равный вклад — это и есть баланс редкого FLIP (2.5e-4) без коллапса
# в «всегда STAY». Домножать ещё на n_stay/n_flip (~3600) НЕЛЬЗЯ — это двойной
# учёт баланса и коллапс в «всегда FLIP». FLIP_WEIGHT фиксируется ДО обучения.
FLIP_WEIGHT = 1.0
CAP_D = float(np.log((1.0 - LABEL_SMOOTH) / LABEL_SMOOTH))  # ~11.51 nat

OBS_PREFIXES = ("m_", "a_", "w_", "rule_")
CLIP_STD = 10.0             # клип стандартизованного входа

# --- ПРОИЗВОДНЫЕ ВХОДНЫЕ ПРИЗНАКИ КЛОНА (input-pipeline, НЕ состав obs-спеки) ---
# Фикс 2026-07-27 «остаток паузы»: решающий для входа сигнал — ПОРОГ РОВНО В НУЛЕ
# признака `a_cooldown_remain` (все 434 истинных входа cd==0 ровно; все 28 спорных
# gate-баров cd в [0.0024,0.0165] — дозревающая пауза, кратная 1/423≈0.00236). При
# глобальной стандартизации (mu≈0.0061, sd≈0.0636) активный хвост сжимается до
# ~0.1–0.3σ и тонет среди unit-масштабного шума 37 других осей; при сырой подаче
# резкий порог у нуля требует веса первого слоя ~5500 (штраф weight_decay
# 1e-4·5500²≈3e3 — обучение туда не идёт). Явный БУЛЕВ флаг делает порог линейно
# разделимым: вклад CAP_D≈11.5 при входе 1 требует веса ~O(11), штраф wd≈0.012 —
# ничтожен. Флаг — ЧИСТАЯ детерминированная функция уже наблюдаемого
# `a_cooldown_remain` (тот же код на serve), НОВОЙ информации/утечки нет. Сам
# остаток `a_cooldown_remain` при этом СОХРАНЯЕТСЯ (флаг + остаток = вариант Б).
# Флаг ≡0 на всех входах без паузы и на in-position с cd==0 → выходная сторона и
# ёмкость не затрагиваются.
EXTRA_COLS = ("a_cooldown_active",)


def obs_columns(df) -> list:
    """38 колонок наблюдения в порядке spec (префиксы m_/a_/w_/rule_)."""
    return [c for c in df.columns if c.startswith(OBS_PREFIXES)]


def derive_extra(df) -> np.ndarray:
    """Производные клон-признаки (N, len(EXTRA_COLS)); порядок = EXTRA_COLS.

    Сейчас единственный — `a_cooldown_active = (a_cooldown_remain > 0)`: булев
    индикатор активной паузы. Тест `>0` точен — активные значения кратны
    1/423≈0.00236 (>0 строго), неактивные ровно 0.0, float-шума у нуля нет.
    """
    cd = df["a_cooldown_remain"].to_numpy(np.float32)
    cd_active = (cd > 0.0).astype(np.float32)
    return cd_active.reshape(-1, 1)


def fit_scaler(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Статистики стандартизации входа (mu, sd) по пулу train. sd-пол 1e-6.

    Стандартизация обязательна: сырые признаки разного масштаба (m_leg_age до
    ~1158 против булевых) иначе доминируют первый слой. Статистики ФИКСИРУЮТСЯ и
    сохраняются в манифест — на serve/RL применять те же (иначе разрыв
    train/serve). Константные признаки → sd=1 (нейтраль).
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
    """Результат FPR-anchored калибровки скалярного сдвига логита (pre-reg §2)."""

    s_lo: float                  # min s: recall >= RECALL_FLOOR (per-epoch)
    s_hi: float                  # max s: FPR <= FPR_TARGET (per-epoch)
    s_star: float                # = s_hi (якорь на FPR, всегда определён)
    feasible: bool               # s_lo <= s_hi (оба порога сразу достижимы)
    gap_nat: float               # min_FLIP(d) − max_STAY(d), пул обеих эпох
    gap_needed: float            # logit(recall)−logit(FPR) ≈ 20.3 nat


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def load_pooled(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Пул обеих эпох для обучения BC (смешивание допустимо, pre-reg §6).

    Returns:
        X (N,38) float32, y (N,) int (голова 0), w (N,) веса bc_weight,
        meta: per_epoch словарь эпоха -> индексы/маски (is_entry/is_exit/is_stay/
        in_pos) для раздельного замера гейта и кросс-эпоховой AUC.
    """
    import pandas as pd
    frames = []
    for epoch in ("2021", "2024"):
        df = pd.read_parquet(Path(data_dir) / f"bc_clone_v7_{epoch}.parquet")
        frames.append(df)
    cols = obs_columns(frames[0])
    cols_out = list(cols) + list(EXTRA_COLS)   # 38 spec + производные клона
    Xs, ys, ws, meta = [], [], [], {}
    off = 0
    for epoch, df in zip(("2021", "2024"), frames):
        X = np.concatenate([df[cols].to_numpy(np.float32), derive_extra(df)],
                           axis=1)
        y = df["bc_action"].to_numpy().astype(np.int64)
        w = df["bc_weight"].to_numpy(np.float32)
        idx = np.arange(off, off + len(df))
        in_pos = (df["a_in_position"].to_numpy() > 0.5
                  if "a_in_position" in df.columns
                  else np.zeros(len(df), bool))
        meta[epoch] = {
            "idx": idx,
            "is_entry": df["is_flip_entry"].to_numpy().astype(bool),
            "is_exit": df["is_flip_exit"].to_numpy().astype(bool),
            "is_stay": (y == 0),
            "in_pos": in_pos,
        }
        Xs.append(X); ys.append(y); ws.append(w); off += len(df)
    return (np.concatenate(Xs), np.concatenate(ys),
            np.concatenate(ws), {"per_epoch": meta, "cols": cols_out})


def build_policy(obs_dim: int, seed: int):
    """Собрать sb3-PPO политику копи-спека; вернуть (model, policy)."""
    import gymnasium as gym
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


def train(policy, X, y, w, seed: int, n_epochs: int = N_EPOCHS,
          lr: float = LR) -> Dict:
    """Регуляризованное обучение головы 0: взвешенная BCE(d) + label smoothing.

    Объектив (взамен ±80 logsumexp): BCE-with-logits на d=z_FLIP−z_STAY с
    целью t = y*(1−eps) + (1−y)*eps (label smoothing) и весом класса FLIP
    w_pos=n_stay/n_flip. Label smoothing даёт мягкий кап |d|≈CAP_D. Каждый шаг =
    ВСЕ FLIP-бары + случайный STAY-чанк (гарантирует FLIP-градиент, балансирует
    редкий класс). Реальный weight_decay + клип нормы градиента.

    Головы SL/TP вырождены (nvec=1) и loss-маскированы (no-op). Бары с
    bc_weight=0 (анти-каузальный forced_eod STAY) исключаются.
    """
    import torch as th
    import torch.nn.functional as F
    th.manual_seed(seed)
    np.random.seed(seed)
    dev = policy.device
    opt = th.optim.Adam(policy.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    Xt = th.as_tensor(X, device=dev)
    keep = w > 0.0
    flip_idx = np.where((y == 1) & keep)[0]
    stay_idx = np.where((y == 0) & keep)[0]
    if len(flip_idx) == 0 or len(stay_idx) == 0:
        return {"loss_hist": [], "final_loss": float("nan"), "w_pos": 0.0}
    w_pos = float(len(stay_idx) / len(flip_idx))   # ДИАГНОСТИКА (не множитель)
    Xf = Xt[flip_idx]
    tgt_flip = th.full((len(flip_idx),), 1.0 - LABEL_SMOOTH, device=dev)
    hist = []
    for _ in range(n_epochs):
        policy.set_training_mode(True)
        perm = np.random.permutation(stay_idx)
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            d_flip = _d_of(policy, Xf)
            d_stay = _d_of(policy, Xt[b])
            tgt_stay = th.full((len(b),), LABEL_SMOOTH, device=dev)
            # макро-баланс: каждый класс = один mean-BCE терм (равный вклад).
            loss_flip = F.binary_cross_entropy_with_logits(
                d_flip, tgt_flip, reduction="mean") * FLIP_WEIGHT
            loss_stay = F.binary_cross_entropy_with_logits(
                d_stay, tgt_stay, reduction="mean")
            loss = loss_flip + loss_stay
            opt.zero_grad(); loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
            opt.step()
            tot += float(loss)
        hist.append(tot)
    return {"loss_hist": hist, "final_loss": hist[-1], "w_pos": w_pos}


def entropy_stats(logits2: np.ndarray) -> Dict[str, float]:
    """Энтропия головы 0 (nat) по логитам (N,2): mean/max + max|d|.

    Health-чек для PPO-handoff: гиперуверенность (энтропия≈0, |d|≈±80) =
    отравленная инициализация. Здесь печатается ПО КЛАССАМ (STAY/FLIP) в main.
    """
    z = logits2 - logits2.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    ent = -(p * np.log(np.clip(p, 1e-30, 1.0))).sum(axis=1)
    d = logits2[:, 1] - logits2[:, 0]
    return {"ent_mean": float(ent.mean()), "ent_max": float(ent.max()),
            "abs_d_max": float(np.abs(d).max())}


def calibrate_shift(d_by_epoch: Dict[str, Dict[str, np.ndarray]]) -> ShiftCalibration:
    """FPR-anchored калибровка сдвига s* по ПРАВИЛУ pre-reg (детерминированно).

    d_by_epoch[epoch] = {"stay": d[stay], "flip_all": d[entry|exit]}.
    recall/FPR при сдвиге s = ожидание доли сэмплированных FLIP (средние
    sigmoid(d+s)) — точный per-decision предел при бесконечных сидах.
    s* = s_hi (максимальный s с FPR<=цели, обе эпохи) — ЯКОРЬ на цель FPR,
    всегда определён; recall при s* — ИСХОД (репортится гейтом), не подгоняется.
    """
    def sig(x):
        """Сигмоида (вероятность FLIP при сдвиге логита)."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))

    all_d = np.concatenate([np.concatenate([v["stay"], v["flip_all"]])
                            for v in d_by_epoch.values()])
    grid = np.linspace(all_d.min() - 40, all_d.max() + 40, 40001)

    recall_ok = np.ones_like(grid, dtype=bool)
    fpr_ok = np.ones_like(grid, dtype=bool)
    for v in d_by_epoch.values():
        rec = np.array([sig(v["flip_all"] + s).mean() for s in grid])
        fpr = np.array([sig(v["stay"] + s).mean() for s in grid])
        recall_ok &= (rec >= RECALL_FLOOR)
        fpr_ok &= (fpr <= FPR_TARGET)
    # recall растёт с s, FPR тоже растёт с s → recall_ok = s>=s_lo, fpr_ok = s<=s_hi.
    s_lo = float(grid[recall_ok][0]) if recall_ok.any() else float("inf")
    s_hi = float(grid[fpr_ok][-1]) if fpr_ok.any() else float(grid[0])
    feasible = s_lo <= s_hi
    s_star = s_hi                        # FPR-anchored, всегда определён

    gap = min(v["flip_all"].min() - v["stay"].max() for v in d_by_epoch.values())
    return ShiftCalibration(s_lo=s_lo, s_hi=s_hi, s_star=s_star, feasible=feasible,
                            gap_nat=float(gap),
                            gap_needed=_logit(RECALL_FLOOR) - _logit(FPR_TARGET))


def main() -> None:
    """CLI: обучить клон BC на пуле обеих эпох, сохранить политику + манифест."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/cubecloud/Data/rlbinancetrader")
    ap.add_argument(
        "--out",
        default="/home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy_reg_cd")
    ap.add_argument("--seed", type=int, default=TRAIN_SEED)
    args = ap.parse_args()

    X_raw, y, w, meta = load_pooled(args.data)
    mu, sd = fit_scaler(X_raw)
    X = apply_scaler(X_raw, mu, sd)
    print(f"пул: {X.shape}, FLIP={int((y == 1).sum())}, "
          f"eps={LABEL_SMOOTH} cap|d|~{CAP_D:.2f} wd={WEIGHT_DECAY}")
    model, policy = build_policy(X.shape[1], args.seed)
    tr = train(policy, X, y, w, args.seed)
    print(f"обучение: final_loss={tr['final_loss']:.4e} w_pos={tr['w_pos']:.1f}")

    logits = head0_logits(policy, X)
    d = logits[:, 1] - logits[:, 0]
    # энтропия ПО КЛАССАМ (health для PPO).
    is_stay = (y == 0)
    is_flip = (y == 1)
    es = entropy_stats(logits[is_stay])
    ef = entropy_stats(logits[is_flip])
    print(f"энтропия head0: STAY mean={es['ent_mean']:.4f} | "
          f"FLIP mean={ef['ent_mean']:.4f} | max|d|={es['abs_d_max']:.2f} "
          f"(cap {CAP_D:.2f})")

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
        "objective": "weighted_BCE_label_smooth",
        "seed": args.seed, "label_smooth": LABEL_SMOOTH,
        "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "cap_d": CAP_D,
        "flip_weight": FLIP_WEIGHT, "w_pos_diag": tr["w_pos"],
        "fpr_target": FPR_TARGET,
        "recall_floor": RECALL_FLOOR, "n_epochs": N_EPOCHS, "lr": LR,
        "hidden": list(HIDDEN), "final_loss": tr["final_loss"],
        "abs_d_max": es["abs_d_max"],
        "entropy_stay_mean": es["ent_mean"], "entropy_flip_mean": ef["ent_mean"],
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
