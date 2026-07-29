"""КАРКАС PPO для RL-этапа (ТОЛЬКО обвязка, НЕ запуск обучения).

Собирает связку, описанную в pre-reg (prereg_rl_stage): семя reg_cd_vw2
(политика reg_cd + масштабно-инициализированный критик) грузится в PPO,
policy-градиент и entropy-бонус считаются ТОЛЬКО по барам ПЕРВОЙ СВОБОДЫ
(dip-in-position; loss-masking головы 0), а к клону добавлен KL-якорь
(BC→PPO handoff: не дать первым апдейтам увести политику от клона).

ЧТО ЗДЕСЬ ЕСТЬ:
  * MaskedKLPPO — подкласс sb3.PPO с переопределённым train():
      - PG-лосс и entropy-бонус маскируются ПОБАРНО маской dip-in-position
        (first_freedom_head_mask по голове 0); forced-бары (flat/transition/
        cooldown) в policy-градиент НЕ входят;
      - KL-якорь к ЗАМОРОЖЕННОМУ клону по ВСЕМ барам (держит политику у клона
        там, где агент не свободен, и штрафует уход на dip-выходах);
      - value-лосс стандартный (критик доучивается on-policy).
  * build_scaffold — построить PPO с gamma=1.0, загрузить семя, приморозить клон.
  * smoke_scaffold — прогнать learn(100) на dummy-env: связка стартует и не падает.

ЧЕГО ЗДЕСЬ НЕТ: реального обучения, реальной среды-прогона, подбора порогов.
Пороги (KL-коэф, entropy, шаги, останов) — в pre-reg, требуют утверждения.
"""
from __future__ import annotations

import copy
from typing import Dict, Optional

import numpy as np

from spotrl.algo.head_masking import first_freedom_head_mask
from spotrl.spec.actions import HEAD_POSITION


def make_seed_scaler_vecenv(venv, mu: np.ndarray, sd: np.ndarray,
                            cooldown_remain_idx: int = 30):
    """Обернуть VecEnv производным признаком + ФИКСИРОВАННЫМ скейлером семени.

    Клон reg_cd обучен на СТАНДАРТИЗОВАННЫХ входах длины 39
    (`apply_scaler(X, mu, sd)`), где 39-й столбец — ПРОИЗВОДНЫЙ `a_cooldown_active`
    (= a_cooldown_remain > 0, `train_clone.derive_extra`). Но `SpotFlipEnv.observe()`
    отдаёт СЫРЫЕ obs длины 38 (ObservationSpec.v2, БЕЗ производного столбца). Без
    инъекции производного признака apply_scaler(38-d, mu/sd 39-d) падает broadcast-
    ошибкой (parity-gate §6 п.8), а семя/клон получают вход не той формы. Обёртка:
      (1) добавляет 39-й столбец a_cooldown_active из raw-столбца a_cooldown_remain;
      (2) применяет apply_scaler фиксированным скейлером семени.
    Итог — buffer в 39-мерном стандартизованном пространстве, где клон корректен и
    деста маски dip точна.

    ВАЖНО: скейлер ФИКСИРОВАН (mu/sd семени), НЕ адаптивный. VecNormalize с
    бегущими mu/sd сломал бы паритет с клоном — использовать нельзя.

    Args:
        cooldown_remain_idx: индекс столбца `a_cooldown_remain` в 38-мерном
            env.observe() (порядок ObservationSpec.v2; по манифесту = 30).
    """
    from stable_baselines3.common.vec_env import VecEnvWrapper
    from spotrl.bc.train_clone import apply_scaler
    import gymnasium as gym

    n_out = int(np.asarray(mu).shape[0])

    def _inject_and_scale(obs: np.ndarray) -> np.ndarray:
        """raw (n,38) → append a_cooldown_active → apply_scaler → (n,39)."""
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape[1] == n_out:            # уже 39 (напр. dummy-env) — не трогаем
            return apply_scaler(obs, mu, sd)
        cd_active = (obs[:, cooldown_remain_idx] > 0.0).astype(np.float32)
        obs39 = np.concatenate([obs, cd_active[:, None]], axis=1)
        return apply_scaler(obs39, mu, sd)

    class _SeedScaler(VecEnvWrapper):
        """VecEnv-обёртка: raw obs → производный признак → apply_scaler семени."""

        def __init__(self, venv):
            """Расширить observation_space до 39-мерного стандартизованного."""
            super().__init__(
                venv,
                observation_space=gym.spaces.Box(
                    low=-10.0, high=10.0, shape=(n_out,), dtype=np.float32))

        def reset(self):
            """Сбросить, инъектировать производный признак, стандартизовать."""
            return _inject_and_scale(self.venv.reset())

        def step_wait(self):
            """Шаг: инъекция производного признака + фиксированный скейлер."""
            obs, rew, done, info = self.venv.step_wait()
            return _inject_and_scale(obs), rew, done, info

    return _SeedScaler(venv)


def _frozen_clone(policy):
    """Глубокая КОПИЯ политики-клона, полностью замороженная (KL-якорь).

    eval-режим, requires_grad=False на всех параметрах: якорь не обучается,
    даёт эталонное распределение головы 0 на каждом баре.
    """
    clone = copy.deepcopy(policy)
    clone.set_training_mode(False)
    for p in clone.parameters():
        p.requires_grad_(False)
    return clone


class MaskedKLPPO:
    """Каркас: sb3.PPO с маской головы 0 (dip-in-position) и KL-якорем к клону.

    Реализован как фабрика вокруг sb3.PPO с monkey-patch train(), чтобы не тащить
    полное переопределение класса в каркас. build_scaffold() возвращает готовый
    экземпляр sb3.PPO с прикреплённым _scaffold с параметрами маски/якоря.
    """


def _make_masked_kl_train(model, mu, sd, col_index, kl_coef, params_own,
                          min_hold_days=0.0):
    """Собрать train()-замену для sb3.PPO с маской dip и KL-якорем.

    Держит замыкание на mu/sd/col_index (де-стандартизация obs для предиката
    dip-in-position — предикат ЧИСТАЯ функция obs), kl_coef и замороженный клон.
    """
    import torch as th
    import torch.nn.functional as F
    from stable_baselines3.common.utils import explained_variance

    clone = _frozen_clone(model.policy)
    mu_t = th.as_tensor(np.asarray(mu, np.float32))
    sd_t = th.as_tensor(np.asarray(sd, np.float32))

    def train() -> None:
        """Один PPO-апдейт с маской dip по голове 0 и KL-якорем к клону."""
        model.policy.set_training_mode(True)
        model._update_learning_rate(model.policy.optimizer)
        clip_range = model.clip_range(model._current_progress_remaining)
        pg_losses, value_losses, ent_losses, kl_anchor = [], [], [], []
        kl_dip_list = []

        for _epoch in range(model.n_epochs):
            for rollout_data in model.rollout_buffer.get(model.batch_size):
                obs = rollout_data.observations
                actions = rollout_data.actions.long()

                values, log_prob, entropy = model.policy.evaluate_actions(
                    obs, actions)
                values = values.flatten()

                # --- МАСКА ПЕРВОЙ СВОБОДЫ (dip-in-position) по голове 0 ---
                obs_np = obs.detach().cpu().numpy()
                m = first_freedom_head_mask(obs_np, mu, sd, col_index, params_own,
                                            min_hold_days)
                dip = th.as_tensor(m[:, HEAD_POSITION], device=obs.device)
                n_dip = dip.sum().clamp(min=1.0)

                # advantage нормируется по DIP-ПОДМНОЖЕСТВУ (pre-reg §6 п.9): в PG
                # идут ТОЛЬКО dip-бары (<~10% буфера; замер калибровки: dip-std ≈
                # 2.97× full-std). Нормировка по полному минибатчу раздула бы
                # dip-градиент ≈3× (rms 3.08 вместо 1.0) и сместила центр
                # статистикой forced-баров. Нормируем статистикой dip-подмножества.
                advantages = rollout_data.advantages
                if model.normalize_advantage and n_dip > 1:
                    a_dip = advantages[dip.bool()]
                    advantages = (advantages - a_dip.mean()) / (a_dip.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                pl1 = advantages * ratio
                pl2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                # policy-лосс ТОЛЬКО по dip-барам (побарная маска на surrogate).
                policy_loss = -(th.min(pl1, pl2) * dip).sum() / n_dip

                value_loss = F.mse_loss(rollout_data.returns, values)

                # entropy-бонус тоже только по dip-барам.
                if entropy is None:
                    entropy_loss = -(-log_prob * dip).sum() / n_dip
                else:
                    entropy_loss = -(entropy * dip).sum() / n_dip

                # --- KL-ЯКОРЬ к клону, РАЗДЕЛЬНЫЙ (amend1 §3): полная сила на
                # FORCED-барах (~dip: точность копирования v7); на dip-in-position
                # барах — СЛАБЫЙ адаптивный якорь `model._kl_dip_coef` (нуль/слабый
                # по amend1). Нулевой dip-якорь → p(FLIP)@dip взрывается (PG гонит
                # ранний выход, энтропия управляет только шумом); слабый адаптивный
                # якорь прямо держит p(FLIP)@dip в коридоре, не задушивая (диагноз
                # ОБНОВЛЕНИЯ 3 — полный якорь душил; ОБНОВЛЕНИЯ 4 — нулевой взрывал).
                with th.no_grad():
                    cats_ref = clone.get_distribution(obs).distribution
                cats_cur = model.policy.get_distribution(obs).distribution
                kl_bar = th.distributions.kl_divergence(
                    cats_ref[HEAD_POSITION], cats_cur[HEAD_POSITION])
                forced = 1.0 - dip
                n_forced = forced.sum().clamp(min=1.0)
                kl_forced = (kl_bar * forced).sum() / n_forced   # FORCED (β полн.)
                kl_dip = (kl_bar * dip).sum() / n_dip            # dip (слабый якорь)
                kl_dip_coef = float(getattr(model, "_kl_dip_coef", 0.0))
                kl_dip_log = float(kl_dip)

                loss = (policy_loss
                        + model.ent_coef * entropy_loss
                        + model.vf_coef * value_loss
                        + kl_coef * kl_forced
                        + kl_dip_coef * kl_dip)

                model.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    model.policy.parameters(), model.max_grad_norm)
                model.policy.optimizer.step()

                pg_losses.append(float(policy_loss))
                value_losses.append(float(value_loss))
                ent_losses.append(float(entropy_loss))
                kl_anchor.append(float(kl_forced))  # KL forced-баров (якорь)
                kl_dip_list.append(kl_dip_log)     # KL dip-баров (свобода, диагн.)

        model._n_updates += model.n_epochs
        explained_var = explained_variance(
            model.rollout_buffer.values.flatten(),
            model.rollout_buffer.returns.flatten())
        model.logger.record("train/policy_gradient_loss", float(np.mean(pg_losses)))
        model.logger.record("train/value_loss", float(np.mean(value_losses)))
        model.logger.record("train/entropy_loss", float(np.mean(ent_losses)))
        model.logger.record("train/kl_anchor_clone", float(np.mean(kl_anchor)))
        model.logger.record("train/kl_dip_free", float(np.mean(kl_dip_list)))
        model.logger.record("train/explained_variance", float(explained_var))

    return train, clone


def build_scaffold(env, seed_model_path: str, manifest_path: str,
                   kl_coef: float, ent_coef: float, kl_dip_coef: float = 0.0,
                   gamma: float = 1.0, params_own: bool = False,
                   seed: int = 0, apply_seed_scaler: bool = True,
                   min_hold_days: float = 0.0, **ppo_kwargs):
    """Построить каркас PPO: gamma=1.0, семя reg_cd_vw2, маска dip, KL-якорь.

    Args:
        env: (Vec)Env первой свободы (exit_own, entry_own=False, params_own=False).
        seed_model_path: путь к семени reg_cd_vw2 (без .zip).
        manifest_path: манифест семени (obs_cols, scaler_mu/sd).
        kl_coef: коэффициент KL-якоря к клону (из pre-reg, требует утверждения).
        ent_coef: коэффициент энтропии (из pre-reg).
        gamma: ОБЯЗАН быть 1.0 (EnvConfig.gamma; иначе кредит на длинных холдах
            затухает). Сохранённое BC-семя имеет gamma=0.99 — здесь переопределяем.

    Returns:
        (model, clone): sb3.PPO с прикреплённым train() и замороженный клон.
    """
    import json
    from pathlib import Path
    from stable_baselines3 import PPO

    if gamma != 1.0:
        raise ValueError("gamma каркаса обязан быть 1.0 (EnvConfig.gamma)")

    mf = json.loads(Path(manifest_path).read_text())
    mu = np.array(mf["scaler_mu"], np.float32)
    sd = np.array(mf["scaler_sd"], np.float32)
    col_index = {name: i for i, name in enumerate(mf["obs_cols"])}

    # train/serve parity: buffer в стандартизованном пространстве семени (иначе
    # клон/маска на сырых obs — мусор). Фиксированный скейлер, НЕ VecNormalize.
    if apply_seed_scaler:
        env = make_seed_scaler_vecenv(env, mu, sd)

    model = PPO("MlpPolicy", env, gamma=gamma, ent_coef=ent_coef, seed=seed,
                device="cpu",
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                **ppo_kwargs)
    # загрузить веса семени (политика reg_cd + масштабный критик).
    seed_model = PPO.load(seed_model_path, device="cpu")
    model.policy.load_state_dict(seed_model.policy.state_dict())

    model._kl_dip_coef = float(kl_dip_coef)   # слабый адаптивный якорь на dip
    train_fn, clone = _make_masked_kl_train(
        model, mu, sd, col_index, kl_coef, params_own, min_hold_days)
    model.train = train_fn                 # monkey-patch train()
    model._scaffold = {"kl_coef": kl_coef, "ent_coef": ent_coef,
                       "kl_dip_coef": kl_dip_coef,
                       "gamma": gamma, "params_own": params_own,
                       "col_index": col_index, "clone": clone}
    return model, clone


def _dummy_vec_env(obs_dim: int = 39, seed: int = 0):
    """Dummy-VecEnv с формами первой свободы: obs (obs_dim,), MultiDiscrete([2,1,1]).

    Для smoke: реальная среда SpotFlipEnv грузит state-parquet (тяжело), для
    проверки «связка стартует» достаточно совпадения пространств.
    """
    import gymnasium as gym
    from stable_baselines3.common.env_util import make_vec_env

    class _Env(gym.Env):
        """Dummy-среда с пространствами первой свободы (для smoke)."""

        def __init__(self):
            """Задать obs (obs_dim,) и MultiDiscrete([2,1,1])."""
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            self.action_space = gym.spaces.MultiDiscrete(np.array([2, 1, 1]))
            self._i = 0

        def reset(self, *, seed=None, options=None):
            """Сброс: нулевое наблюдение."""
            self._i = 0
            return np.zeros(obs_dim, np.float32), {}

        def step(self, action):
            """Шаг: случайный obs, малый reward, усечение через 64 бара."""
            self._i += 1
            obs = np.random.randn(obs_dim).astype(np.float32)
            return obs, float(np.random.randn()) * 1e-3, False, self._i >= 64, {}

    return make_vec_env(_Env, n_envs=1, seed=seed)


def smoke_scaffold(seed_model_path: str, manifest_path: str,
                   kl_coef: float = 0.1, ent_coef: float = 0.01,
                   n_steps: int = 100) -> Dict:
    """SMOKE: собрать каркас на dummy-env и прогнать learn(n_steps). НЕ обучение.

    Проверяет: gamma==1.0, клон заморожен (ни один параметр не требует градиента),
    learn() не падает. Возвращает диагностику.
    """
    env = _dummy_vec_env()
    model, clone = build_scaffold(
        env, seed_model_path, manifest_path, kl_coef=kl_coef, ent_coef=ent_coef,
        gamma=1.0, n_steps=64, batch_size=64, n_epochs=2)

    assert model.gamma == 1.0, "gamma каркаса != 1.0"
    clone_frozen = all(not p.requires_grad for p in clone.parameters())
    assert clone_frozen, "клон-якорь НЕ заморожен"

    model.learn(total_timesteps=n_steps, progress_bar=False)
    scaler_applied = "_SeedScaler" in type(model.env).__name__
    return {"started": True, "gamma": model.gamma,
            "clone_frozen": clone_frozen, "scaler_applied": scaler_applied,
            "n_steps": int(n_steps),
            "kl_coef": kl_coef, "ent_coef": ent_coef}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seed-model",
        default="/home/cubecloud/Data/rlbinancetrader/bc_clone_v7_policy_reg_cd_vw2")
    ap.add_argument(
        "--manifest",
        default="/home/cubecloud/Data/rlbinancetrader/"
                "bc_clone_v7_policy_reg_cd_vw2.manifest.json")
    args = ap.parse_args()
    r = smoke_scaffold(args.seed_model, args.manifest)
    print("=== SMOKE каркаса PPO (100 шагов, dummy-env) ===")
    print(r)
