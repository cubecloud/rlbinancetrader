"""PSLPriorPolicy (amend5, путь 1, механизм B): структурный приор на FLIP-логит.

Сдвигает логит выхода (FLIP, индекс 1 головы позиции) на alpha·(w·obs+b), где
(w,b) — ЗАМОРОЖЕННЫЙ линейный K3-классификатор p_SL на стандартизованных 39 obs
(тех же, что в буфере PPO). Тогда политика ПО ПОСТРОЕНИЮ склонна выходить только
при высоком p_SL (SL-подобные dips), а RL дообучает. Приор — чистая функция obs
→ нет хирургии скейлера/семени, нет смены размерности, train/serve парность
сохранена (obs бит-в-бит одинаковы). Награду НЕ трогает (не reward-hacking).

Сдвиг применяется СОГЛАСОВАННО в forward (сбор роллаута), evaluate_actions и
get_distribution (train/якорь/диагностика) — importance ratio PPO корректен.
Замороженный клон (KL-якорь) — deepcopy ЭТОЙ политики → тоже с приором, значит
KL(clone‖current)=0 на старте (приор общий).
"""
from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy


class PSLPriorPolicy(ActorCriticPolicy):
    """ActorCriticPolicy со сдвигом FLIP-логита на alpha·K3(obs)."""

    def __init__(self, *args, psl_w=None, psl_b=0.0, psl_alpha=0.0, **kwargs):
        """psl_w(39,), psl_b — веса линейного K3; psl_alpha — сила приора."""
        super().__init__(*args, **kwargs)
        w = np.zeros(self.observation_space.shape[0], np.float32) if psl_w is None \
            else np.asarray(psl_w, np.float32)
        self.register_buffer("psl_w", th.as_tensor(w))
        self.register_buffer("psl_b", th.as_tensor(np.float32(psl_b)))
        self.psl_alpha = float(psl_alpha)

    def _psl_score(self, obs: th.Tensor) -> th.Tensor:
        """K3-скор = w·obs + b (логит p_SL) по стандартизованным obs, (batch,)."""
        return (obs.float() * self.psl_w).sum(dim=1) + self.psl_b

    def _dist_from_obs_latent(self, obs: th.Tensor, latent_pi: th.Tensor):
        """Построить распределение действий со сдвигом FLIP-логита на alpha·score."""
        mean_actions = self.action_net(latent_pi)
        if self.psl_alpha != 0.0:
            shift = self.psl_alpha * self._psl_score(obs)
            mean_actions = mean_actions.clone()
            mean_actions[:, 1] = mean_actions[:, 1] + shift   # FLIP = индекс 1
        return self.action_dist.proba_distribution(action_logits=mean_actions)

    def _latent_pi(self, obs: th.Tensor) -> th.Tensor:
        """Латент актора (учёт shared/separate features_extractor)."""
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, _ = self.mlp_extractor(features)
        else:
            pi_features, _ = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
        return latent_pi

    def get_distribution(self, obs: th.Tensor):
        """Распределение действий с приором (для train/якоря/диагностики)."""
        return self._dist_from_obs_latent(obs, self._latent_pi(obs))

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        """Сбор роллаута: действия/значения/log_prob со сдвинутым распределением."""
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        dist = self._dist_from_obs_latent(obs, latent_pi)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        """Оценка действий (PPO train): values/log_prob/entropy с приором."""
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        dist = self._dist_from_obs_latent(obs, latent_pi)
        log_prob = dist.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, dist.entropy()

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """Значения критика (приор не влияет на value-ветку)."""
        return super().predict_values(obs)
