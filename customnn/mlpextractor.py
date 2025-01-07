import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box
from rllab.labtools import deserialize_kwargs
from customnn.nnserializerdata import nn_serializer

__version__ = 0.003


class MlpExtractorNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
        and hidden layer will be features_dim // 4
    """

    def __init__(self, observation_space: Box, features_dim: int = 256, last_features_dim: int = 256,
                 activation_fn='LeakyReLU'):
        super().__init__(observation_space, features_dim)
        print(f'Input features = {features_dim}')
        self._features_dim = last_features_dim
        self.activation_fn = deserialize_kwargs(activation_fn, lab_serializer=nn_serializer)
        self.mlp_extractor = nn.Sequential(
            nn.Linear(observation_space.shape[0], features_dim),
            self.activation_fn(),
            nn.Linear(features_dim, int(features_dim // 2)),
            self.activation_fn(),
            nn.Linear(int(features_dim // 2), int(features_dim // 4)),
            self.activation_fn(),
        )

        self.linear = nn.Sequential(
            nn.Linear(features_dim // 4, last_features_dim),
            self.activation_fn(), )
        print(f'Features extractor features_dim = {self._features_dim}')

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.mlp_extractor(observations))
