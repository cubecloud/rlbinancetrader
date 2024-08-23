import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box


class MlpExtractorNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
        and hidden layer will be features_dim // 4
    """

    def __init__(self, observation_space: Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.mlp_extractor = nn.Sequential(
            nn.Linear(observation_space.shape[0], features_dim),
            nn.LeakyReLU(),
            nn.Linear(features_dim, int(features_dim // 4)),
            nn.LeakyReLU(),
        )

        self.linear = nn.Sequential(
            nn.Linear(features_dim // 4, features_dim),
            nn.LeakyReLU(),)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.mlp_extractor(observations))
