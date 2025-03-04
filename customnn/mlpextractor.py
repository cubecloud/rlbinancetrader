import numpy as np

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box
from rllab.labtools import deserialize_kwargs
from customnn.nnserializerdata import nn_serializer

__version__ = 0.006


class MlpExtractorNN(BaseFeaturesExtractor):
    """Multilayer Perceptron network for extracting features from observations.

    This extractor is used to process input observations in reinforcement learning (RL) tasks.
    It supports both one-dimensional feature vectors and multidimensional
    observations (such as images), which are flattened using a Flatten layer.

    Args:
        observation_space (gym.Space): The environment's observation space.
        features_dim (int, optional): The number of output features after the first layer. Defaults to 256.
        last_features_dim (int, optional): The final size of the extracted features. Defaults to 256.
        activation_fn (str or callable, optional): Activation function for hidden layers. Defaults to LeakyReLU.

    Attributes:
        mlp_extractor (torch.nn.Sequential): A sequence of fully connected layers for processing observations.
        linear (torch.nn.Sequential): A linear layer for final feature processing.
    """

    def __init__(self, observation_space: Box, features_dim: int = 256, last_features_dim: int = 256,
                 activation_fn='LeakyReLU'):
        super().__init__(observation_space, features_dim)
        print(f'Input features = {features_dim}')
        self._features_dim = last_features_dim
        self.activation_fn = deserialize_kwargs(activation_fn, lab_serializer=nn_serializer)

        # Determine whether we need to flatten the input based on its shape
        if len(observation_space.shape) > 1:
            input_size = np.prod(observation_space.shape)
            self.preprocess = nn.Flatten()
        else:
            input_size = observation_space.shape[0]
            self.preprocess = nn.Identity()  # No-op for non-image inputs

        self.mlp_extractor = nn.Sequential(
            nn.Linear(input_size, features_dim),
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
        # Apply preprocessing if needed (flatten for images, identity otherwise)
        observations = self.preprocess(observations)
        return self.linear(self.mlp_extractor(observations))
