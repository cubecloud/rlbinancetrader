import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box

__version__ = 0.006  # Update version to reflect changes


class SeparatedCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, features_dim: int = 256, assets_features=6, actions_features=4):
        assert len(observation_space.shape) == 2, "Observation space must be 2D (lookback, features)"

        # Define feature splits dimensions
        self.asset_features = assets_features  # Update based on your environment
        self.action_features = actions_features  # Number of possible actions (one-hot encoded)
        self.indicator_features = observation_space.shape[-1] - self.asset_features - self.action_features

        self.lookback = observation_space.shape[0]
        super().__init__(observation_space, features_dim)
        print(f'Input features breakdown: Assets={self.asset_features}, '
              f'Actions={self.action_features}, Indicators={self.indicator_features}')

        # Asset feature processing branch
        self.asset_cnn = nn.Sequential(
            nn.Conv1d(self.asset_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Action feature processing branch
        self.action_cnn = nn.Sequential(
            nn.Conv1d(self.action_features, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Indicator feature processing branch
        self.indicator_cnn = nn.Sequential(
            nn.Conv1d(self.indicator_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Calculate combined feature dimension
        with torch.no_grad():
            dummy_asset = torch.randn(1, self.asset_features, self.lookback)
            dummy_action = torch.randn(1, self.action_features, self.lookback)
            dummy_ind = torch.randn(1, self.indicator_features, self.lookback)

            asset_out = self.asset_cnn(dummy_asset)
            action_out = self.action_cnn(dummy_action)
            ind_out = self.indicator_cnn(dummy_ind)

            total_features = asset_out.shape[1] + action_out.shape[1] + ind_out.shape[1]

        self.final_layer = nn.Sequential(
            nn.Linear(total_features, features_dim),
            nn.LayerNorm(features_dim),
            nn.Dropout(0.1)
        )
        print(f'Total features before final layer: {total_features}')
        print(f'Final feature dimension: {features_dim}')

    def _preprocess(self, observations: torch.Tensor) -> tuple:
        """Split observations into components and permute dimensions"""
        observations = observations.permute(0, 2, 1)  # (batch, features, lookback)
        return (
            observations[:, :self.asset_features, :],  # Asset features
            observations[:, self.asset_features:self.asset_features + self.action_features, :],  # Action features
            observations[:, self.asset_features + self.action_features:, :]  # Indicator features
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        asset_input, action_input, indicator_input = self._preprocess(observations)

        # Process each feature group through its CNN
        asset_out = self.asset_cnn(asset_input).flatten(1)
        action_out = self.action_cnn(action_input).flatten(1)
        indicator_out = self.indicator_cnn(indicator_input).flatten(1)

        # Combine features and pass through final layer
        combined = torch.cat([asset_out, action_out, indicator_out], dim=1)
        return self.final_layer(combined)
