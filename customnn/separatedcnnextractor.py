import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box

__version__ = 0.005


class SeparatedCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, features_dim: int = 256):
        assert len(observation_space.shape) == 2, "Observation space must be 2D (lookback, features)"
        # Calculate feature splits
        self.asset_features = 6  # Update this based on your actual asset features
        self.indicator_features = observation_space.shape[-1] - self.asset_features
        self.lookback = observation_space.shape[0]
        super().__init__(observation_space, features_dim)
        print(f'Input features = {features_dim}, obs_space = {observation_space}')

        # Asset feature processing branch (handles 0-1 normalized values)
        self.asset_cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.asset_features, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Indicator feature processing branch (handles -1 to 1 normalized values)
        self.indicator_cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.indicator_features, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.Tanh(),  # Match indicator normalization range
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Calculate output dimensions
        with torch.no_grad():
            dummy_asset = torch.randn(1, self.asset_features, self.lookback)
            dummy_ind = torch.randn(1, self.indicator_features, self.lookback)
            asset_out = self.asset_cnn(dummy_asset)
            ind_out = self.indicator_cnn(dummy_ind)
            total_features = int(asset_out.shape[1] + ind_out.shape[1])

        self.final_layer = nn.Sequential(
            nn.Linear(total_features, features_dim),
            nn.LayerNorm(features_dim),
            nn.Dropout(0.1)
        )
        print(f'Total extracted features (before Linear) = {total_features}')
        print(f'Features extractor features_dim = {self._features_dim}')

    def _preprocess(self, observations: torch.Tensor) -> tuple:
        """Split and reshape observations"""
        # Convert (batch, lookback, features) to (batch, features, lookback)
        observations = observations.permute(0, 2, 1)
        return (
            observations[:, :self.asset_features, :],
            observations[:, self.asset_features:, :]
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        asset_input, indicator_input = self._preprocess(observations)
        asset_features = self.asset_cnn(asset_input).flatten(1)
        indicator_features = self.indicator_cnn(indicator_input).flatten(1)
        combined = torch.cat([asset_features, indicator_features], dim=1)
        return self.final_layer(combined)
