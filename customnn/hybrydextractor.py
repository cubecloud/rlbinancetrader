import torch
import torch.nn as nn
import math
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box

__version__ = 0.017  # Version with positional encoding and tailored activations


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class HybridFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, assets_features: int = 6, actions_features: int = 4,
                 indicators_sign: bool = False, final_dropout: float = 0.25):
        assert len(observation_space.shape) == 2, "Observation space must be 2D (lookback, features)"
        # Input dimensions
        self.final_dropout = final_dropout
        self.asset_features = assets_features  # Update based on your environment
        self.action_features = actions_features  # Changed to 4 as requested
        self.indicator_features = observation_space.shape[-1] - self.asset_features - self.action_features
        self.lookback = observation_space.shape[0]
        if indicators_sign:
            self.indicator_temporal_activation = nn.Tanh
        else:
            self.indicator_temporal_activation = nn.ReLU

        super().__init__(observation_space, features_dim=1)
        print(f'Feature breakdown - Assets: {self.asset_features}, '
              f'Actions: {self.action_features}, Indicators: {self.indicator_features}')

        # Asset feature processor
        self.asset_net = nn.Sequential(
            nn.Conv1d(self.asset_features, self.asset_features * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.asset_features * 4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1)
        )

        # Action feature processor (handles one-hot encoded actions)
        self.action_net = nn.Sequential(
            nn.Conv1d(self.action_features, self.action_features * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.action_features * 4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1)
        )

        # Indicator processor with transformer and positional encoding
        self.pos_encoder = PositionalEncoding(self.indicator_features, self.lookback)
        self.indicator_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.indicator_features,
                nhead=4,
                dim_feedforward=128,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2
        )

        # Temporal CNN tailored for [-1, 1] or [0,  1] data
        self.indicator_temporal = nn.Sequential(
            nn.Conv1d(self.indicator_features, self.indicator_features ** 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.indicator_features ** 2),
            self.indicator_temporal_activation(),  # Matches input range [-1, 1] or [0, 1]
            nn.MaxPool1d(2),  # Preserves extreme values
            nn.Conv1d(self.indicator_features ** 2, self.indicator_features ** 2 * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.indicator_features ** 2 * 2),
            self.indicator_temporal_activation(),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )

        # Calculate output dimensions for all branches
        with torch.no_grad():
            # Asset branch
            dummy_asset = torch.randn(1, self.asset_features, self.lookback)
            last_asset_out = torch.Tensor(dummy_asset[:, :self.asset_features, -1]).flatten(1)
            asset_out = self.asset_net(dummy_asset).flatten(1)

            # Action branch
            dummy_action = torch.randn(1, self.action_features, self.lookback)
            last_actions_out = torch.Tensor(
                dummy_action[:, self.asset_features:self.asset_features + self.action_features, -1]).flatten(1)
            action_out = self.action_net(dummy_action).flatten(1)

            # Indicator branch
            dummy_ind = torch.randn(1, self.lookback, self.indicator_features)
            dummy_ind = self.pos_encoder(dummy_ind)
            trans_out = self.indicator_transformer(dummy_ind).permute(0, 2, 1)
            ind_out = self.indicator_temporal(trans_out).flatten(1)

            comb_out = torch.cat([asset_out, last_asset_out, action_out, last_actions_out, ind_out], dim=1)
            total_concat = comb_out.shape[-1]
        print(f'Features extractor total_concat = {total_concat}')

        # Final layers
        self.final_layer = nn.Sequential(
            nn.LayerNorm(total_concat),
            nn.Dropout(self.final_dropout),
        )
        with torch.no_grad():
            final_out = self.final_layer(comb_out)
            self._features_dim = final_out.shape[-1]
        print(f'Features extractor features_dim = {self._features_dim}')

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Split and permute inputs
        observations = observations.permute(0, 2, 1)  # (batch, features, lookback)
        asset_input = observations[:, :self.asset_features, :]
        last_asset_out = torch.Tensor(asset_input[:, :self.asset_features, -1]).flatten(1)
        action_input = observations[:, self.asset_features:self.asset_features + self.action_features, :]
        last_action_out = torch.Tensor(
            action_input[:, self.asset_features:self.asset_features + self.action_features, -1]).flatten(1)

        indicator_input = observations[:, self.asset_features + self.action_features:, :]

        # Process assets
        asset_features = self.asset_net(asset_input).flatten(1)

        # Process actions
        action_features = self.action_net(action_input).flatten(1)

        # Process indicators
        indicator_input = indicator_input.permute(0, 2, 1)  # (batch, lookback, features)
        indicator_input = self.pos_encoder(indicator_input)
        transformed = self.indicator_transformer(indicator_input).permute(0, 2, 1)  # (batch, features, lookback)
        indicator_features = self.indicator_temporal(transformed).flatten(1)

        # Combine all features
        combined = torch.cat([asset_features, last_asset_out,
                              action_features, last_action_out,
                              indicator_features],
                             dim=1)
        return self.final_layer(combined)
