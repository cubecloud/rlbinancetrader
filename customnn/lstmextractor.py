import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box
from rllab.labtools import deserialize_kwargs


# Swish Function
class Swish(nn.Module):
    def __init__(self):
        """
        Init method.
        """
        super(Swish, self).__init__()

    def forward(self, x):
        """
        Forward pass of the function.

        Args:
            x: tensor
        """
        return x * torch.sigmoid(x)


nn_serializer: dict = {'ReLU': nn.ReLU,
                       'LeakyReLU': nn.LeakyReLU,
                       'Tanh': nn.Tanh,
                       'Swish': Swish
                       }


class LSTMExtractorNN(BaseFeaturesExtractor):
    """
    Args:   
        observation_space (gym.Space):
        features_dim: (int):            Number of features extracted. 
                                        This corresponds to the number of unit for the last layer.
                                        and hidden layer will be features_dim // 4

    """

    def __init__(self, observation_space: Box,
                 features_dim: int = 256,
                 activation_fn='ReLU'):
        super().__init__(observation_space, features_dim)
        self._features_dim = features_dim
        self.activation_fn = deserialize_kwargs(activation_fn, lab_serializer=nn_serializer)
        self.lstm = nn.LSTM(input_size=observation_space.shape[-1],
                            hidden_size=features_dim,
                            num_layers=1,
                            batch_first=True)
        self.linear = nn.Linear(features_dim, features_dim)
        self.activation = self.activation_fn()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # outputs is in (batch, sequence, features)
        # hidden is in (num_layers * num_directions, batch, hidden_size)
        # cell is in (num_layers * num_directions, batch, hidden_size)
        _, (hidden, _) = self.lstm(observations)
        # we need to use the last hidden state as the feature
        # hidden is the last hidden state for each sequence in the batch
        return self.activation(self.linear(hidden.squeeze(0)))


if __name__ == "__main__":
    _observation_space = Box(low=0, high=1, shape=(48, 21))
    feature_extractor = LSTMExtractorNN(_observation_space, features_dim=256)
    _observations = torch.randn(2, 48, 21)
    _x = feature_extractor(_observations)
    print(_x)
    print(_x.shape)
