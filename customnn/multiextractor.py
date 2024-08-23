import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

__version__ = 0.003


class MlpExtractorNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
        and hidden layer will be features_dim // 4
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.mlp_extractor = nn.Sequential(
            nn.Linear(observation_space.shape[0], features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, int(features_dim // 4)),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            nn.Linear(features_dim // 4, features_dim),
            nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.mlp_extractor(observations))


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "indicators":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                # extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], features_dim),
                                                nn.ReLU(),
                                                nn.Linear(features_dim, int(features_dim // 4)),
                                                nn.ReLU(),
                                                )

                total_concat_size += int(features_dim // 4)
            elif key == "ohlc":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], subspace.shape[0] // 2),
                                                nn.ReLU(),
                                                nn.Linear(subspace.shape[0] // 2, subspace.shape[0] // 2),
                                                nn.ReLU(),
                                                )
                total_concat_size += subspace.shape[0] // 2
            elif key == "asset":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], int(subspace.shape[0] * 2))
                total_concat_size += int(subspace.shape[0] * 2)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)
