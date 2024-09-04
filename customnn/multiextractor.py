import numpy as np

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from rllab.labtools import deserialize_kwargs

__version__ = 0.011


# Swish Function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


nn_serializer: dict = {'ReLU': nn.ReLU,
                       'LeakyReLU': nn.LeakyReLU,
                       'Tanh': nn.Tanh,
                       'Swish': Swish
                       }


class MultiExtractorNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, activation_fn='LeakyReLU'):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1, )
        self.activation_fn = deserialize_kwargs(activation_fn, lab_serializer=nn_serializer)
        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "indicators":
                # indicators_features = subspace.shape[1] * 2
                extractors[key] = nn.Sequential(nn.Conv1d(in_channels=subspace.shape[0],
                                                          out_channels=subspace.shape[0] * 2,
                                                          kernel_size=2),
                                                self.activation_fn(),
                                                nn.Conv1d(in_channels=subspace.shape[0] * 2,
                                                          out_channels=subspace.shape[0] * 4,
                                                          kernel_size=2),
                                                self.activation_fn(),
                                                nn.AdaptiveMaxPool1d(2),
                                                nn.Flatten(-2, -1)
                                                )
                # total_concat_size += indicators_features

            elif key == "ohlc":
                # ohlc_features = 2 * subspace.shape[1]
                extractors[key] = nn.Sequential(nn.Conv1d(in_channels=subspace.shape[0],
                                                          out_channels=subspace.shape[0] * 2,
                                                          kernel_size=2),
                                                self.activation_fn(),
                                                nn.Conv1d(in_channels=subspace.shape[0] * 2,
                                                          out_channels=subspace.shape[0] * 4,
                                                          kernel_size=2),
                                                self.activation_fn(),
                                                nn.AdaptiveMaxPool1d(2),
                                                nn.Flatten(-2, -1)
                                                )
                # total_concat_size += ohlc_features

            elif key == "assets":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], subspace.shape[0] * 2),
                                                self.activation_fn(),
                                                nn.Linear(subspace.shape[0] * 2, subspace.shape[0] * 2),
                                                self.activation_fn(),
                                                )
                # total_concat_size += (subspace.shape[0] * 2)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        # self._features_dim = total_concat_size
        self._features_dim = 1
        # print(total_concat_size)

        # Compute shape by doing one forward pass
        _calc_tensor_list = []
        obs_sample = observation_space.sample()
        with torch.no_grad():
            for key, extractor in self.extractors.items():
                calc_sample = extractor(torch.as_tensor(np.expand_dims(obs_sample[key], axis=0)).float())
                _calc_tensor_list.append(calc_sample)
            cat_t = torch.cat(_calc_tensor_list, dim=-1)
        self._features_dim = cat_t.shape[-1]

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=-1)
