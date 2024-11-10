import torch
import torch.nn as nn

__version__ = 0.020


# Swish Function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


nn_serializer: dict = {'ReLU': nn.ReLU,
                       'LeakyReLU': nn.LeakyReLU,
                       'Tanh': nn.Tanh,
                       'Swish': Swish
                       }