import copy
from typing import Union, Callable, ClassVar
import numpy as np
from rllab.labcosheduller import CoSheduller
from torch.nn import ReLU, LeakyReLU, Tanh
from customnn.mlpextractor import MlpExtractorNN
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import A2C, PPO, DQN, TD3, DDPG, SAC
from binanceenv import BinanceEnvBase
from binanceenv import BinanceEnvCash
from rllab.labtools import deserialize_kwargs
__version__ = 0.011

#   underscore at the end of class name -> call object itself to get method
lab_serializer: dict = {'learning_rate': {'CoSheduller_': CoSheduller},
                        'MlpExtractorNN': MlpExtractorNN,
                        'ReLU': ReLU,
                        'LeakyReLU': LeakyReLU,
                        'Tanh': Tanh,
                        'action_noise': {'OrnsteinUhlenbeckActionNoise': OrnsteinUhlenbeckActionNoise},
                        'SAC': SAC,
                        'A2C': A2C,
                        'PPO': PPO,
                        'DQN': DQN,
                        'TD3': TD3,
                        'DDPG': DDPG,
                        'BinanceEnvBase': BinanceEnvBase,
                        'BinanceEnvCash': BinanceEnvCash
                        }

if __name__ == '__main__':
    total_timesteps = 16_000_000
    buffer_size = 1_000_000
    learning_start = 750_000
    batch_size = 1024

    action_noise_box = OrnsteinUhlenbeckActionNoise(mean=5e-1 * np.ones(3), sigma=4.99e-1 * np.ones(3), dt=1e-2)

    sac_policy_kwargs = dict(
        features_extractor_class='MlpExtractorNN',
        features_extractor_kwargs=dict(features_dim=256, activation_fn='ReLU'),
        share_features_extractor=True,
        activation_fn='ReLU',
        # net_arch=net_arch,
    )
    sac_kwargs = dict(policy="MlpPolicy",
                      buffer_size=buffer_size,
                      learning_starts=learning_start,
                      policy_kwargs=sac_policy_kwargs,
                      batch_size=batch_size,
                      stats_window_size=100,
                      ent_coef='auto_0.0001',
                      learning_rate={'CoSheduller': dict(warmup=learning_start,
                                                         learning_rate=2e-4,
                                                         min_learning_rate=1e-5,
                                                         total_epochs=total_timesteps,
                                                         epsilon=100)},
                      action_noise={'OrnsteinUhlenbeckActionNoise': dict(mean=5e-1 * np.ones(3),
                                                                         sigma=4.99e-1 * np.ones(3),
                                                                         dt=1e-2)},
                      train_freq=(2, 'step'),
                      target_update_interval=10,  # update target network every 10 _gradient_ steps
                      device="auto",
                      verbose=1)

    print(deserialize_kwargs(sac_kwargs, lab_serializer=lab_serializer))
